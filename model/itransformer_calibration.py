"""Context-conditioned low-rank calibration after the original iTransformer.

The inherited forward keeps its original signature and attention return value.
Only forecast is extended, after the backbone's existing de-normalization.
No label fitting, auxiliary loss or additional forecasting head is required.
"""

import math

import torch
from torch import nn

from model.iTransformer import Model as OriginalITransformer


class HorizonChannelCalibrator(nn.Module):
    """Generate scale, bias and gate from history and horizon factors.

    [B,L,N] history -> [B,N,5] statistics -> [B,N,R] context features
    -> three [B,N,R] channel factors. Each contracts with a learned [T,R]
    horizon factor to produce [B,T,N]. Parameters are independent of N.

    Bias is expressed directly in the same units as base_pred. Disabling
    channel context uses a shared zero context, with no hidden dependency on
    sample statistics. Disabling horizon factors learns one shared row.
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(getattr(configs, 'pred_len', 96))
        self.rank = int(getattr(configs, 'calibration_rank', 8))
        hidden = int(getattr(configs, 'calibration_hidden', 32))
        dropout = float(getattr(configs, 'calibration_dropout', 0.0))
        self.trend_window = int(getattr(configs, 'calibration_trend_window', 24))
        self.scale_limit = float(getattr(configs, 'calibration_scale_limit', 0.2))
        self.use_horizon_factor = bool(getattr(configs, 'use_horizon_factor', True))
        self.use_channel_context = bool(getattr(configs, 'use_channel_context', True))
        self.use_gate = bool(getattr(configs, 'use_gate', True))
        if min(self.pred_len, self.rank, hidden, self.trend_window) < 1:
            raise ValueError('pred_len, calibration_rank, calibration_hidden and '
                             'calibration_trend_window must be positive')
        if not 0 <= dropout < 1:
            raise ValueError('calibration_dropout must be in [0, 1)')
        if not math.isfinite(self.scale_limit) or self.scale_limit < 0:
            raise ValueError('calibration_scale_limit must be finite and nonnegative')

        self.context_encoder = nn.Sequential(
            nn.Linear(5, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, self.rank),
        )
        self.scale_channel_proj = nn.Linear(self.rank, self.rank)
        self.bias_channel_proj = nn.Linear(self.rank, self.rank)
        self.gate_channel_proj = nn.Linear(self.rank, self.rank)
        horizon_rows = self.pred_len if self.use_horizon_factor else 1
        self.horizon_scale = nn.Parameter(torch.empty(horizon_rows, self.rank))
        self.horizon_bias = nn.Parameter(torch.empty(horizon_rows, self.rank))
        self.horizon_gate = nn.Parameter(torch.empty(horizon_rows, self.rank))
        self.gate_offset = nn.Parameter(torch.tensor(-3.0))

        # Zero only ONE side of the scale/bias products. Nonzero horizon
        # factors let their channel projections receive gradients immediately.
        for projection in (self.scale_channel_proj, self.bias_channel_proj):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.normal_(self.horizon_scale, std=self.rank ** -0.5)
        nn.init.normal_(self.horizon_bias, std=self.rank ** -0.5)
        nn.init.normal_(self.horizon_gate, std=0.02)
        # Constant small initial gate, with nonzero factors on its horizon side.
        nn.init.zeros_(self.gate_channel_proj.weight)
        nn.init.zeros_(self.gate_channel_proj.bias)

    def context_statistics(self, x_enc):
        """Return [last, mean, std, recent mean difference, delta], [B,N,5]."""
        if x_enc.ndim != 3 or x_enc.size(1) < 1 or x_enc.size(2) < 1:
            raise ValueError('x_enc must be [B,L,N] with L and N positive')
        # Population variance handles L=1; epsilon keeps flat-window gradients
        # finite. These are the input units, before backbone normalization.
        with torch.autocast(device_type=x_enc.device.type, enabled=False):
            x = x_enc.float()
            last = x[:, -1, :]
            mean = x.mean(dim=1)
            std = (x.var(dim=1, unbiased=False) + 1e-5).sqrt()
            delta = last - x[:, 0, :]
            if x.size(1) == 1:
                trend = torch.zeros_like(last)
            else:
                k = min(self.trend_window, x.size(1) - 1)
                recent = x[:, -(k + 1):, :]
                trend = (recent[:, 1:, :] - recent[:, :-1, :]).mean(dim=1)
            return torch.stack((last, mean, std, trend, delta), dim=-1)

    def _factorize(self, horizon, channel):
        values = torch.einsum('tr,bnr->btn', horizon, channel)
        return values.expand(-1, self.pred_len, -1)

    def calibration_parameters(self, x_enc):
        """Return scale, bias, gate, each [B,T,N], for inspection or ablations."""
        if x_enc.ndim != 3 or x_enc.size(1) < 1 or x_enc.size(2) < 1:
            raise ValueError('x_enc must be [B,L,N] with L and N positive')
        # The model uses float32 parameters, including during normal AMP
        # training; keep statistics, factor products and gating in float32.
        with torch.autocast(device_type=x_enc.device.type, enabled=False):
            if self.use_channel_context:
                context = self.context_statistics(x_enc)
            else:
                context = torch.zeros(x_enc.size(0), x_enc.size(2), 5,
                                      device=x_enc.device, dtype=torch.float32)
            features = self.context_encoder(context)
            scale_raw = self._factorize(self.horizon_scale, self.scale_channel_proj(features))
            bias = self._factorize(self.horizon_bias, self.bias_channel_proj(features))
            scale = self.scale_limit * torch.tanh(scale_raw)
            if self.use_gate:
                gate_raw = self._factorize(self.horizon_gate, self.gate_channel_proj(features))
                gate = torch.sigmoid(gate_raw + self.gate_offset)
            else:
                gate = torch.ones_like(scale)
            return scale, bias, gate

    def forward(self, base_pred, x_enc):
        if (base_pred.ndim != 3 or x_enc.ndim != 3
                or base_pred.size(1) != self.pred_len
                or base_pred.size(0) != x_enc.size(0)
                or base_pred.size(2) != x_enc.size(2)):
            raise ValueError('base_pred must be [B,pred_len,N] matching x_enc [B,L,N]')
        with torch.autocast(device_type=base_pred.device.type, enabled=False):
            scale, bias, gate = self.calibration_parameters(x_enc)
            base = base_pred.float()
            correction = gate * (scale * base + bias)
            return (base + correction).to(dtype=base_pred.dtype)


class Model(OriginalITransformer):
    """Original forecast plus optional calibration; original forward inherited."""

    def __init__(self, configs):
        super().__init__(configs)
        self.use_calibration = bool(getattr(configs, 'use_calibration', True))
        self.calibrator = HorizonChannelCalibrator(configs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        base_pred, attention = super().forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # Original forecast rebinds its normalized x_enc locally, so this x_enc
        # still holds the untouched input in base_pred's de-normalized units.
        if self.use_calibration:
            base_pred = self.calibrator(base_pred, x_enc)
        return base_pred, attention


def calibration_setting_suffix(configs):
    """Keep calibration ablation checkpoints distinct in training and testing."""
    return '_cal{}_r{}_h{}_do{}_tw{}_sl{}_hf{}_cc{}_g{}'.format(
        int(getattr(configs, 'use_calibration', True)),
        getattr(configs, 'calibration_rank', 8),
        getattr(configs, 'calibration_hidden', 32),
        getattr(configs, 'calibration_dropout', 0.0),
        getattr(configs, 'calibration_trend_window', 24),
        getattr(configs, 'calibration_scale_limit', 0.2),
        int(getattr(configs, 'use_horizon_factor', True)),
        int(getattr(configs, 'use_channel_context', True)),
        int(getattr(configs, 'use_gate', True)))
