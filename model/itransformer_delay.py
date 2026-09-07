"""iTransformer with component-specific, observed-leader forecast correction.

Identify target(t) ~ source(t-lag) in HISTORY, then use source(T+h-lag)
to correct target forecast h, only when h <= lag. No future labels or source
forecasts are used as evidence. These are associations, not causal estimates.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from model.iTransformer import Model as OriginalITransformer


def _delay_options(configs):
    raw_lags = getattr(configs, 'delay_lags', '0,1,2,4,8,12,24')
    if isinstance(raw_lags, str):
        raw_lags = raw_lags.split(',')
    lags = tuple(sorted(set(int(lag) for lag in raw_lags)))
    if not lags or lags[0] < 0:
        raise ValueError('delay_lags must contain nonnegative integer offsets')
    context_len = getattr(configs, 'delay_context_len', 0)
    if context_len == 0:
        context_len = configs.seq_len - max(lags)
    if context_len < 2 or context_len + max(lags) > configs.seq_len:
        raise ValueError('delay_context_len must be >= 2 and context + max(lags) <= seq_len')
    mode = getattr(configs, 'delay_mode', 'component')
    if mode not in ('component', 'raw', 'off'):
        raise ValueError('delay_mode must be component, raw or off')
    hidden = getattr(configs, 'delay_hidden', 32)
    if hidden < 1:
        raise ValueError('delay_hidden must be positive')
    moving_avg = getattr(configs, 'delay_moving_avg', 25)
    if moving_avg < 1 or moving_avg % 2 == 0:
        raise ValueError('delay_moving_avg must be a positive odd integer')
    gate_init = float(getattr(configs, 'delay_gate_init', 0.02))
    if not math.isfinite(gate_init) or not 0 < gate_init < 1:
        raise ValueError('delay_gate_init must be in (0, 1); use delay_mode=off to disable')
    return dict(mode=mode, lags=lags, context_len=context_len, hidden=hidden,
                moving_avg=moving_avg, gate_init=gate_init)


def delay_setting_suffix(configs):
    options = _delay_options(configs)
    lag_text = '-'.join(map(str, options['lags']))
    return ('_delayv2_observed_{mode}_ma{moving_avg}_l' + lag_text
            + '_c{context_len}_r{hidden}_g{gate_init:g}').format(**options)


class ComponentLagCorrection(nn.Module):
    """Return [B,H,N] corrections using component-specific observed values.

    Correlation preserves explicit time positions; signed ridge slopes transfer
    source changes into target units. Null attention retains the base forecast.
    A sample/target-specific gate controls each component. Missing horizons
    retain the base forecast too. Avoids a [B,N,N,K,H] intermediate tensor.
    """

    def __init__(self, seq_len, d_model, pred_len, *, mode, lags, context_len,
                 hidden, moving_avg, gate_init):
        super().__init__()
        self.mode = mode
        self.context_len = context_len
        self.moving_avg = moving_avg
        self.n_guides = 2 if mode == 'component' else 1
        self.ridge = 1e-4
        self.direct_steps = min(pred_len, max(lags))
        indices = torch.stack([
            torch.arange(seq_len - context_len - lag, seq_len - lag)
            for lag in lags
        ])
        self.register_buffer('lag_indices', indices, persistent=False)
        self.register_buffer('lags', torch.tensor(lags), persistent=False)
        # L-1 is the last observed time T; forecast h starts at 1.
        horizon = torch.arange(1, pred_len + 1)
        lead_indices = seq_len - 1 + horizon[None, :] - self.lags[:, None]
        available = (lead_indices >= 0) & (lead_indices < seq_len)
        self.register_buffer('lead_available', available, persistent=False)
        # Clamp only for safe indexing; the availability mask always removes
        # placeholders before they can affect a prediction.
        self.register_buffer('lead_indices', lead_indices.clamp(0, seq_len - 1),
                             persistent=False)
        self.lag_bias = nn.Parameter(torch.zeros(self.n_guides, len(lags)))
        self.log_temperature = nn.Parameter(torch.full(
            (self.n_guides,), math.log(math.expm1(8.0))))
        self.null_logits = nn.Parameter(torch.zeros(self.n_guides))
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model + 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
            for _ in range(self.n_guides)
        ])
        for gate in self.gates:
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, math.log(gate_init / (1 - gate_init)))
        self.last_summary = None

    def lagged_windows(self, x):
        return x[:, self.lag_indices, :].permute(0, 3, 1, 2)  # B,N,K,W

    def lead_values(self, x):
        return x[:, self.lead_indices, :].permute(0, 3, 1, 2)  # B,N,K,H

    def guide_components(self, x):
        if self.mode != 'component':
            return (x,)
        # One-sided smoothing prevents an old source component from containing
        # observations after that source time; consistent at forecast boundary.
        padded = F.pad(x.transpose(1, 2), (self.moving_avg - 1, 0), mode='replicate')
        trend = F.avg_pool1d(padded, self.moving_avg, stride=1).transpose(1, 2)
        return trend, x - trend

    def relation_weights(self, component, index):
        """Estimate weights, signed slopes and null mass from observed history."""
        target = component[:, -self.context_len:, :].transpose(1, 2)
        source = self.lagged_windows(component)
        q = target - target.mean(-1, keepdim=True)
        k = source - source.mean(-1, keepdim=True)
        target_var = q.square().mean(-1)
        source_var = k.square().mean(-1)
        covariance = torch.einsum('biw,bjkw->bijk', q, k) / self.context_len
        denominator = (target_var[:, :, None, None] * source_var[:, None]).clamp_min(
            self.ridge ** 2).sqrt()
        correlation = (covariance / denominator).clamp(-1, 1)
        slopes = covariance / (source_var[:, None] + self.ridge)
        # Prevent large transfers from nearly constant source histories.
        slopes = slopes.clamp(-3, 3)
        n_variables = component.size(-1)
        valid = (~torch.eye(n_variables, device=component.device, dtype=torch.bool))[
            None, :, :, None]
        valid = valid & (self.lags[None, None, None, :] > 0)
        valid = valid & (target_var[:, :, None, None] > self.ridge)
        valid = valid & (source_var[:, None] > self.ridge)
        temperature = F.softplus(self.log_temperature[index]) + 1e-4
        scores = temperature * (correlation.abs() - 0.5) + self.lag_bias[index]
        # Candidate-count correction prevents many weak relations from beating
        # the single null expert merely because there are more of them.
        count = valid.sum((-1, -2), keepdim=True).clamp_min(1)
        scores = (scores - count.to(scores.dtype).log()).masked_fill(~valid, float('-inf'))
        flat = scores.flatten(2)
        null = self.null_logits[index].expand(*flat.shape[:2], 1)
        probabilities = torch.softmax(torch.cat((flat, null), dim=-1), dim=-1)
        weights = probabilities[..., :-1].reshape_as(scores)
        return weights, slopes, probabilities[..., -1], correlation

    def forward(self, x, variable_tokens, base_prediction, output_attention=False):
        # Keep moments in float32 even with AMP and nearly flat components.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._correct(x.float(), variable_tokens, base_prediction, output_attention)

    def _correct(self, x, variable_tokens, base_prediction, output_attention):
        batch, _, n_variables = x.shape
        correction = torch.zeros_like(base_prediction, dtype=torch.float32)
        weights_list, null_list, gate_list, mass_list = [], [], [], []
        if self.mode != 'off' and n_variables > 1:
            history_components = self.guide_components(x)
            # BASE predictions are decomposed solely to define the fallback.
            # Relation discovery and source values use history_components only.
            base_components = self.guide_components(torch.cat((x, base_prediction.float()), dim=1))
            availability = self.lead_available.to(x.dtype)
            for index, component in enumerate(history_components):
                weights, slopes, null_mass, correlation = self.relation_weights(component, index)
                source_anchor = self.lagged_windows(component)[..., -1]
                source_change = (self.lead_values(component) - source_anchor[..., None]) * availability
                mass = torch.einsum('bijk,kh->bih', weights, availability)
                # Expert: target(T) + slope * [source(T+h-lag) - source(T-lag)].
                message = torch.einsum('bijk,bjkh->bih', weights * slopes, source_change)
                message = message + component[:, -1, :, None] * mass
                fallback = base_components[index][:, -base_prediction.size(1):].transpose(1, 2)
                strength = (weights * correlation.abs()).sum((-1, -2))
                features = torch.cat((variable_tokens.float(), strength[..., None],
                                      (1 - null_mass)[..., None]), dim=-1)
                gate = torch.sigmoid(self.gates[index](features))
                delta = gate * (message - mass * fallback)
                correction = correction + delta.transpose(1, 2)
                weights_list.append(weights.detach())
                null_list.append(null_mass.detach())
                gate_list.append(gate.squeeze(-1).detach())
                mass_list.append(mass.detach())

        if weights_list:
            gates = torch.stack(gate_list, 1)
            null_mass = torch.stack(null_list, 1)
            known_mass = torch.stack(mass_list, 1)
        else:
            gates = x.new_zeros(batch, self.n_guides, n_variables)
            null_mass = torch.ones_like(gates)
            known_mass = x.new_zeros(batch, self.n_guides, n_variables, base_prediction.size(1))
        correction_rms = correction.detach().square().mean(1).sqrt()
        prediction_rms = base_prediction.detach().float().square().mean(1).sqrt()
        self.last_summary = {
            'gate_mean': gates.mean(), 'null_mass': null_mass.mean(),
            'known_mass': known_mass.mean(),
            'correction_rms': correction_rms.mean(), 'base_rms': prediction_rms.mean(),
            # Prefix metrics avoid mistaking long-horizon dilution for an
            # inactive adapter; e.g. only 24 of 720 steps have direct evidence.
            'direct_steps': x.new_tensor(float(self.direct_steps)),
            'known_mass_supported': (known_mass[..., :self.direct_steps].mean()
                                     if self.direct_steps else x.new_zeros(())),
            'correction_rms_supported': (
                correction.detach()[:, :self.direct_steps].square().mean(1).sqrt().mean()
                if self.direct_steps else x.new_zeros(())),
        }
        diagnostics = None
        if output_attention:
            weights = (torch.stack(weights_list, 1) if weights_list else
                       x.new_zeros(batch, self.n_guides, n_variables, n_variables, self.lags.numel()))
            diagnostics = {
                'weights': weights, 'lags': self.lags.unsqueeze(0).expand(batch, -1),
                'gates': gates, 'null_mass': null_mass, 'known_mass': known_mass,
                'available': self.lead_available.unsqueeze(0).expand(batch, -1, -1),
                'correction_rms': correction_rms, 'prediction_rms': prediction_rms,
            }
        return correction.to(base_prediction.dtype), diagnostics


class Model(OriginalITransformer):
    """Original backbone forward path followed by observed-leader correction.

    Jointly trained: preserving the path does not freeze backbone weights.
    mode=off matches the original with matching weights.
    v1 adapter checkpoints are intentionally incompatible with v2.
    """

    def __init__(self, configs):
        options = _delay_options(configs)
        super().__init__(configs)
        with torch.random.fork_rng(devices=[]):
            self.delay_correction = ComponentLagCorrection(
                configs.seq_len, configs.d_model, configs.pred_len, **options)

    @property
    def delay_metrics(self):
        return self.delay_correction.last_summary

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len or x_enc.size(2) < 1:
            raise ValueError('x_enc must be [batch, seq_len, variables] with variables > 0')
        if x_mark_enc is not None and (
            x_mark_enc.ndim != 3 or x_mark_enc.shape[:2] != x_enc.shape[:2]
        ):
            raise ValueError('x_mark_enc must match the batch/time dimensions of x_enc')
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(centered.var(1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = centered / stdev

        n_variables = x_enc.size(-1)
        tokens = self.enc_embedding(x_enc, x_mark_enc)
        encoded, attention = self.encoder(tokens, attn_mask=None)
        base_prediction = self.projector(encoded).transpose(1, 2)[:, :, :n_variables]
        correction, delay_attention = self.delay_correction(
            x_enc, encoded[:, :n_variables], base_prediction, self.output_attention)
        prediction = base_prediction + correction
        if self.use_norm:
            prediction = prediction * stdev + means
        diagnostics = None
        if self.output_attention:
            diagnostics = {'backbone_attention': attention, 'delay': delay_attention}
        return prediction, diagnostics

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        prediction, diagnostics = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return (prediction, diagnostics) if self.output_attention else prediction
