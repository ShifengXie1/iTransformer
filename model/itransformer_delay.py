"""iTransformer with component-guided, cross-variate lag corrections.

The original embedding, encoder and single prediction head are retained.
Trend/residual histories guide separate lag attentions; BOTH attentions read
values from the original normalized history. There are no component forecast
heads or auxiliary losses. Positive lag means an earlier source segment.

Centered moving averages describe associations between smoothed histories,
not causal delays. No decoder values or future labels enter the correction.
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
    seq_len = configs.seq_len
    context_len = getattr(configs, 'delay_context_len', 0)
    if context_len == 0:
        context_len = seq_len - max(lags)
    if context_len < 1 or context_len + max(lags) > seq_len:
        raise ValueError('delay_context_len + max(delay_lags) must not exceed seq_len')
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
    if not math.isfinite(gate_init) or not 0 <= gate_init < 1:
        raise ValueError('delay_gate_init must be finite and in [0, 1)')
    return dict(mode=mode, lags=lags, context_len=context_len, hidden=hidden,
                moving_avg=moving_avg, gate_init=gate_init)


def delay_setting_suffix(configs):
    """Use the same effective configuration for train/test checkpoint names."""
    options = _delay_options(configs)
    lag_text = '-'.join(map(str, options['lags']))
    return ('_delayv1_{mode}_ma{moving_avg}_l' + lag_text
            + '_c{context_len}_r{hidden}_g{gate_init:g}').format(**options)


class ComponentLagCorrection(nn.Module):
    """Return a [B, N, D] correction for physical variable tokens only.

    Attention axes are [batch, guide, target, source, lag]. Component mode
    has two guides (trend, residual); raw mode has one. Source == target is
    masked out: the unchanged iTransformer path carries each variable's own
    information. A univariate input consequently receives zero correction.
    """

    def __init__(self, seq_len, d_model, *, mode, lags, context_len, hidden,
                 moving_avg, gate_init):
        super().__init__()
        self.mode = mode
        self.context_len = context_len
        self.moving_avg = moving_avg
        self.hidden = hidden
        self.n_guides = 2 if mode == 'component' else 1
        # Same-length, non-wrapping windows. With L=96, W=72, lag=4,
        # compare target[24:96] against source[20:92].
        indices = torch.stack([
            torch.arange(seq_len - context_len - lag, seq_len - lag)
            for lag in lags
        ])
        self.register_buffer('lag_indices', indices, persistent=False)
        self.register_buffer('lags', torch.tensor(lags), persistent=False)
        self.queries = nn.ModuleList([
            nn.Linear(context_len, hidden, bias=False) for _ in range(self.n_guides)
        ])
        self.keys = nn.ModuleList([
            nn.Linear(context_len, hidden, bias=False) for _ in range(self.n_guides)
        ])
        # Share raw-value encoding/output projection across guides and lags.
        self.values = nn.Linear(context_len, hidden, bias=False)
        self.output_projection = nn.Linear(hidden, d_model, bias=False)
        self.lag_bias = nn.Parameter(torch.zeros(self.n_guides, len(lags)))
        self.gate_logits = nn.Parameter(torch.full(
            (self.n_guides,), math.atanh(gate_init)
        ))

    def lagged_windows(self, x):
        """[B, L, N] -> [B, N, K, W], with K candidate historical offsets."""
        return x[:, self.lag_indices, :].permute(0, 3, 1, 2)

    def guide_components(self, x):
        if self.mode != 'component':
            return (x,)
        pad = self.moving_avg // 2
        padded = F.pad(x.transpose(1, 2), (pad, pad), mode='replicate')
        trend = F.avg_pool1d(padded, self.moving_avg, stride=1).transpose(1, 2)
        return trend, x - trend

    def forward(self, x, variable_tokens, output_attention=False):
        batch, _, n_variables = x.shape
        gates = torch.tanh(self.gate_logits)
        correction = torch.zeros_like(variable_tokens)
        attentions = []
        if self.mode != 'off' and n_variables > 1:
            # All guides aggregate RAW history values, never component forecasts.
            values = self.values(self.lagged_windows(x))
            self_mask = torch.eye(n_variables, device=x.device, dtype=torch.bool)
            self_mask = self_mask.view(1, n_variables, n_variables, 1)
            for index, component in enumerate(self.guide_components(x)):
                target = component[:, -self.context_len:, :].transpose(1, 2)
                query = self.queries[index](target)
                key = self.keys[index](self.lagged_windows(component))
                scores = torch.einsum('bir,bjkr->bijk', query, key) / math.sqrt(self.hidden)
                scores = scores + self.lag_bias[index]
                scores = scores.masked_fill(self_mask, float('-inf'))
                # Joint selection of a source variable and its historical offset.
                weights = torch.softmax(scores.flatten(2), dim=-1).reshape_as(scores)
                message = torch.einsum('bijk,bjkr->bir', weights, values)
                correction = correction + gates[index] * self.output_projection(message)
                if output_attention:
                    attentions.append(weights.detach())

        diagnostics = None
        if output_attention:
            if attentions:
                weights = torch.stack(attentions, dim=1)
            else:
                # Avoid all-masked softmax for N=1; preserve diagnostic shapes.
                weights = x.new_zeros(batch, self.n_guides, n_variables,
                                      n_variables, self.lags.numel())
            diagnostics = {
                'weights': weights,
                'lags': self.lags.unsqueeze(0).expand(batch, -1),
                'gates': gates.detach().unsqueeze(0).expand(batch, -1),
                'correction_rms': correction.detach().float().square().mean(-1).sqrt(),
                'token_rms': variable_tokens.detach().float().square().mean(-1).sqrt(),
            }
        return correction, diagnostics


class Model(OriginalITransformer):
    """Original iTransformer backbone plus a small component-guided adapter.

    Backbone state_dict keys match model/iTransformer.py. With matching weights
    and delay_mode='off' (or all gates zero), forecasts match the original.
    No separate training objective or decoder input is required.
    """

    def __init__(self, configs):
        options = _delay_options(configs)
        super().__init__(configs)
        # Additional parameter initialization must not consume the random stream
        # used by the existing training loader/backbone dropout. CPU init only.
        with torch.random.fork_rng(devices=[]):
            self.delay_correction = ComponentLagCorrection(
                configs.seq_len, configs.d_model, **options
            )

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
        correction, delay_attention = self.delay_correction(
            x_enc, tokens[:, :n_variables], self.output_attention
        )
        # Timestamp tokens retain their original embedding and are not lag sources.
        tokens = torch.cat((tokens[:, :n_variables] + correction,
                            tokens[:, n_variables:]), dim=1)
        encoded, attention = self.encoder(tokens, attn_mask=None)
        prediction = self.projector(encoded).transpose(1, 2)[:, :, :n_variables]
        if self.use_norm:
            prediction = prediction * stdev + means
        diagnostics = None
        if self.output_attention:
            diagnostics = {'backbone_attention': attention, 'delay': delay_attention}
        return prediction, diagnostics

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        prediction, diagnostics = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return (prediction, diagnostics) if self.output_attention else prediction
