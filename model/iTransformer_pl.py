"""iTransformer with period patches and component-specific Fourier phase lags."""

import hashlib
import json

import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_inverted
from layers.Period_Lag import (
    AdaptiveComponentFusion,
    ComponentLagAttention,
    ComponentLagEstimator,
    ComponentPatchEncoder,
    DominantPeriodEstimator,
    FrequencyComponentDecomposer,
    PeriodPatchEmbedding,
)
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


def _period_lag_options(configs):
    period = int(getattr(configs, 'period', 24))
    max_period = getattr(configs, 'max_period', None)
    max_lag = getattr(configs, 'max_lag', None)
    return dict(
        period_mode=getattr(configs, 'period_mode', 'fixed'),
        period=period,
        min_period=int(getattr(configs, 'min_period', 4)),
        max_period=int(configs.seq_len if max_period is None else max_period),
        period_sigma=float(getattr(configs, 'period_sigma', 1.5)),
        harmonic_max_bin=int(getattr(configs, 'harmonic_max_bin', 4)),
        max_lag=float(period if max_lag is None else max_lag),
        lag_alpha=float(getattr(configs, 'lag_alpha', 0.1)),
        lag_beta=float(getattr(configs, 'lag_beta', 0.1)),
        component_temporal_mixer=getattr(configs, 'component_temporal_mixer', 'conv'),
        use_adaptive_fusion=bool(getattr(configs, 'use_adaptive_fusion', True)),
    )


def period_lag_setting_suffix(configs):
    """Keep different period/lag ablations from overwriting the same checkpoint."""
    options = _period_lag_options(configs)
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    return '_pl_{}_p{}_{}'.format(options['period_mode'], options['period'], digest)


class Model(nn.Module):
    """One variable = one token, with four frequency-specific relation branches.

    In FFT mode, P is shared across the current batch on each device, so the
    estimate can depend on batch composition. ``last_period`` exposes that P
    for debugging a single-device run. Fixed mode is batch-independent.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.output_attention = bool(getattr(configs, 'output_attention', False))
        self.use_norm = bool(getattr(configs, 'use_norm', True))
        self.class_strategy = getattr(configs, 'class_strategy', 'projection')
        d_model = int(configs.d_model)
        n_heads = int(configs.n_heads)
        if self.seq_len < 1 or self.pred_len < 1:
            raise ValueError('seq_len and pred_len must be positive')
        if n_heads < 1 or d_model < n_heads:
            raise ValueError('d_model must be at least n_heads, and n_heads must be positive')
        if configs.e_layers < 1:
            raise ValueError('e_layers must be positive')
        options = _period_lag_options(configs)
        self.period_estimator = DominantPeriodEstimator(
            self.seq_len, mode=options['period_mode'], period=options['period'],
            min_period=options['min_period'], max_period=options['max_period'],
            period_sigma=options['period_sigma'],
        )
        self.patch_embedding = PeriodPatchEmbedding()
        self.decomposer = FrequencyComponentDecomposer(options['harmonic_max_bin'])
        self.component_encoders = nn.ModuleList([
            ComponentPatchEncoder(options['period'], d_model, configs.dropout,
                                  options['component_temporal_mixer'])
            for _ in range(4)
        ])
        self.lag_estimator = ComponentLagEstimator()
        self.component_attention = ComponentLagAttention(
            d_model, n_heads, configs.d_ff, dropout=configs.dropout,
            activation=configs.activation, max_lag=options['max_lag'],
            lag_alpha=options['lag_alpha'], lag_beta=options['lag_beta'],
            output_attention=self.output_attention,
        )
        self.fusion = AdaptiveComponentFusion(d_model, options['use_adaptive_fusion'])
        # Preserve the original optional timestamp-token path after fusion.
        # Calendar covariates do not participate in period or lag estimation.
        self.mark_embedding = DataEmbedding_inverted(
            self.seq_len, d_model, getattr(configs, 'embed', 'timeF'),
            getattr(configs, 'freq', 'h'), configs.dropout,
        )
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=self.output_attention), d_model, n_heads,
                ),
                d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)
        ], norm_layer=nn.LayerNorm(d_model))
        self.projector = nn.Linear(d_model, self.pred_len)
        self.last_period = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, N]. Decoder inputs are unused, as in iTransformer.
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len:
            raise ValueError('x_enc must have shape [B, seq_len, N]')
        batch, _, variables = x_enc.shape
        if batch < 1 or variables < 1:
            raise ValueError('x_enc must contain at least one sample and variable')
        input_dtype = x_enc.dtype
        # Same mean/variance normalization as iTransformer, with float32
        # statistics to protect half/bfloat16 inputs from variance overflow.
        with torch.autocast(device_type=x_enc.device.type, enabled=False):
            x = x_enc.float()
            if self.use_norm:
                means = x.mean(1, keepdim=True).detach()
                x = x - means
                stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                x = x / stdev

        period = self.period_estimator(x)
        self.last_period = period
        patches, coverage = self.patch_embedding(x, period)  # [B, N, M, P]
        components, spec, masks = self.decomposer(patches)  # [B, C=4, N, M, P]
        # Per-component patch embeddings [B, N, M, D] are pooled over M.
        tokens = torch.stack([
            encoder(components[:, c], coverage)
            for c, encoder in enumerate(self.component_encoders)
        ], dim=1)  # [B, C, N, D]
        lag, strength = self.lag_estimator(spec, masks, period, coverage)  # [B, C, N, N]
        tokens, component_attns = self.component_attention(
            tokens, lag, strength, period,
        )  # [B, C, N, D]
        fused, fusion_weights = self.fusion(tokens)  # [B, N, D], [B, N, C]

        if x_mark_enc is not None:
            if x_mark_enc.ndim != 3 or x_mark_enc.shape[:2] != x_enc.shape[:2]:
                raise ValueError('x_mark_enc must have shape [B, seq_len, K]')
            if x_mark_enc.shape[-1]:
                mark_tokens = self.mark_embedding(
                    x_mark_enc.to(self.mark_embedding.value_embedding.weight.dtype), None,
                )  # [B, K, D]
                fused = torch.cat((fused, mark_tokens.to(fused.dtype)), dim=1)
        # [B, N (+ K optional covariates), D]; attention remains over variables.
        encoded, encoder_attns = self.encoder(fused, attn_mask=None)
        forecast = self.projector(encoded[:, :variables]).transpose(1, 2)  # [B, pred_len, N]
        if self.use_norm:
            forecast = forecast.float() * stdev + means  # Broadcast [B, 1, N].
        forecast = forecast.to(input_dtype)

        attentions = None
        if self.output_attention:
            # All diagnostic values are tensors/lists, compatible with DataParallel.
            attentions = dict(
                component=component_attns, encoder=encoder_attns,
                lag=lag, strength=strength, fusion_weights=fusion_weights,
                period=lag.new_full((batch,), period),
            )
        return forecast, attentions

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # mask is accepted for interface compatibility, as in the original model.
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, N]
