"""Separate cross-variate modeling of trend and seasonal/residual components.

Normalize history once, decompose it along time, and give each component its
own iTransformer embedding, encoder and prediction head. Add the forecasts in
normalized space before restoring the original history's scale and mean.

The moving-average decomposition follows the standard construction also used
by the supplied Ister reference. Unlike its CD_Ister trend MLP, BOTH branches
here use the repository's original full cross-variate attention. There is no
runtime dependency on the reference repository and no auxiliary loss.

"Seasonal" denotes the moving-average residual, which can also contain noise;
neither the decomposition nor the output heads enforce exact periodicity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class MovingAverageDecomposition(nn.Module):
    """Return (trend, residual) for [batch, time, variables] history only."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError('decomp_moving_avg must be a positive odd integer')

    def forward(self, x):
        if x.ndim != 3 or x.size(1) == 0 or x.size(2) == 0:
            raise ValueError('Expected nonempty history shaped [batch, time, variables]')
        # Replicate the observed endpoints; never pad with future targets.
        half_window = self.kernel_size // 2
        padded = F.pad(x.transpose(1, 2), (half_window, half_window), mode='replicate')
        trend = F.avg_pool1d(padded, self.kernel_size, stride=1).transpose(1, 2)
        return trend, x - trend


class ComponentTransformer(nn.Module):
    """One complete iTransformer operating on one type of component."""

    def __init__(self, configs):
        super().__init__()
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, component, x_mark_enc):
        n_variables = component.size(-1)
        # [B, L, N] -> [B, N (+ timestamp tokens), D]. Each token is a
        # variable's full trend/residual history, not an individual time step.
        tokens = self.enc_embedding(component, x_mark_enc)
        encoded, attention = self.encoder(tokens, attn_mask=None)
        prediction = self.projector(encoded).transpose(1, 2)[:, :, :n_variables]
        return prediction, attention


class Model(nn.Module):
    """Two independent inverted Transformers trained by the sum's forecast MSE.

    Common d_model/d_ff/e_layers/n_heads settings apply to EACH branch, so
    equal settings roughly double the original iTransformer parameter count.
    Decoder inputs are unused, as in the original encoder-only iTransformer.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        if self.seq_len < 1 or self.pred_len < 1 or configs.e_layers < 1:
            raise ValueError('seq_len, pred_len and e_layers must be positive')
        if configs.n_heads < 1 or configs.d_model < 1 or configs.d_model % configs.n_heads:
            raise ValueError('d_model must be positive and divisible by n_heads')

        self.decomposition = MovingAverageDecomposition(
            getattr(configs, 'decomp_moving_avg', 25)
        )
        self.trend_branch = ComponentTransformer(configs)
        self.seasonal_branch = ComponentTransformer(configs)

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len:
            raise ValueError('x_enc must have shape [batch, seq_len, variables]')
        if x_mark_enc is not None and (
            x_mark_enc.ndim != 3 or x_mark_enc.shape[:2] != x_enc.shape[:2]
        ):
            raise ValueError('x_mark_enc must match the batch and time dimensions of x_enc')

        if self.use_norm:
            # Match original iTransformer statistics, computed ONLY on history.
            means = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            normalized = centered / stdev
        else:
            means = x_enc.new_zeros(x_enc.size(0), 1, x_enc.size(2))
            stdev = torch.ones_like(means)
            normalized = x_enc

        trend, seasonal = self.decomposition(normalized)
        # No separate normalization: residual amplitude and trend level remain
        # on a common scale and neither component's statistics are discarded.
        trend_pred, trend_attention = self.trend_branch(trend, x_mark_enc)
        seasonal_pred, seasonal_attention = self.seasonal_branch(seasonal, x_mark_enc)
        prediction = (trend_pred + seasonal_pred) * stdev + means

        diagnostics = None
        if self.output_attention:
            # Assign the original mean to trend for an additive visualization.
            # These are learned contributions, not separately supervised labels.
            diagnostics = {
                'trend_attention': trend_attention,
                'seasonal_attention': seasonal_attention,
                'trend_prediction': (trend_pred * stdev + means).detach(),
                'seasonal_prediction': (seasonal_pred * stdev).detach(),
            }
        return prediction, diagnostics

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        prediction, diagnostics = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return prediction, diagnostics
        return prediction  # [B, pred_len, N]; the runner selects the MS target.
