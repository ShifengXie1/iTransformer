import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class ChannelIndependentPatchTST(nn.Module):
    """PatchTST branch with shared weights and no cross-variate attention."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.d_model = int(configs.d_model)
        self.output_attention = bool(configs.output_attention)

        self.patch_len = int(getattr(configs, 'three_patch_len', 16))
        self.stride = int(getattr(configs, 'three_stride', 8))
        if not 1 <= self.patch_len <= self.seq_len:
            raise ValueError(
                f'three_patch_len must be in [1, {self.seq_len}], '
                f'got {self.patch_len}'
            )
        if self.stride < 1:
            raise ValueError('three_stride must be at least 1')

        # Replicating one stride at the forecasting boundary follows the
        # PatchTST "padding_patch=end" convention and retains the latest data.
        self.patch_num = (
            (self.seq_len + self.stride - self.patch_len) // self.stride + 1
        )
        self.patch_projection = nn.Linear(self.patch_len, self.d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 1, self.patch_num, self.d_model)
        )
        self.input_dropout = nn.Dropout(configs.dropout)
        nn.init.normal_(self.position_embedding, std=0.02)

        patch_layers = max(
            1, int(getattr(configs, 'three_patch_layers', configs.e_layers))
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        self.d_model,
                        configs.n_heads,
                    ),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(patch_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )
        self.head_dropout = nn.Dropout(
            float(getattr(configs, 'three_head_dropout', configs.dropout))
        )
        self.head = nn.Linear(
            self.patch_num * self.d_model, self.pred_len, bias=True
        )

    def forward(self, x):
        # x: [B, L, C]. Folding C into the batch dimension is what makes the
        # entire PatchTST encoder strictly channel-independent.
        batch_size, seq_len, n_vars = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f'Expected seq_len={self.seq_len}, got {seq_len}')

        channel_series = x.permute(0, 2, 1)
        channel_series = F.pad(
            channel_series, (0, self.stride), mode='replicate'
        )
        patches = channel_series.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )
        if patches.shape[2] != self.patch_num:
            raise RuntimeError(
                f'Expected {self.patch_num} patches, got {patches.shape[2]}'
            )

        tokens = self.patch_projection(patches)
        tokens = self.input_dropout(tokens + self.position_embedding)
        tokens = tokens.reshape(
            batch_size * n_vars, self.patch_num, self.d_model
        )
        encoded, attentions = self.encoder(tokens, attn_mask=None)

        prediction = self.head(
            self.head_dropout(encoded.reshape(batch_size * n_vars, -1))
        )
        prediction = prediction.reshape(
            batch_size, n_vars, self.pred_len
        ).permute(0, 2, 1)
        state = encoded.mean(dim=1).reshape(batch_size, n_vars, self.d_model)

        if self.output_attention:
            attentions = [
                None if attention is None else attention.reshape(
                    batch_size, n_vars, *attention.shape[1:]
                )
                for attention in attentions
            ]
        return prediction, state, attentions


class ITransformerBranch(nn.Module):
    """Joint historical-space modeling with one token per variable."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(configs.pred_len)
        self.embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(max(1, int(configs.e_layers)))
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(
            configs.d_model, self.pred_len, bias=True
        )

    def forward(self, x):
        # Time features are deliberately omitted here: the token axis must
        # contain exactly the variables described by x.
        tokens = self.embedding(x, None)
        tokens, attentions = self.encoder(tokens, attn_mask=None)
        prediction = self.projector(tokens).permute(0, 2, 1)
        return prediction, tokens, attentions


class DynamicForecastFusion(nn.Module):
    """Produce a convex, sample/variable/horizon-specific branch mixture."""

    def __init__(self, d_model, pred_len, hidden_size, dropout):
        super().__init__()
        self.pred_len = int(pred_len)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2 * self.pred_len),
        )
        # Both forecasters contribute equally before the gate has learned a
        # preference, which avoids suppressing either branch at initialization.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, patch_prediction, joint_prediction,
                patch_state, joint_state):
        batch_size, n_vars, _ = patch_state.shape
        logits = self.gate(torch.cat([patch_state, joint_state], dim=-1))
        logits = logits.reshape(batch_size, n_vars, self.pred_len, 2)
        weights = torch.softmax(logits, dim=-1).permute(0, 2, 1, 3)
        candidates = torch.stack(
            [patch_prediction, joint_prediction], dim=-1
        )
        base_prediction = (weights * candidates).sum(dim=-1)
        return base_prediction, weights


class MaskedPredictionCrossAttention(nn.Module):
    """Correct each variable using values from other prediction tokens only."""

    def __init__(self, n_vars, pred_len, d_model, n_heads, d_ff,
                 dropout, activation, output_attention):
        super().__init__()
        self.n_vars = int(n_vars)
        self.output_attention = bool(output_attention)
        self.prediction_embedding = nn.Linear(pred_len, d_model)
        self.variate_embedding = nn.Embedding(self.n_vars, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.residual_projector = nn.Linear(d_model, pred_len, bias=True)

        # Start with a small correction while keeping every stage trainable.
        nn.init.xavier_uniform_(self.residual_projector.weight, gain=0.1)
        nn.init.zeros_(self.residual_projector.bias)

    def forward(self, base_prediction):
        # base_prediction: [B, S, C] -> prediction tokens: [B, C, D]
        batch_size, _, n_vars = base_prediction.shape
        if n_vars != self.n_vars:
            raise ValueError(
                f'Expected {self.n_vars} variables, got {n_vars}'
            )

        variable_ids = torch.arange(n_vars, device=base_prediction.device)
        tokens = self.prediction_embedding(base_prediction.permute(0, 2, 1))
        tokens = tokens + self.variate_embedding(variable_ids).unsqueeze(0)
        tokens = self.input_dropout(self.input_norm(tokens))

        # True entries are forbidden for nn.MultiheadAttention. Therefore the
        # diagonal prevents target c from ever reading the value of source c.
        self_mask = torch.eye(
            n_vars, dtype=torch.bool, device=base_prediction.device
        )
        cross_value, attention = self.attention(
            tokens,
            tokens,
            tokens,
            attn_mask=self_mask,
            need_weights=self.output_attention,
            average_attn_weights=False,
        )

        # There is intentionally no residual connection from `tokens` here:
        # the correction value path must contain only other-variable content.
        cross_value = self.attention_norm(self.dropout(cross_value))
        cross_value = self.ffn_norm(
            cross_value + self.dropout(self.ffn(cross_value))
        )
        residual = self.residual_projector(cross_value).permute(0, 2, 1)
        return residual, attention, self_mask


class Model(nn.Module):
    """
    Three-stage forecaster:

    1. channel-independent PatchTST temporal modeling;
    2. iTransformer historical cross-variate modeling and dynamic fusion;
    3. self-masked prediction-token cross-attention residual correction.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.n_vars = int(configs.enc_in)
        self.output_attention = bool(configs.output_attention)
        self.use_norm = bool(configs.use_norm)

        if configs.d_model % configs.n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads for all three stages'
            )

        self.patchtst = ChannelIndependentPatchTST(configs)
        self.itransformer = ITransformerBranch(configs)
        fusion_hidden = int(getattr(
            configs, 'three_fusion_hidden', max(16, configs.d_model // 2)
        ))
        if fusion_hidden < 1:
            raise ValueError('three_fusion_hidden must be at least 1')
        self.dynamic_fusion = DynamicForecastFusion(
            configs.d_model,
            self.pred_len,
            fusion_hidden,
            configs.dropout,
        )
        self.prediction_correction = MaskedPredictionCrossAttention(
            self.n_vars,
            self.pred_len,
            configs.d_model,
            configs.n_heads,
            configs.d_ff,
            configs.dropout,
            configs.activation,
            configs.output_attention,
        )

        gamma_init = float(getattr(configs, 'three_gamma_init', 0.1))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_enc.ndim != 3:
            raise ValueError('x_enc must have shape [batch, seq_len, variables]')
        _, seq_len, n_vars = x_enc.shape
        if seq_len != self.seq_len:
            raise ValueError(f'Expected seq_len={self.seq_len}, got {seq_len}')
        if n_vars != self.n_vars:
            raise ValueError(
                f'iTransformer_three expected {self.n_vars} variables, '
                f'got {n_vars}'
            )

        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False)
                + 1e-5
            )
            model_input = centered / stdev
        else:
            means = None
            stdev = None
            model_input = x_enc

        patch_prediction, patch_state, patch_attentions = self.patchtst(
            model_input
        )
        joint_prediction, joint_state, joint_attentions = self.itransformer(
            model_input
        )
        base_prediction, fusion_weights = self.dynamic_fusion(
            patch_prediction,
            joint_prediction,
            patch_state,
            joint_state,
        )

        if self.n_vars > 1:
            correction, correction_attention, correction_mask = (
                self.prediction_correction(base_prediction)
            )
        else:
            correction = torch.zeros_like(base_prediction)
            correction_attention = None
            correction_mask = torch.ones(
                1, 1, dtype=torch.bool, device=x_enc.device
            )

        scaled_correction = self.gamma * correction
        prediction = base_prediction + scaled_correction

        if self.use_norm:
            scale = stdev[:, 0, :].unsqueeze(1)
            location = means[:, 0, :].unsqueeze(1)
            prediction = prediction * scale + location

        diagnostics = {
            'patchtst_attention': patch_attentions,
            'itransformer_attention': joint_attentions,
            'masked_cross_attention': correction_attention,
            'masked_cross_attention_mask': correction_mask,
            'fusion_weights': fusion_weights,
            'gamma': self.gamma,
            'patch_len': self.patchtst.patch_len,
            'stride': self.patchtst.stride,
            'num_patches': self.patchtst.patch_num,
        }
        return prediction, diagnostics

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        prediction, diagnostics = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        prediction = prediction[:, -self.pred_len:, :]
        if self.output_attention:
            return prediction, diagnostics
        return prediction
