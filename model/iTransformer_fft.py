import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class ChannelWisePeriodTokenizer(nn.Module):
    def __init__(self, seq_len, d_model, base_patch_len=16, dropout=0.1):
        super(ChannelWisePeriodTokenizer, self).__init__()
        self.seq_len = seq_len
        self.base_patch_len = max(1, int(base_patch_len))
        self.patch_projection = nn.Linear(self.base_patch_len, d_model)
        self.patch_position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.dropout = nn.Dropout(p=dropout)
        nn.init.normal_(self.patch_position_embedding, std=0.02)

    def _detect_periods_by_fft(self, x):
        # x: [B, L, C]
        _, seq_len, num_channels = x.shape
        if seq_len <= 1:
            return [1 for _ in range(num_channels)]

        with torch.no_grad():
            x_centered = x - x.mean(dim=1, keepdim=True)
            spectrum = torch.fft.rfft(x_centered, dim=1)
            amplitude = spectrum.abs().mean(dim=0)  # [F, C]
            amplitude[0, :] = -float('inf')

            top_freq = amplitude.argmax(dim=0).clamp(min=1)  # [C]
            periods = torch.round(seq_len / top_freq.float()).long()
            periods = periods.clamp(min=1, max=seq_len).detach().cpu().tolist()

        return [int(p) for p in periods]

    def _resample_patch(self, patches):
        # patches: [B, N_c, P_c]
        patch_len = patches.shape[-1]
        if patch_len == self.base_patch_len:
            return patches

        batch_size, num_patches, _ = patches.shape
        patches = patches.reshape(batch_size * num_patches, 1, patch_len)
        patches = F.interpolate(patches, size=self.base_patch_len, mode='linear', align_corners=False)
        return patches.reshape(batch_size, num_patches, self.base_patch_len)

    def forward(self, x):
        # x: [B, L, C]
        batch_size, seq_len, num_channels = x.shape
        periods = self._detect_periods_by_fft(x)
        channel_tokens = []

        for channel_idx in range(num_channels):
            period = periods[channel_idx]
            num_patches = max(1, seq_len // period)
            usable_len = num_patches * period

            # Drop the final incomplete patch directly.
            # x_c: [B, usable_len] -> patches: [B, N_c, P_c]
            x_c = x[:, :usable_len, channel_idx]
            patches = x_c.reshape(batch_size, num_patches, period)
            patches = self._resample_patch(patches)  # [B, N_c, base_patch_len]

            # [B, N_c, base_patch_len] -> [B, N_c, d_model]
            tokens = self.patch_projection(patches)
            tokens = tokens + self.patch_position_embedding[:, :num_patches, :]
            channel_tokens.append(self.dropout(tokens))

        return channel_tokens, periods

class PeriodAwareEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, d_ff=None, factor=1,
                 base_patch_len=16, dropout=0.1, activation='gelu', output_attention=False):
        super(PeriodAwareEmbedding, self).__init__()
        self.tokenizer = ChannelWisePeriodTokenizer(
            seq_len=seq_len,
            d_model=d_model,
            base_patch_len=base_patch_len,
            dropout=dropout
        )
        self.temporal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, x, x_mark=None):
        # x: [B, L, C]. x_mark is kept only for interface compatibility.
        channel_tokens, periods = self.tokenizer(x)

        variable_tokens = []
        temporal_attns = []
        for tokens in channel_tokens:
            # tokens: [B, M_c, d_model] -> encoded: [B, M_c, d_model]
            encoded, attn = self.temporal_encoder(tokens, attn_mask=None)
            variable_tokens.append(encoded[:, -1, :])
            temporal_attns.append(attn)

        # [B, d_model] * C -> [B, C, d_model]
        variable_tokens = torch.stack(variable_tokens, dim=1)
        return variable_tokens, temporal_attns, periods

class CPTA_iTransformer(nn.Module):
    """
    Channel-wise period tokenization + intra-variate attention + iTransformer cross-variate attention.
    """

    def __init__(self, configs):
        super(CPTA_iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.base_patch_len = int(getattr(configs, 'base_patch_len', 16))

        # B L C -> B C d_model
        self.period_embedding = PeriodAwareEmbedding(
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            factor=configs.factor,
            base_patch_len=self.base_patch_len,
            dropout=configs.dropout,
            activation=configs.activation,
            output_attention=configs.output_attention
        )

        # iTransformer encoder: [B, C, d_model] -> [B, C, d_model]
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Per-channel head: [B, C, d_model] -> [B, C, pred_len]
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev


        # Channel-wise period tokenization and intra-variate attention.
        # x_enc: [B, L, C] -> variable_tokens: [B, C, d_model]
        variable_tokens, temporal_attns, periods = self.period_embedding(x_enc, x_mark_enc)

        # Cross-variate attention, following the original iTransformer token layout.
        enc_out, cross_attns = self.encoder(variable_tokens, attn_mask=None)

        # [B, C, d_model] -> [B, C, pred_len] -> [B, pred_len, C]
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        attns = {
            'intra_variate': temporal_attns,
            'cross_variate': cross_attns,
            'periods': periods,
        }
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, C]

class Model(CPTA_iTransformer):
    pass




