import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class PeriodKeyPaddingMask:
    def __init__(self, valid_mask, query_len):
        # valid_mask: [B, S], True means the patch token has real values.
        self.mask = (~valid_mask).unsqueeze(1).unsqueeze(2).expand(-1, 1, query_len, -1)

class ChannelWisePeriodTokenizer(nn.Module):
    def __init__(self, seq_len, d_model, base_patch_len=16, dropout=0.1):
        super(ChannelWisePeriodTokenizer, self).__init__()
        self.seq_len = seq_len
        self.base_patch_len = max(1, int(base_patch_len))
        self.patch_projection = nn.Linear(self.base_patch_len * 2, d_model)
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

    def _resample_mask(self, patch_masks):
        # patch_masks: [B, N_c, P_c], 1.0 means real value and 0.0 means padding.
        patch_len = patch_masks.shape[-1]
        if patch_len == self.base_patch_len:
            return patch_masks

        batch_size, num_patches, _ = patch_masks.shape
        patch_masks = patch_masks.reshape(batch_size * num_patches, 1, patch_len)
        patch_masks = F.interpolate(patch_masks, size=self.base_patch_len, mode='nearest')
        return patch_masks.reshape(batch_size, num_patches, self.base_patch_len)

    def forward(self, x):
        # x: [B, L, C]
        batch_size, seq_len, num_channels = x.shape
        periods = self._detect_periods_by_fft(x)
        channel_tokens = []
        channel_token_masks = []

        for channel_idx in range(num_channels):
            period = periods[channel_idx]
            num_patches = max(1, (seq_len + period - 1) // period)
            padded_len = num_patches * period
            pad_len = padded_len - seq_len

            # Left padding keeps the latest observations in the final patch.
            # x_c: [B, L] -> padded_x_c: [B, N_c * P_c]
            x_c = x[:, :, channel_idx]
            value_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype)
            if pad_len > 0:
                x_c = F.pad(x_c, (pad_len, 0), mode='constant', value=0.0)
                value_mask = F.pad(value_mask, (pad_len, 0), mode='constant', value=0.0)

            # [B, N_c * P_c] -> [B, N_c, P_c]
            patches = x_c.reshape(batch_size, num_patches, period)
            patch_masks = value_mask.reshape(batch_size, num_patches, period)
            patch_token_mask = patch_masks.sum(dim=-1) > 0

            patches = self._resample_patch(patches)  # [B, N_c, base_patch_len]
            patch_masks = self._resample_mask(patch_masks)  # [B, N_c, base_patch_len]

            # Concatenate the padding mask so projection can distinguish real values from padding.
            # [B, N_c, 2 * base_patch_len] -> [B, N_c, d_model]
            patch_inputs = torch.cat([patches, patch_masks], dim=-1)
            tokens = self.patch_projection(patch_inputs)
            tokens = tokens + self.patch_position_embedding[:, :num_patches, :]
            channel_tokens.append(self.dropout(tokens))
            channel_token_masks.append(patch_token_mask)

        return channel_tokens, channel_token_masks, periods


class FixedQueryPeriodAggregation(nn.Module):
    def __init__(self, d_model, n_heads, query_num, factor=1, dropout=0.1, output_attention=False):
        super(FixedQueryPeriodAggregation, self).__init__()
        self.query_num = query_num
        self.queries = nn.Parameter(torch.empty(query_num, d_model))
        self.attention = AttentionLayer(
            FullAttention(True, factor, attention_dropout=dropout, output_attention=output_attention),
            d_model,
            n_heads
        )
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.queries)

    def forward(self, period_tokens, patch_token_mask):
        # period_tokens: [B, N_c, d_model], patch_token_mask: [B, N_c]
        batch_size = period_tokens.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, d_model]
        attn_mask = PeriodKeyPaddingMask(patch_token_mask.bool(), self.query_num)
        aligned_tokens, attn = self.attention(queries, period_tokens, period_tokens, attn_mask=attn_mask)
        return self.dropout(aligned_tokens), attn  # [B, K, d_model]

class PeriodAwareEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, query_num, factor=1,
                 base_patch_len=16, dropout=0.1, output_attention=False):
        super(PeriodAwareEmbedding, self).__init__()
        self.tokenizer = ChannelWisePeriodTokenizer(
            seq_len=seq_len,
            d_model=d_model,
            base_patch_len=base_patch_len,
            dropout=dropout
        )
        self.aggregation = FixedQueryPeriodAggregation(
            d_model=d_model,
            n_heads=n_heads,
            query_num=query_num,
            factor=factor,
            dropout=dropout,
            output_attention=output_attention
        )

    def forward(self, x, x_mark=None):
        # x: [B, L, C]. x_mark is kept only for interface compatibility.
        channel_tokens, channel_token_masks, periods = self.tokenizer(x)

        aligned_channels = []
        align_attns = []
        for tokens, token_mask in zip(channel_tokens, channel_token_masks):
            # tokens: [B, N_c, d_model] -> aligned: [B, K, d_model]
            aligned, attn = self.aggregation(tokens, token_mask)
            aligned_channels.append(aligned)
            align_attns.append(attn)

        # [B, K, d_model] * C -> [B, C, K, d_model]
        aligned_channels = torch.stack(aligned_channels, dim=1)
        return aligned_channels, align_attns, periods

class CPTA_iTransformer(nn.Module):
    """
    Channel-wise Period Tokenization + fixed-query alignment + iTransformer cross-variate attention.
    """

    def __init__(self, configs):
        super(CPTA_iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.period_query_num = max(1, int(getattr(configs, 'period_query_num', 4)))
        self.base_patch_len = int(getattr(configs, 'base_patch_len', 16))

        # B L C -> B C K d_model
        self.period_embedding = PeriodAwareEmbedding(
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            query_num=self.period_query_num,
            factor=configs.factor,
            base_patch_len=self.base_patch_len,
            dropout=configs.dropout,
            output_attention=configs.output_attention
        )

        # Phase-wise iTransformer encoder: for each k, [B, C, d_model] -> [B, C, d_model]
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

        # Per-channel head: [B, C, K * d_model] -> [B, C, pred_len]
        self.projector = nn.Linear(self.period_query_num * configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        batch_size, _, num_channels = x_enc.shape  # [B, L, C]

        # Channel-wise period tokenization and fixed-query alignment.
        # x_enc: [B, L, C] -> period_out: [B, C, K, d_model]
        period_out, period_attns, periods = self.period_embedding(x_enc, x_mark_enc)

        # Phase-wise cross-variate attention.
        phase_outs = []
        phase_attns = []
        for phase_idx in range(self.period_query_num):
            # [B, C, K, d_model] -> [B, C, d_model]
            phase_tokens = period_out[:, :, phase_idx, :]
            phase_tokens, attn = self.encoder(phase_tokens, attn_mask=None)
            phase_outs.append(phase_tokens)
            phase_attns.append(attn)

        # [B, C, d_model] * K -> [B, C, K, d_model]
        enc_out = torch.stack(phase_outs, dim=2)

        # [B, C, K, d_model] -> [B, C, K * d_model] -> [B, pred_len, C]
        enc_out = enc_out.reshape(batch_size, num_channels, self.period_query_num * self.d_model)
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        attns = {
            'period_alignment': period_attns,
            'cross_variate': phase_attns,
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
