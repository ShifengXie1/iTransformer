import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedEncoderLayer(nn.Module):
    """Transformer encoder layer accepting a structural boolean mask."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation='gelu',
                 output_attention=False, input_residual=True):
        super().__init__()
        self.output_attention = output_attention
        self.input_residual = input_residual
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x, attn_mask):
        attn_out, attn = self.attention(
            x, x, x, attn_mask=attn_mask,
            need_weights=self.output_attention,
            average_attn_weights=False,
        )
        if self.input_residual:
            x = self.norm1(x + self.dropout(attn_out))
        else:
            # Prevent the cross-variate correction value path from directly
            # copying the query channel through a Transformer residual.
            x = self.norm1(self.dropout(attn_out))
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(y))
        return x, attn


class MaskedEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, attn_mask):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            attns.append(attn)
        return x, attns


class CTFPeriodEmbedding(nn.Module):
    """CTF-style fixed-period patch embedding without interpolation."""

    def __init__(self, seq_len, n_vars, d_model, channel_periods,
                 num_channel_tokens=4, dropout=0.1):
        super().__init__()
        if len(channel_periods) != n_vars:
            raise ValueError(
                f'Expected {n_vars} channel periods, got {len(channel_periods)}'
            )
        self.seq_len = int(seq_len)
        self.n_vars = int(n_vars)
        self.d_model = int(d_model)
        self.channel_periods = [int(p) for p in channel_periods]
        self.num_channel_tokens = int(num_channel_tokens)

        for period in self.channel_periods:
            if period < 1 or period > self.seq_len:
                raise ValueError(
                    f'Channel period {period} must be in [1, {self.seq_len}]'
                )

        unique_periods = sorted(set(self.channel_periods))
        self.period_to_scale = {
            period: idx for idx, period in enumerate(unique_periods)
        }
        self.patch_embeddings = nn.ModuleDict({
            str(period): nn.Linear(period, d_model)
            for period in unique_periods
        })
        max_num_patches = max(self.seq_len // p for p in self.channel_periods)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_num_patches, d_model)
        )
        self.channel_embedding = nn.Embedding(n_vars, d_model)
        self.scale_embedding = nn.Embedding(len(unique_periods), d_model)
        self.base_channel_tokens = nn.Parameter(
            torch.randn(1, self.num_channel_tokens, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.base_channel_tokens, std=0.02)

    def forward(self, x):
        # x: [B, L, C]
        batch_size, seq_len, n_vars = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f'Expected seq_len={self.seq_len}, got {seq_len}')
        if n_vars != self.n_vars:
            raise ValueError(f'Expected {self.n_vars} variables, got {n_vars}')

        token_groups = []
        num_patches_list = []
        for channel_idx, patch_len in enumerate(self.channel_periods):
            num_patches = max(1, seq_len // patch_len)
            usable_len = num_patches * patch_len
            # Keep observations nearest to the forecasting boundary.
            x_c = x[:, -usable_len:, channel_idx]
            patches = x_c.reshape(batch_size, num_patches, patch_len)
            local_tokens = self.patch_embeddings[str(patch_len)](patches)

            channel_idx_tensor = torch.tensor(channel_idx, device=x.device)
            scale_idx_tensor = torch.tensor(
                self.period_to_scale[patch_len], device=x.device
            )
            identity = (
                self.channel_embedding(channel_idx_tensor)
                + self.scale_embedding(scale_idx_tensor)
            ).view(1, 1, self.d_model)
            local_tokens = (
                local_tokens
                + self.position_embedding[:, :num_patches, :]
                + identity
            )
            channel_tokens = (
                self.base_channel_tokens.expand(batch_size, -1, -1)
                + identity
            )
            # CTF layout: [local tokens, channel tokens] for each channel.
            token_groups.append(torch.cat([local_tokens, channel_tokens], dim=1))
            num_patches_list.append(num_patches)

        return self.dropout(torch.cat(token_groups, dim=1)), num_patches_list

    def build_intra_mask(self, num_patches_list, device):
        """Make Channel Tokens strictly summarize their own local tokens."""
        group_sizes = [n + self.num_channel_tokens for n in num_patches_list]
        total_tokens = sum(group_sizes)
        allowed = torch.zeros(
            total_tokens, total_tokens, dtype=torch.bool, device=device
        )
        offset = 0
        for num_patches, group_size in zip(num_patches_list, group_sizes):
            local_slice = slice(offset, offset + num_patches)
            channel_slice = slice(offset + num_patches, offset + group_size)
            allowed[local_slice, local_slice] = True
            allowed[channel_slice, local_slice] = True
            offset += group_size
        # nn.MultiheadAttention: True means forbidden.
        return ~allowed

    def extract_channel_tokens(self, x, num_patches_list):
        groups = []
        offset = 0
        for num_patches in num_patches_list:
            start = offset + num_patches
            end = start + self.num_channel_tokens
            groups.append(x[:, start:end, :])
            offset = end
        return torch.stack(groups, dim=1)  # [B, C, M, D]


class CPTA_iTransformer(nn.Module):
    """
    Fixed period-aware patching + CTF-style Channel Tokens + single-variate
    base prediction + gated cross-variate residual correction.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.n_vars = configs.enc_in
        self.num_channel_tokens = int(
            getattr(configs, 'num_channel_tokens',
                    getattr(configs, 'period_query_num', 4))
        )
        channel_periods = getattr(configs, 'channel_periods', None)
        if channel_periods is None:
            raise ValueError(
                'iTransformer_fft requires channel_periods. Pass '
                '--channel_periods or let run.py estimate them from train data.'
            )
        self.channel_periods = [int(p) for p in channel_periods]
        self.last_batch_periods = list(self.channel_periods)

        self.period_embedding = CTFPeriodEmbedding(
            configs.seq_len, self.n_vars, configs.d_model,
            self.channel_periods, self.num_channel_tokens, configs.dropout
        )

        self.intra_encoder = MaskedEncoder([
            MaskedEncoderLayer(
                configs.d_model, configs.n_heads, configs.d_ff,
                configs.dropout, configs.activation, configs.output_attention,
                input_residual=True,
            )
            for _ in range(max(1, int(getattr(configs, 'intra_layers', 1))))
        ])

        feature_dim = self.num_channel_tokens * configs.d_model
        self.self_projector = nn.Linear(feature_dim, configs.pred_len)

        cross_layers = []
        for layer_idx in range(max(1, configs.e_layers)):
            cross_layers.append(MaskedEncoderLayer(
                configs.d_model, configs.n_heads, configs.d_ff,
                configs.dropout, configs.activation, configs.output_attention,
                input_residual=layer_idx > 0,
            ))
        self.cross_encoder = MaskedEncoder(cross_layers)
        self.delta_projector = nn.Linear(feature_dim, configs.pred_len)
        self.gate_projector = nn.Linear(feature_dim * 2, configs.pred_len)
        self.head_dropout = nn.Dropout(configs.dropout)

        # Begin as a single-variate forecaster; learn cross-variate correction.
        nn.init.zeros_(self.delta_projector.weight)
        nn.init.zeros_(self.delta_projector.bias)
        nn.init.zeros_(self.gate_projector.weight)
        nn.init.constant_(self.gate_projector.bias, -2.0)

    def _build_cross_mask(self, device):
        channel_ids = torch.arange(self.n_vars, device=device)
        channel_ids = channel_ids.repeat_interleave(self.num_channel_tokens)
        slot_ids = torch.arange(self.num_channel_tokens, device=device)
        slot_ids = slot_ids.repeat(self.n_vars)
        # Match CTF's global-token routing: a Channel Token communicates with
        # the same token slot of other variables, never with its own channel.
        allowed = (
            channel_ids[:, None].ne(channel_ids[None, :])
            & slot_ids[:, None].eq(slot_ids[None, :])
        )
        return ~allowed

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        all_tokens, num_patches_list = self.period_embedding(x_enc)
        intra_mask = self.period_embedding.build_intra_mask(
            num_patches_list, x_enc.device
        )
        intra_out, intra_attns = self.intra_encoder(all_tokens, intra_mask)
        self_tokens = self.period_embedding.extract_channel_tokens(
            intra_out, num_patches_list
        )
        self_features = self.head_dropout(self_tokens.flatten(start_dim=2))
        self_pred = self.self_projector(self_features).permute(0, 2, 1)

        if self.n_vars > 1:
            batch_size = x_enc.shape[0]
            cross_input = self_tokens.reshape(
                batch_size,
                self.n_vars * self.num_channel_tokens,
                self.d_model,
            )
            cross_out, cross_attns = self.cross_encoder(
                cross_input, self._build_cross_mask(x_enc.device)
            )
            cross_tokens = cross_out.reshape(
                batch_size, self.n_vars,
                self.num_channel_tokens, self.d_model
            )
            cross_features = self.head_dropout(
                cross_tokens.flatten(start_dim=2)
            )
            delta = self.delta_projector(cross_features).permute(0, 2, 1)
            gate = torch.sigmoid(self.gate_projector(torch.cat(
                [self_features, cross_features], dim=-1
            ))).permute(0, 2, 1)
            dec_out = self_pred + gate * delta
        else:
            cross_attns = []
            gate = torch.zeros_like(self_pred)
            dec_out = self_pred

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        attns = {
            'intra_variate': intra_attns,
            'cross_variate': cross_attns,
            'periods': list(self.channel_periods),
            'num_patches': list(num_patches_list),
            'fusion_gate': gate,
        }
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]


class Model(CPTA_iTransformer):
    pass
