from collections import Counter

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


class FutureTokenGenerator(nn.Module):
    """Generate future Patch Tokens by attending to encoded history Tokens."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 activation='gelu', output_attention=False):
        super().__init__()
        self.output_attention = output_attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, future_queries, history_tokens):
        update, attn = self.attention(
            future_queries, history_tokens, history_tokens,
            need_weights=self.output_attention,
            average_attn_weights=False,
        )
        x = self.norm1(future_queries + self.dropout(update))
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout(y)), attn


class PeriodPatchEmbedding(nn.Module):
    """Variable-wise fixed-period patch embedding without interpolation."""

    def __init__(self, seq_len, n_vars, d_model, channel_periods, dropout=0.1):
        super().__init__()
        if len(channel_periods) != n_vars:
            raise ValueError(
                f'Expected {n_vars} channel periods, got {len(channel_periods)}'
            )
        self.seq_len = int(seq_len)
        self.n_vars = int(n_vars)
        self.d_model = int(d_model)
        self.channel_periods = [int(p) for p in channel_periods]
        for period in self.channel_periods:
            if period < 1 or period > self.seq_len:
                raise ValueError(
                    f'Channel period {period} must be in [1, {self.seq_len}]'
                )

        unique_periods = sorted(set(self.channel_periods))
        self.patch_embeddings = nn.ModuleDict({
            str(period): nn.Linear(period, d_model)
            for period in unique_periods
        })
        max_num_patches = max(self.seq_len // p for p in self.channel_periods)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_num_patches, d_model)
        )
        self.channel_embedding = nn.Embedding(n_vars, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.position_embedding, std=0.02)

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
            identity = self.channel_embedding(channel_idx_tensor).view(
                1, 1, self.d_model
            )
            local_tokens = (
                local_tokens
                + self.position_embedding[:, :num_patches, :]
                + identity
            )
            token_groups.append(local_tokens)
            num_patches_list.append(num_patches)

        return self.dropout(torch.cat(token_groups, dim=1)), num_patches_list

    def build_intra_mask(self, num_patches_list, device):
        """Restrict each Patch Token to its own variable."""
        total_tokens = sum(num_patches_list)
        allowed = torch.zeros(
            total_tokens, total_tokens, dtype=torch.bool, device=device
        )
        offset = 0
        for num_patches in num_patches_list:
            patch_slice = slice(offset, offset + num_patches)
            allowed[patch_slice, patch_slice] = True
            offset += num_patches
        # nn.MultiheadAttention: True means forbidden.
        return ~allowed

    @staticmethod
    def split_patch_tokens(x, num_patches_list):
        groups = []
        offset = 0
        for num_patches in num_patches_list:
            end = offset + num_patches
            groups.append(x[:, offset:end, :])
            offset = end
        return groups


class CPTA_iTransformer(nn.Module):
    """
    Variable-period self forecasting with future Patch Tokens, plus a
    mode-period cross-variate Token correction branch.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.n_vars = configs.enc_in
        channel_periods = getattr(configs, 'channel_periods', None)
        if channel_periods is None:
            raise ValueError(
                'iTransformer_fft requires channel_periods. Pass '
                '--channel_periods or let run.py estimate them from train data.'
            )
        self.channel_periods = [int(p) for p in channel_periods]
        self.last_batch_periods = list(self.channel_periods)

        self.period_embedding = PeriodPatchEmbedding(
            configs.seq_len, self.n_vars, configs.d_model,
            self.channel_periods, configs.dropout
        )

        self.intra_encoder = MaskedEncoder([
            MaskedEncoderLayer(
                configs.d_model, configs.n_heads, configs.d_ff,
                configs.dropout, configs.activation, configs.output_attention,
                input_residual=True,
            )
            for _ in range(max(1, int(getattr(configs, 'intra_layers', 1))))
        ])

        unique_periods = sorted(set(self.channel_periods))
        self.self_future_queries = nn.ParameterDict({
            str(period): nn.Parameter(torch.randn(
                1, (self.pred_len + period - 1) // period, configs.d_model
            ))
            for period in unique_periods
        })
        self.self_future_generator = FutureTokenGenerator(
            configs.d_model, configs.n_heads, configs.d_ff,
            configs.dropout, configs.activation, configs.output_attention,
        )
        self.self_patch_decoders = nn.ModuleDict({
            str(period): nn.Linear(configs.d_model, period)
            for period in unique_periods
        })

        period_counts = Counter(self.channel_periods)
        max_count = max(period_counts.values())
        inferred_cross_period = min(
            period for period, count in period_counts.items()
            if count == max_count
        )
        self.cross_period = int(getattr(
            configs, 'cross_period', inferred_cross_period
        ))
        if self.cross_period != inferred_cross_period:
            raise ValueError(
                f'cross_period must be the mode of channel_periods: expected '
                f'{inferred_cross_period}, got {self.cross_period}'
            )
        self.cross_period_embedding = PeriodPatchEmbedding(
            configs.seq_len, self.n_vars, configs.d_model,
            [self.cross_period] * self.n_vars, configs.dropout
        )

        cross_layers = []
        for layer_idx in range(max(1, configs.e_layers)):
            cross_layers.append(MaskedEncoderLayer(
                configs.d_model, configs.n_heads, configs.d_ff,
                configs.dropout, configs.activation, configs.output_attention,
                input_residual=layer_idx > 0,
            ))
        self.cross_encoder = MaskedEncoder(cross_layers)
        self.cross_future_query = nn.Parameter(torch.randn(
            1,
            (self.pred_len + self.cross_period - 1) // self.cross_period,
            configs.d_model,
        ))
        self.cross_future_generator = FutureTokenGenerator(
            configs.d_model, configs.n_heads, configs.d_ff,
            configs.dropout, configs.activation, configs.output_attention,
        )
        self.cross_patch_decoder = nn.Linear(
            configs.d_model, self.cross_period
        )
        self.gate_projector = nn.Linear(2, 1)
        self.head_dropout = nn.Dropout(configs.dropout)

        for queries in self.self_future_queries.values():
            nn.init.normal_(queries, std=0.02)
        nn.init.normal_(self.cross_future_query, std=0.02)
        nn.init.xavier_uniform_(self.cross_patch_decoder.weight, gain=0.1)
        nn.init.zeros_(self.cross_patch_decoder.bias)
        nn.init.zeros_(self.gate_projector.weight)
        nn.init.constant_(self.gate_projector.bias, -2.0)

    def _build_cross_variate_mask(self, num_patches, device):
        """Allow every Patch Token to read only other variables."""
        channel_ids = torch.arange(self.n_vars, device=device)
        channel_ids = channel_ids.repeat_interleave(num_patches)
        allowed = channel_ids[:, None].ne(channel_ids[None, :])
        return ~allowed

    def _generate_self_future_tokens(self, history_groups):
        future_groups = []
        attns = []
        for channel_idx, (period, history) in enumerate(zip(
                self.channel_periods, history_groups)):
            identity = self.period_embedding.channel_embedding.weight[
                channel_idx
            ].view(1, 1, self.d_model)
            queries = self.self_future_queries[str(period)].expand(
                history.shape[0], -1, -1
            ) + identity
            future, attn = self.self_future_generator(queries, history)
            future_groups.append(future)
            attns.append(attn)
        return future_groups, attns

    def _generate_cross_future_tokens(self, history_groups):
        future_groups = []
        attns = []
        for channel_idx, history in enumerate(history_groups):
            identity = self.cross_period_embedding.channel_embedding.weight[
                channel_idx
            ].view(1, 1, self.d_model)
            queries = self.cross_future_query.expand(
                history.shape[0], -1, -1
            ) + identity
            future, attn = self.cross_future_generator(queries, history)
            future_groups.append(future)
            attns.append(attn)
        return future_groups, attns

    def _decode_self_future(self, future_groups):
        predictions = []
        for period, tokens in zip(self.channel_periods, future_groups):
            patches = self.self_patch_decoders[str(period)](
                self.head_dropout(tokens)
            )
            predictions.append(
                patches.flatten(start_dim=1)[:, :self.pred_len]
            )
        return torch.stack(predictions, dim=-1)

    def _decode_cross_future(self, future_groups):
        predictions = []
        for tokens in future_groups:
            patches = self.cross_patch_decoder(self.head_dropout(tokens))
            predictions.append(
                patches.flatten(start_dim=1)[:, :self.pred_len]
            )
        return torch.stack(predictions, dim=-1)

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
        self_patch_tokens = self.period_embedding.split_patch_tokens(
            intra_out, num_patches_list
        )
        self_future_tokens, self_future_attns = (
            self._generate_self_future_tokens(self_patch_tokens)
        )
        self_pred = self._decode_self_future(self_future_tokens)

        if self.n_vars > 1:
            cross_input, cross_num_patches_list = (
                self.cross_period_embedding(x_enc)
            )
            cross_num_patches = cross_num_patches_list[0]
            if any(
                    count != cross_num_patches
                    for count in cross_num_patches_list):
                raise RuntimeError(
                    'Mode-period cross branch must produce equal Token counts'
                )
            cross_out, cross_attns = self.cross_encoder(
                cross_input,
                self._build_cross_variate_mask(
                    cross_num_patches, x_enc.device
                ),
            )
            cross_patch_tokens = (
                self.cross_period_embedding.split_patch_tokens(
                    cross_out, cross_num_patches_list
                )
            )
            cross_future_tokens, cross_future_attns = (
                self._generate_cross_future_tokens(cross_patch_tokens)
            )
            # Decoded cross patches can exceed pred_len when it is not a
            # multiple of cross_period. _decode_cross_future truncates first,
            # so only an exact-length correction is added to self_pred.
            delta = self._decode_cross_future(cross_future_tokens)
            gate_input = torch.stack([self_pred, delta], dim=-1)
            gate = torch.sigmoid(self.gate_projector(gate_input).squeeze(-1))
            dec_out = self_pred + gate * delta
        else:
            cross_attns = []
            cross_future_attns = []
            cross_num_patches_list = []
            gate = torch.zeros_like(self_pred)
            dec_out = self_pred

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        attns = {
            'intra_variate': intra_attns,
            'self_to_future': self_future_attns,
            'cross_variate': cross_attns,
            'cross_to_future': cross_future_attns,
            'periods': list(self.channel_periods),
            'num_patches': list(num_patches_list),
            'cross_period': self.cross_period,
            'num_cross_patches': list(cross_num_patches_list),
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
