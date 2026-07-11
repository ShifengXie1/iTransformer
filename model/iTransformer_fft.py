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


class PatchBroadcastLayer(nn.Module):
    """Broadcast CTF Global Token information back to Patch Tokens."""

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

    def forward(self, patch_groups, global_tokens):
        outputs = []
        attns = []
        for channel_idx, patches in enumerate(patch_groups):
            channel_global = global_tokens[:, channel_idx, :, :]
            update, attn = self.attention(
                patches, channel_global, channel_global,
                need_weights=self.output_attention,
                average_attn_weights=False,
            )
            x = self.norm1(patches + self.dropout(update))
            y = self.linear2(self.dropout(
                self.activation(self.linear1(x))
            ))
            outputs.append(self.norm2(x + self.dropout(y)))
            attns.append(attn)
        return outputs, attns


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
    Period-aware Patch Tokens + single-variate Patch Transformer prediction
    + CTF-style Global Token routing + Patch Token fusion prediction.
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

        # Global Tokens are used only as CTF-style cross-variable routers.
        # Forecast heads always consume Patch Tokens.
        self.num_global_tokens = int(
            getattr(configs, 'num_global_tokens', 4)
        )
        if self.num_global_tokens < 1:
            raise ValueError('num_global_tokens must be at least 1')
        self.global_tokens = nn.Parameter(
            torch.randn(1, self.num_global_tokens, configs.d_model)
        )
        nn.init.normal_(self.global_tokens, std=0.02)

        # Variables with the same period share a projection head, while
        # different patch counts retain their full token sequence.
        self.period_feature_dims = {
            period: (self.seq_len // period) * configs.d_model
            for period in sorted(set(self.channel_periods))
        }
        self.self_projectors = nn.ModuleDict({
            str(period): nn.Linear(feature_dim, configs.pred_len)
            for period, feature_dim in self.period_feature_dims.items()
        })

        cross_layers = []
        for _ in range(max(1, configs.e_layers)):
            cross_layers.append(MaskedEncoderLayer(
                configs.d_model, configs.n_heads, configs.d_ff,
                configs.dropout, configs.activation, configs.output_attention,
                input_residual=True,
            ))
        self.cross_encoder = MaskedEncoder(cross_layers)
        self.patch_broadcast = PatchBroadcastLayer(
            configs.d_model, configs.n_heads, configs.d_ff,
            configs.dropout, configs.activation, configs.output_attention,
        )
        self.delta_projectors = nn.ModuleDict({
            str(period): nn.Linear(feature_dim, configs.pred_len)
            for period, feature_dim in self.period_feature_dims.items()
        })
        self.gate_projectors = nn.ModuleDict({
            str(period): nn.Linear(feature_dim * 2, configs.pred_len)
            for period, feature_dim in self.period_feature_dims.items()
        })
        self.head_dropout = nn.Dropout(configs.dropout)

        # Start close to the single-variate forecast without blocking gradients
        # from reaching the cross-variate Transformer on the first update.
        for projector in self.delta_projectors.values():
            nn.init.xavier_uniform_(projector.weight, gain=0.1)
            nn.init.zeros_(projector.bias)
        for projector in self.gate_projectors.values():
            nn.init.zeros_(projector.weight)
            nn.init.constant_(projector.bias, -2.0)

    def _append_global_tokens(self, patch_groups):
        token_groups = []
        for channel_idx, patches in enumerate(patch_groups):
            identity = self.period_embedding.channel_embedding.weight[
                channel_idx
            ].view(1, 1, self.d_model)
            global_token = (
                self.global_tokens.expand(patches.shape[0], -1, -1)
                + identity
            )
            token_groups.append(torch.cat([patches, global_token], dim=1))
        return torch.cat(token_groups, dim=1)

    def _build_ctf_mask(self, num_patches_list, device):
        """Reproduce CTF routing while keeping prediction Patch-based."""
        group_sizes = [
            num_patches + self.num_global_tokens
            for num_patches in num_patches_list
        ]
        total_tokens = sum(group_sizes)
        allowed = torch.zeros(
            total_tokens, total_tokens, dtype=torch.bool, device=device
        )

        query_offset = 0
        for query_channel, (query_patches, query_size) in enumerate(zip(
                num_patches_list, group_sizes)):
            # Exactly as in CTF: every token can read its own Patch Tokens.
            own_patch_slice = slice(query_offset,
                                    query_offset + query_patches)
            query_group_slice = slice(query_offset,
                                      query_offset + query_size)
            allowed[query_group_slice, own_patch_slice] = True

            key_offset = 0
            for key_channel, (key_patches, key_size) in enumerate(zip(
                    num_patches_list, group_sizes)):
                if key_channel != query_channel:
                    for slot in range(self.num_global_tokens):
                        query_idx = query_offset + query_patches + slot
                        key_idx = key_offset + key_patches + slot
                        allowed[query_idx, key_idx] = True
                key_offset += key_size
            query_offset += query_size
        return ~allowed

    def _extract_global_tokens(self, x, num_patches_list):
        global_groups = []
        offset = 0
        for num_patches in num_patches_list:
            start = offset + num_patches
            end = start + self.num_global_tokens
            global_groups.append(x[:, start:end, :])
            offset = end
        return torch.stack(global_groups, dim=1)

    def _project_by_period(self, token_groups, projectors):
        predictions = []
        features = []
        for period, tokens in zip(self.channel_periods, token_groups):
            feature = self.head_dropout(tokens.flatten(start_dim=1))
            expected_dim = self.period_feature_dims[period]
            if feature.shape[-1] != expected_dim:
                raise RuntimeError(
                    f'Period {period} produced feature dim {feature.shape[-1]}, '
                    f'expected {expected_dim}'
                )
            features.append(feature)
            predictions.append(projectors[str(period)](feature))
        return torch.stack(predictions, dim=-1), features

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
        self_pred, self_features = self._project_by_period(
            self_patch_tokens, self.self_projectors
        )

        if self.n_vars > 1:
            cross_input = self._append_global_tokens(self_patch_tokens)
            cross_out, cross_attns = self.cross_encoder(
                cross_input,
                self._build_ctf_mask(num_patches_list, x_enc.device),
            )
            global_tokens = self._extract_global_tokens(
                cross_out, num_patches_list
            )
            cross_patch_tokens, broadcast_attns = self.patch_broadcast(
                self_patch_tokens, global_tokens
            )
            delta, cross_features = self._project_by_period(
                cross_patch_tokens, self.delta_projectors
            )
            gates = []
            for period, self_feature, cross_feature in zip(
                    self.channel_periods, self_features, cross_features):
                gates.append(torch.sigmoid(
                    self.gate_projectors[str(period)](torch.cat(
                        [self_feature, cross_feature], dim=-1
                    ))
                ))
            gate = torch.stack(gates, dim=-1)
            dec_out = self_pred + gate * delta
        else:
            cross_attns = []
            broadcast_attns = []
            gate = torch.zeros_like(self_pred)
            dec_out = self_pred

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        attns = {
            'intra_variate': intra_attns,
            'cross_variate': cross_attns,
            'cross_to_patch': broadcast_attns,
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
