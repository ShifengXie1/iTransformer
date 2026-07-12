import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted


class VariateSelfLayer(nn.Module):
    """Process every variate Token independently without cross-variate mixing."""

    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.value_projection = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.value_projection(x)))
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout(y))


class PeriodConditionedRouter(nn.Module):
    """Select directed Top-K source variates for every target and batch."""

    def __init__(self, d_model, periods, confidence, seq_len,
                 top_k=3, temperature=1.0, dropout=0.1):
        super().__init__()
        self.n_vars = len(periods)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        if self.top_k < 1:
            raise ValueError('cross_top_k must be at least 1')
        if self.temperature <= 0:
            raise ValueError('router_temperature must be positive')

        period_tensor = torch.tensor(periods, dtype=torch.float32)
        if confidence is None:
            confidence_tensor = torch.ones_like(period_tensor)
        else:
            if len(confidence) != self.n_vars:
                raise ValueError(
                    f'Expected {self.n_vars} confidence values, '
                    f'got {len(confidence)}'
                )
            confidence_tensor = torch.tensor(confidence, dtype=torch.float32)

        period_ratio = (period_tensor / float(seq_len)).clamp_min(1e-6)
        log_period = torch.log(period_ratio)
        log_confidence = torch.log1p(confidence_tensor.clamp_min(0))
        confidence_scale = log_confidence.max().clamp_min(1e-6)
        confidence_feature = log_confidence / confidence_scale
        descriptors = torch.stack(
            [period_ratio, log_period, confidence_feature], dim=-1
        )
        self.register_buffer('period_descriptors', descriptors)

        self.period_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.router_norm = nn.LayerNorm(d_model)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, variate_tokens):
        # variate_tokens: [B, C, D]
        batch_size, n_vars, d_model = variate_tokens.shape
        if n_vars != self.n_vars:
            raise ValueError(
                f'Router expected {self.n_vars} variates, got {n_vars}'
            )
        if n_vars < 2:
            empty_indices = torch.empty(
                batch_size, n_vars, 0, dtype=torch.long,
                device=variate_tokens.device
            )
            empty_values = variate_tokens.new_empty(batch_size, n_vars, 0)
            return empty_indices, empty_values, empty_values

        period_tokens = self.period_encoder(
            self.period_descriptors.to(variate_tokens.dtype)
        ).unsqueeze(0)
        router_tokens = self.router_norm(variate_tokens + period_tokens)
        router_tokens = self.dropout(router_tokens)
        queries = self.query_projection(router_tokens)
        keys = self.key_projection(router_tokens)
        scores = torch.matmul(queries, keys.transpose(-1, -2))
        scores = scores / math.sqrt(d_model)

        diagonal = torch.eye(
            n_vars, dtype=torch.bool, device=variate_tokens.device
        ).unsqueeze(0)
        scores = scores.masked_fill(diagonal, -torch.inf)
        selected_k = min(self.top_k, n_vars - 1)
        top_scores, top_indices = torch.topk(
            scores, k=selected_k, dim=-1
        )
        top_weights = torch.softmax(
            top_scores / self.temperature, dim=-1
        )
        return top_indices, top_weights, scores


class SparseCrossVariateLayer(nn.Module):
    """Cross-attend each target Token only to routed source variate Tokens."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 activation='gelu', output_attention=False):
        super().__init__()
        self.n_heads = n_heads
        self.output_attention = output_attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, target_tokens, source_tokens,
                source_indices, source_weights):
        # target_tokens: [B, C, D]
        # source_tokens: [B, C, D]
        batch_size, n_vars, d_model = source_tokens.shape
        selected_k = source_indices.shape[-1]

        source_bank = source_tokens.unsqueeze(1).expand(
            -1, n_vars, -1, -1
        )
        gather_indices = source_indices.unsqueeze(-1).expand(
            -1, -1, -1, d_model
        )
        selected_sources = torch.gather(
            source_bank, dim=2, index=gather_indices
        )
        # Scaling the selected source groups makes the differentiable routing
        # weights affect the prediction, while Top-K controls actual sparsity.
        selected_sources = selected_sources * (
            source_weights.unsqueeze(-1) * selected_k
        )
        memory = selected_sources.reshape(
            batch_size * n_vars, selected_k, d_model
        )
        queries = target_tokens.reshape(
            batch_size * n_vars, 1, d_model
        )
        cross_value, attention = self.attention(
            queries, memory, memory,
            need_weights=self.output_attention,
            average_attn_weights=False,
        )
        # Do not add the target query as a value residual: this branch should
        # contain only information written from selected source variates.
        cross_value = self.norm1(self.dropout(cross_value))
        y = self.linear2(self.dropout(
            self.activation(self.linear1(cross_value))
        ))
        cross_value = self.norm2(cross_value + self.dropout(y))
        cross_value = cross_value.reshape(batch_size, n_vars, d_model)

        if attention is not None:
            attention = attention.reshape(
                batch_size, n_vars, self.n_heads, selected_k
            )
        return cross_value, attention


class Model(nn.Module):
    """
    iTransformer-style self-variate forecasting plus period-conditioned,
    dynamic sparse cross-variate residual routing.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.n_vars = configs.enc_in

        periods = getattr(configs, 'channel_periods', None)
        if periods is None:
            raise ValueError(
                'iTransformer_cross requires channel_periods from FFT'
            )
        if len(periods) != self.n_vars:
            raise ValueError(
                f'Expected {self.n_vars} channel periods, got {len(periods)}'
            )
        self.channel_periods = [int(period) for period in periods]
        confidence = getattr(configs, 'channel_period_confidence', None)

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )
        self.self_encoder = nn.ModuleList([
            VariateSelfLayer(
                configs.d_model, configs.d_ff, configs.dropout,
                configs.activation
            )
            for _ in range(max(1, configs.e_layers))
        ])
        self.self_projector = nn.Linear(
            configs.d_model, configs.pred_len, bias=True
        )

        self.router = PeriodConditionedRouter(
            configs.d_model,
            self.channel_periods,
            confidence,
            configs.seq_len,
            top_k=getattr(configs, 'cross_top_k', 3),
            temperature=getattr(configs, 'router_temperature', 1.0),
            dropout=configs.dropout,
        )
        self.cross_layer = SparseCrossVariateLayer(
            configs.d_model, configs.n_heads, configs.d_ff,
            configs.dropout, configs.activation, configs.output_attention
        )
        self.cross_projector = nn.Linear(
            configs.d_model, configs.pred_len, bias=True
        )
        self.fusion_gate = nn.Linear(
            configs.d_model * 2, configs.pred_len, bias=True
        )

        # Begin near the pure self-variate forecaster while retaining gradient
        # flow through the sparse cross-variate correction branch.
        nn.init.xavier_uniform_(self.cross_projector.weight, gain=0.1)
        nn.init.zeros_(self.cross_projector.bias)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -2.0)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        _, _, n_vars = x_enc.shape
        if n_vars != self.n_vars:
            raise ValueError(
                f'iTransformer_cross expected {self.n_vars} variates, '
                f'got {n_vars}'
            )

        # Whole-history variate Tokens: [B, L, C] -> [B, C, D]. Time marks
        # are intentionally excluded because this branch must contain exactly
        # one Token per predicted variate.
        self_tokens = self.enc_embedding(x_enc, None)
        for layer in self.self_encoder:
            self_tokens = layer(self_tokens)
        self_pred = self.self_projector(self_tokens).permute(0, 2, 1)

        if self.n_vars > 1:
            source_indices, source_weights, route_scores = self.router(
                self_tokens
            )
            cross_tokens, cross_attention = self.cross_layer(
                self_tokens,
                self_tokens,
                source_indices,
                source_weights,
            )
            delta = self.cross_projector(cross_tokens).permute(0, 2, 1)
            gate = torch.sigmoid(
                self.fusion_gate(torch.cat(
                    [self_tokens, cross_tokens], dim=-1
                )).permute(0, 2, 1)
            )
            dec_out = self_pred + gate * delta
        else:
            source_indices = torch.empty(
                x_enc.shape[0], self.n_vars, 0, dtype=torch.long,
                device=x_enc.device
            )
            source_weights = x_enc.new_empty(x_enc.shape[0], self.n_vars, 0)
            route_scores = x_enc.new_empty(
                x_enc.shape[0], self.n_vars, self.n_vars
            )
            cross_attention = None
            gate = torch.zeros_like(self_pred)
            dec_out = self_pred

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        routing = {
            'periods': list(self.channel_periods),
            'selected_sources': source_indices,
            'selected_weights': source_weights,
            'route_scores': route_scores,
            'cross_attention': cross_attention,
            'fusion_gate': gate,
        }
        return dec_out, routing

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, routing = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], routing
        return dec_out[:, -self.pred_len:, :]
