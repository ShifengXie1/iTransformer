"""Decomposition-aware, utility-routed multivariate forecasting.

The model performs a single DFT decomposition on the original input history and
does not construct temporal patches or a multi-scale input pyramid.  A strong
trend predictor forms the channel-independent baseline, a small gated
fluctuation branch completes the self forecast, and cross-variate information
is used only as a sparse, gated residual correction.
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_int_list(value, default: Iterable[int]) -> List[int]:
    if value is None:
        values = list(default)
    elif isinstance(value, str):
        values = [int(item.strip()) for item in value.split(',') if item.strip()]
    else:
        values = [int(item) for item in value]
    if not values:
        raise ValueError('Expected at least one integer value')
    return values


class DFTSeriesDecomposition(nn.Module):
    """Split each original input series into dominant-frequency and trend parts.

    This is the single-scale DFT decomposition used by TimeMixer, with the FFT
    explicitly applied along the temporal dimension.  The DC component remains
    in the trend, while the strongest non-DC frequencies form the fluctuation.
    """

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError('decomp_dft_top_k must be at least 1')

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, C, L]. Decomposition is independent per sample and variate.
        spectrum = torch.fft.rfft(x, dim=-1)
        amplitudes = spectrum.abs()
        frequency_mask = torch.zeros_like(amplitudes, dtype=torch.bool)

        # Bin 0 is the DC component and is assigned to the trend.  Clamp top-k
        # for very short input histories that have fewer available frequencies.
        non_dc_bins = amplitudes.shape[-1] - 1
        selected_k = min(self.top_k, non_dc_bins)
        if selected_k > 0:
            selected = torch.topk(
                amplitudes[..., 1:], k=selected_k, dim=-1
            ).indices + 1
            frequency_mask.scatter_(-1, selected, True)

        seasonal_spectrum = spectrum * frequency_mask.to(spectrum.dtype)
        fluctuation = torch.fft.irfft(
            seasonal_spectrum, n=x.shape[-1], dim=-1
        )
        trend = x - fluctuation
        return trend, fluctuation, frequency_mask


class CausalDepthwiseBlock(nn.Module):
    """Residual causal temporal block without temporal patching."""

    def __init__(self, hidden: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = 2 * int(dilation)
        self.depthwise = nn.Conv1d(
            hidden, hidden, kernel_size=3, dilation=dilation,
            groups=hidden, bias=True
        )
        self.pointwise_in = nn.Conv1d(hidden, hidden * 2, kernel_size=1)
        self.pointwise_out = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.depthwise(F.pad(x, (self.left_padding, 0)))
        value, gate = self.pointwise_in(y).chunk(2, dim=1)
        y = value * torch.sigmoid(gate)
        y = self.pointwise_out(self.dropout(y))
        return self.norm(residual + self.dropout(y))


class TrendSelfPredictor(nn.Module):
    """Strong channel-independent baseline driven by the decomposed trend."""

    def __init__(
            self, seq_len: int, pred_len: int, hidden: int, dropout: float
    ):
        super().__init__()
        # The direct temporal map is deliberately the main path.  A small
        # nonlinear residual head improves regime adaptation without replacing
        # the stable linear extrapolation bias that works well for trends.
        self.direct_head = nn.Linear(seq_len, pred_len)
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.residual_head = nn.Linear(hidden, pred_len)
        nn.init.xavier_uniform_(self.residual_head.weight, gain=0.1)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
            self, trend: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # trend: [B, C, L]. The same weights are shared across all variates.
        token = self.token_encoder(trend)
        prediction = self.direct_head(trend) + self.residual_head(token)
        return token, prediction


class FluctuationEncoder(nn.Module):
    """Shared channel-independent TCN for detrended fluctuations."""

    def __init__(
            self, seq_len: int, pred_len: int, hidden: int,
            layers: int, dropout: float
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(1, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalDepthwiseBlock(hidden, 2 ** layer, dropout)
            for layer in range(max(1, layers))
        ])
        self.token_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )
        # The direct temporal map is a strong no-patch baseline.  The TCN head
        # adds nonlinear local-dynamics correction.
        self.direct_head = nn.Linear(seq_len, pred_len)
        self.token_head = nn.Linear(hidden, pred_len)

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, L]; variables are folded into the batch dimension.
        batch, variates, length = x.shape
        y = x.reshape(batch * variates, 1, length)
        hidden = self.input_projection(y)
        for block in self.blocks:
            hidden = block(hidden)
        pooled = torch.cat(
            [hidden[..., -1], hidden.mean(dim=-1)], dim=-1
        )
        token = self.token_projection(pooled).reshape(batch, variates, -1)
        prediction = self.direct_head(x)
        prediction = prediction + self.token_head(token)
        return token, prediction


class ComponentLagRouter(nn.Module):
    """Sparse directed routing over (source variate, component, lag)."""

    def __init__(
            self, hidden: int, lags: Iterable[int], top_k: int,
            temperature: float, dropout: float, variate_top_k: int = 8
    ):
        super().__init__()
        self.lags = [int(lag) for lag in lags]
        self.register_buffer(
            'lag_values', torch.tensor(self.lags, dtype=torch.long)
        )
        self.top_k = int(top_k)
        self.variate_top_k = int(variate_top_k)
        self.temperature = float(temperature)
        if self.top_k < 1:
            raise ValueError('decomp_top_k must be at least 1')
        if self.variate_top_k < 1:
            raise ValueError('decomp_variate_top_k must be at least 1')
        if self.temperature <= 0:
            raise ValueError('decomp_router_temperature must be positive')
        if min(self.lags) < 0:
            raise ValueError('decomposition lags must be non-negative')

        self.component_embedding = nn.Embedding(2, hidden)
        self.lag_embedding = nn.Embedding(len(self.lags), hidden)
        self.query_projection = nn.Linear(hidden, hidden, bias=False)
        self.key_projection = nn.Linear(hidden, hidden, bias=False)
        self.value_projection = nn.Linear(hidden, hidden, bias=False)
        self.query_norm = nn.LayerNorm(hidden)
        self.source_norm = nn.LayerNorm(hidden)
        self.component_bias = nn.Parameter(torch.zeros(2, 2))
        self.lag_bias = nn.Parameter(torch.zeros(len(self.lags)))
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, query_tokens: torch.Tensor, source_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # query_tokens:  [B, C, A=2, D]
        # source_tokens: [B, C, A=2, M, D]
        batch, variates, components, hidden = query_tokens.shape
        num_lags = source_tokens.shape[-2]
        if components != 2 or num_lags != len(self.lags):
            raise ValueError('Unexpected component or lag token shape')
        if variates < 2:
            empty_index = torch.empty(
                batch, variates, components, 0, dtype=torch.long,
                device=query_tokens.device
            )
            empty_weight = query_tokens.new_empty(
                batch, variates, components, 0
            )
            empty_value = query_tokens.new_empty(
                batch, variates, components, 0, hidden
            )
            return {
                'indices': empty_index,
                'weights': empty_weight,
                'values': empty_value,
                'scores': query_tokens.new_empty(batch, 0, 0),
                'source_variates': empty_index,
                'source_components': empty_index,
                'source_lags': empty_index,
            }

        component_ids = torch.arange(components, device=query_tokens.device)
        lag_ids = torch.arange(num_lags, device=query_tokens.device)

        # Stage 1: shortlist source variables using one pooled token per
        # variable. This bounds the expensive component-lag scoring tensor by
        # C * variate_top_k instead of C * C * components * lags.
        variable_queries = query_tokens.mean(dim=2)
        variable_sources = source_tokens.mean(dim=(2, 3))
        variable_queries = self.query_projection(
            self.dropout(self.query_norm(variable_queries))
        )
        variable_keys = self.key_projection(
            self.dropout(self.source_norm(variable_sources))
        )
        variable_scores = torch.matmul(
            variable_queries, variable_keys.transpose(-1, -2)
        ) / math.sqrt(hidden)
        diagonal = torch.eye(
            variates, dtype=torch.bool, device=query_tokens.device
        ).unsqueeze(0)
        variable_scores = variable_scores.masked_fill(diagonal, -torch.inf)
        selected_variates = min(self.variate_top_k, variates - 1)
        _, variable_indices = torch.topk(
            variable_scores, k=selected_variates, dim=-1
        )  # [B, C, V]

        source_bank = source_tokens.unsqueeze(1).expand(
            -1, variates, -1, -1, -1, -1
        )
        source_gather = variable_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        source_gather = source_gather.expand(
            -1, -1, -1, components, num_lags, hidden
        )
        candidates = torch.gather(
            source_bank, dim=2, index=source_gather
        )  # [B, C, V, source_component, lag, D]
        candidates = candidates + self.component_embedding(component_ids).view(
            1, 1, 1, components, 1, hidden
        )
        candidates = candidates + self.lag_embedding(lag_ids).view(
            1, 1, 1, 1, num_lags, hidden
        )

        query = query_tokens + self.component_embedding(component_ids).view(
            1, 1, components, hidden
        )
        query = self.query_projection(self.dropout(self.query_norm(query)))
        candidate_keys = self.key_projection(
            self.dropout(self.source_norm(candidates))
        )
        candidate_values = self.value_projection(candidates)
        scores = (
            query.unsqueeze(3).unsqueeze(4).unsqueeze(5)
            * candidate_keys.unsqueeze(2)
        ).sum(dim=-1) / math.sqrt(hidden)

        # Component relation and lag priors are shared across variables, which
        # preserves channel permutation equivariance.
        scores = scores + self.component_bias.view(
            1, 1, components, 1, components, 1
        )
        scores = scores + self.lag_bias.view(
            1, 1, 1, 1, 1, num_lags
        )
        candidate_count = selected_variates * components * num_lags
        scores = scores.reshape(batch, variates, components, candidate_count)
        values = candidate_values.unsqueeze(2).expand(
            -1, -1, components, -1, -1, -1, -1
        ).reshape(batch, variates, components, candidate_count, hidden)

        selected_k = min(self.top_k, candidate_count)
        top_scores, top_indices = torch.topk(scores, k=selected_k, dim=-1)
        top_weights = torch.softmax(
            top_scores / self.temperature, dim=-1
        )
        gather_indices = top_indices.unsqueeze(-1).expand(
            -1, -1, -1, -1, hidden
        )
        selected_values = torch.gather(
            values, dim=3, index=gather_indices
        )

        source_variate_candidates = variable_indices.unsqueeze(-1).unsqueeze(-1)
        source_variate_candidates = source_variate_candidates.expand(
            -1, -1, -1, components, num_lags
        ).reshape(batch, variates, candidate_count)
        source_variate_candidates = source_variate_candidates.unsqueeze(2).expand(
            -1, -1, components, -1
        )
        source_component_candidates = component_ids.view(
            1, 1, 1, 1, components, 1
        ).expand(
            batch, variates, components, selected_variates,
            components, num_lags
        ).reshape(batch, variates, components, candidate_count)
        source_lag_candidates = self.lag_values.view(
            1, 1, 1, 1, 1, num_lags
        ).expand(
            batch, variates, components, selected_variates,
            components, num_lags
        ).reshape(batch, variates, components, candidate_count)

        return {
            'indices': top_indices,
            'weights': top_weights,
            'values': selected_values,
            'scores': scores,
            'variable_scores': variable_scores,
            'candidate_variates': variable_indices,
            'source_variates': torch.gather(
                source_variate_candidates, dim=3, index=top_indices
            ),
            'source_components': torch.gather(
                source_component_candidates, dim=3, index=top_indices
            ),
            'source_lags': torch.gather(
                source_lag_candidates, dim=3, index=top_indices
            ),
        }


class CrossMessageRefiner(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        return self.norm(message + self.ffn(message))


class Model(nn.Module):
    """Single-scale DFT decomposition and utility-routed residual fusion."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.output_attention = bool(configs.output_attention)
        self.use_norm = bool(configs.use_norm)
        self.hidden = int(getattr(
            configs, 'decomp_hidden', min(int(configs.d_model), 128)
        ))

        lags = _parse_int_list(
            getattr(configs, 'decomp_lags', None), (0, 1, 2, 4, 8)
        )
        if max(lags) >= self.seq_len:
            raise ValueError(
                'Every decomp lag must be smaller than seq_len: '
                f'max lag {max(lags)}, seq_len {self.seq_len}'
            )
        self.lags = lags
        self.decomposition = DFTSeriesDecomposition(
            getattr(configs, 'decomp_dft_top_k', 5)
        )

        self.trend_predictor = TrendSelfPredictor(
            self.seq_len, self.pred_len, self.hidden, configs.dropout
        )
        self.fluctuation_encoder = FluctuationEncoder(
            self.seq_len, self.pred_len, self.hidden,
            getattr(configs, 'decomp_tcn_layers', configs.e_layers),
            configs.dropout,
        )
        self.self_fluctuation_gate = nn.Sequential(
            nn.LayerNorm(self.hidden * 2),
            nn.Linear(self.hidden * 2, self.pred_len),
        )
        self.fluctuation_gate_bias = float(getattr(
            configs, 'decomp_fluctuation_gate_bias', -2.0
        ))
        nn.init.zeros_(self.self_fluctuation_gate[-1].weight)
        nn.init.constant_(
            self.self_fluctuation_gate[-1].bias,
            self.fluctuation_gate_bias,
        )

        # Whole-history lag projections: no temporal patch construction.
        self.trend_lag_projection = nn.Linear(self.seq_len, self.hidden)
        self.fluctuation_lag_projection = nn.Linear(
            self.seq_len, self.hidden
        )
        self.router = ComponentLagRouter(
            self.hidden,
            lags,
            getattr(configs, 'decomp_top_k', 3),
            getattr(configs, 'decomp_router_temperature', 1.0),
            configs.dropout,
            getattr(configs, 'decomp_variate_top_k', 8),
        )
        self.message_refiner = CrossMessageRefiner(
            self.hidden, configs.dropout
        )
        self.delta_heads = nn.ModuleList([
            nn.Linear(self.hidden, self.pred_len) for _ in range(2)
        ])
        self.gate_heads = nn.ModuleList([
            nn.Linear(self.hidden * 3, self.pred_len) for _ in range(2)
        ])
        self.cross_gate_bias = float(getattr(
            configs, 'decomp_cross_gate_bias', -2.5
        ))

        for head in self.delta_heads:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)
        for head in self.gate_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, self.cross_gate_bias)

        self.aux_weights = {
            'trend': float(getattr(configs, 'decomp_trend_loss', 0.2)),
            'fluctuation': float(getattr(
                configs, 'decomp_fluctuation_loss', 0.05
            )),
            'self': float(getattr(configs, 'decomp_self_loss', 0.1)),
            'utility': float(getattr(configs, 'decomp_utility_loss', 0.05)),
            'safe': float(getattr(configs, 'decomp_safe_loss', 0.05)),
            'entropy': float(getattr(configs, 'decomp_entropy_loss', 1e-3)),
        }
        self._aux_state: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _lag_sequence(x: torch.Tensor, lag: int) -> torch.Tensor:
        if lag == 0:
            return x
        return F.pad(x[..., :-lag], (lag, 0), mode='replicate')

    def _lag_tokens(
            self, trend: torch.Tensor, fluctuation: torch.Tensor
    ) -> torch.Tensor:
        trend_tokens = []
        fluctuation_tokens = []
        for lag in self.lags:
            trend_tokens.append(self.trend_lag_projection(
                self._lag_sequence(trend, lag)
            ))
            fluctuation_tokens.append(self.fluctuation_lag_projection(
                self._lag_sequence(fluctuation, lag)
            ))
        trend_tokens = torch.stack(trend_tokens, dim=-2)
        fluctuation_tokens = torch.stack(fluctuation_tokens, dim=-2)
        return torch.stack(
            [trend_tokens, fluctuation_tokens], dim=2
        )  # [B, C, 2, M, D]

    def _component_corrections(
            self, self_tokens: torch.Tensor, messages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deltas = []
        gates = []
        contributions = []
        # Each correction gate sees the complete target self state (trend and
        # fluctuation) before deciding whether an external message is useful.
        self_context = self_tokens.flatten(start_dim=2)
        for component in range(2):
            message = messages[:, :, component, :]
            delta = self.delta_heads[component](message)
            gate_input = torch.cat([
                self_context, message
            ], dim=-1)
            gate = torch.sigmoid(self.gate_heads[component](gate_input))
            deltas.append(delta)
            gates.append(gate)
            contributions.append(gate * delta)
        return (
            torch.stack(deltas, dim=2),
            torch.stack(gates, dim=2),
            torch.stack(contributions, dim=2),
        )

    @staticmethod
    def _denormalize(
            prediction: torch.Tensor,
            means: torch.Tensor,
            stdev: torch.Tensor,
    ) -> torch.Tensor:
        # prediction: [B, C, ..., H]
        extra_dims = prediction.ndim - 3
        shape = [means.shape[0], means.shape[1]] + [1] * (extra_dims + 1)
        return prediction * stdev.reshape(shape) + means.reshape(shape)

    @staticmethod
    def _rescale_residual(
            residual: torch.Tensor, stdev: torch.Tensor
    ) -> torch.Tensor:
        # Residual corrections are scaled back without adding the series mean.
        extra_dims = residual.ndim - 3
        shape = [stdev.shape[0], stdev.shape[1]] + [1] * (extra_dims + 1)
        return residual * stdev.reshape(shape)

    def forecast(
            self, x_enc: torch.Tensor, x_mark_enc=None,
            x_dec=None, x_mark_dec=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x_enc.shape[1] != self.seq_len:
            raise ValueError(
                f'Expected seq_len={self.seq_len}, got {x_enc.shape[1]}'
            )
        x = x_enc
        if self.use_norm:
            means_time = x.mean(dim=1, keepdim=True).detach()
            centered = x - means_time
            stdev_time = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x = centered / stdev_time
            means = means_time[:, 0, :]
            stdev = stdev_time[:, 0, :]
        else:
            means = x.new_zeros(x.shape[0], x.shape[2])
            stdev = x.new_ones(x.shape[0], x.shape[2])

        # All component encoders operate on [B, C, L] and share their weights
        # across C, preserving strict channel independence in the self path.
        x_channel_first = x.transpose(1, 2)
        trend, fluctuation, frequency_mask = self.decomposition(
            x_channel_first
        )
        trend_token, trend_prediction = self.trend_predictor(trend)
        fluctuation_token, fluctuation_prediction = (
            self.fluctuation_encoder(fluctuation)
        )
        self_tokens = torch.stack(
            [trend_token, fluctuation_token], dim=2
        )
        self_fluctuation_gate = torch.sigmoid(self.self_fluctuation_gate(
            torch.cat([trend_token, fluctuation_token], dim=-1)
        ))
        fluctuation_contribution = (
            self_fluctuation_gate * fluctuation_prediction
        )
        self_prediction = trend_prediction + fluctuation_contribution

        if x.shape[2] > 1:
            source_tokens = self._lag_tokens(trend, fluctuation)
            routed = self.router(self_tokens, source_tokens)
            weighted_message = (
                routed['weights'].unsqueeze(-1) * routed['values']
            ).sum(dim=-2)
            messages = self.message_refiner(weighted_message)
            deltas, gates, contributions = self._component_corrections(
                self_tokens, messages
            )
            cross_correction = contributions.sum(dim=2)
            prediction = self_prediction + cross_correction
        else:
            routed = self.router(
                self_tokens,
                self._lag_tokens(trend, fluctuation),
            )
            messages = torch.zeros_like(self_tokens)
            deltas = self_prediction.new_zeros(
                *self_prediction.shape[:2], 2, self.pred_len
            )
            gates = torch.zeros_like(deltas)
            contributions = torch.zeros_like(deltas)
            cross_correction = torch.zeros_like(self_prediction)
            prediction = self_prediction

        final_output = self._denormalize(
            prediction, means, stdev
        ).transpose(1, 2)
        self_output = self._denormalize(
            self_prediction, means, stdev
        ).transpose(1, 2)
        trend_output = self._denormalize(
            trend_prediction, means, stdev
        ).transpose(1, 2)
        fluctuation_self_output = self._rescale_residual(
            fluctuation_contribution, stdev
        ).transpose(1, 2)
        fluctuation_candidate_output = self._rescale_residual(
            fluctuation_prediction, stdev
        ).transpose(1, 2)
        cross_correction_output = self._rescale_residual(
            cross_correction, stdev
        ).transpose(1, 2)

        if routed['weights'].numel():
            entropy_loss = -(
                routed['weights'] * torch.log(routed['weights'] + 1e-8)
            ).sum(dim=-1).mean()
        else:
            entropy_loss = trend.new_zeros(())

        # Retain graph-connected tensors only during training.  Utility loss is
        # computed on demand by auxiliary_loss(), after the target is known.
        self._aux_state = {
            'final': final_output,
            'self': self_output,
            'trend': trend_output,
            'self_fluctuation': fluctuation_self_output,
            'cross_correction': cross_correction_output,
            'self_normalized': self_prediction,
            'self_tokens': self_tokens,
            'full_contributions': contributions,
            'routed_values': routed['values'],
            'routed_weights': routed['weights'],
            'means': means,
            'stdev': stdev,
            'entropy': entropy_loss,
        } if self.training else None

        diagnostics = {
            'trend': trend,
            'fluctuation': fluctuation,
            'decomposition_mask': frequency_mask,
            'trend_prediction': trend_output,
            'fluctuation_candidate': fluctuation_candidate_output,
            'fluctuation_self_correction': fluctuation_self_output,
            'self_fluctuation_gate': self_fluctuation_gate,
            'self_prediction': self_output,
            'selected_sources': routed['source_variates'],
            'selected_components': routed['source_components'],
            'selected_lags': routed['source_lags'],
            'selected_weights': routed['weights'],
            'route_scores': routed['scores'],
            'variable_route_scores': routed.get('variable_scores'),
            'cross_messages': messages,
            'cross_deltas': deltas,
            'cross_fusion_gates': gates,
            'fusion_gates': gates,
            'cross_correction': cross_correction_output,
        }
        return final_output, diagnostics

    def auxiliary_loss(self, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return component, self, safety, utility and routing losses.

        ``target`` must already be restricted to the prediction horizon.  For
        an MS task, the final target channel is aligned with the final model
        channel.  Trend and fluctuation targets use the same single-scale DFT
        split as the input.  Leave-one-routed-source-out predictions supervise
        routing by marginal forecasting utility rather than similarity.
        """
        state = self._aux_state
        if state is None:
            zero = target.new_zeros(())
            return {'total': zero}

        target_channels = target.shape[-1]
        model_channels = state['final'].shape[-1]
        if target_channels > model_channels:
            raise ValueError('Target has more channels than model prediction')
        channel_slice = slice(model_channels - target_channels, model_channels)
        final = state['final'][..., channel_slice]
        self_output = state['self'][..., channel_slice]
        trend_output = state['trend'][..., channel_slice]
        fluctuation_output = state['self_fluctuation'][..., channel_slice]
        cross_correction = state['cross_correction'][..., channel_slice]
        target = target[..., -target_channels:]

        means = state['means'][:, channel_slice]
        stdev = state['stdev'][:, channel_slice]
        target_channel_first = target.transpose(1, 2)
        target_normalized = (
            target_channel_first - means.unsqueeze(-1)
        ) / stdev.unsqueeze(-1)
        target_trend_normalized, target_fluctuation_normalized, _ = (
            self.decomposition(target_normalized)
        )
        target_trend = (
            target_trend_normalized * stdev.unsqueeze(-1)
            + means.unsqueeze(-1)
        ).transpose(1, 2)
        target_fluctuation = (
            target_fluctuation_normalized * stdev.unsqueeze(-1)
        ).transpose(1, 2)

        trend_loss = F.mse_loss(trend_output, target_trend)
        fluctuation_loss = F.mse_loss(
            fluctuation_output, target_fluctuation
        )
        self_loss = F.mse_loss(self_output, target)
        # Stop the safety objective at the self baseline.  It therefore teaches
        # only the cross-variate path to help rather than perturbing the strong
        # channel-independent predictor when a correction is harmful.
        safe_prediction = self_output.detach() + cross_correction
        final_error = (safe_prediction - target).pow(2)
        self_error = (self_output.detach() - target).pow(2)
        safe_loss = F.relu(final_error - self_error).mean()
        utility_loss = target.new_zeros(())

        routed_weights = state['routed_weights']
        selected_k = routed_weights.shape[-1]
        if selected_k > 0 and self.aux_weights['utility'] > 0:
            values = state['routed_values']
            leave_one_out_messages = []
            for removed in range(selected_k):
                keep = torch.ones_like(routed_weights)
                keep[..., removed] = 0
                weights = routed_weights * keep
                denominator = weights.sum(dim=-1, keepdim=True)
                weights = torch.where(
                    denominator > 0,
                    weights / denominator.clamp_min(1e-8),
                    weights,
                )
                message = (weights.unsqueeze(-1) * values).sum(dim=-2)
                leave_one_out_messages.append(
                    self.message_refiner(message)
                )
            # [B, C, A, K, D]
            leave_one_out_messages = torch.stack(
                leave_one_out_messages, dim=3
            )

            loo_contributions = []
            self_context = state['self_tokens'].flatten(start_dim=2)
            self_context = self_context.unsqueeze(2).expand(
                -1, -1, selected_k, -1
            )
            for component in range(2):
                message = leave_one_out_messages[:, :, component, :, :]
                delta = self.delta_heads[component](message)
                gate = torch.sigmoid(self.gate_heads[component](
                    torch.cat([self_context, message], dim=-1)
                ))
                loo_contributions.append(gate * delta)
            loo_contributions = torch.stack(
                loo_contributions, dim=2
            )  # [B, C, A, K, H]

            full_contributions = state['full_contributions']
            total_full = full_contributions.sum(dim=2)
            loo_prediction = (
                state['self_normalized'].unsqueeze(2).unsqueeze(3)
                + total_full.unsqueeze(2).unsqueeze(3)
                - full_contributions.unsqueeze(3)
                + loo_contributions
            )
            loo_prediction = self._denormalize(
                loo_prediction, state['means'], state['stdev']
            )
            loo_prediction = loo_prediction[:, channel_slice, ...]
            target_cf = target.transpose(1, 2).unsqueeze(2).unsqueeze(3)
            loo_error = (loo_prediction - target_cf).pow(2).mean(dim=-1)
            full_error = (
                final.transpose(1, 2) - target.transpose(1, 2)
            ).pow(2).mean(dim=-1).unsqueeze(2).unsqueeze(3)
            marginal_utility = F.relu(loo_error - full_error).detach()
            utility_sum = marginal_utility.sum(dim=-1, keepdim=True)
            valid = utility_sum.squeeze(-1) > 1e-8
            utility_target = marginal_utility / utility_sum.clamp_min(1e-8)
            selected_weights = routed_weights[:, channel_slice, ...]
            cross_entropy = -(
                utility_target * torch.log(selected_weights + 1e-8)
            ).sum(dim=-1)
            if valid.any():
                utility_loss = cross_entropy[valid].mean()

        losses = {
            'trend': trend_loss,
            'fluctuation': fluctuation_loss,
            'self': self_loss,
            'utility': utility_loss,
            'safe': safe_loss,
            'entropy': state['entropy'],
        }
        total = sum(
            self.aux_weights[name] * loss for name, loss in losses.items()
        )
        losses['total'] = total
        return losses

    def forward(
            self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None
    ):
        output, diagnostics = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        if self.output_attention:
            return output[:, -self.pred_len:, :], diagnostics
        return output[:, -self.pred_len:, :]
