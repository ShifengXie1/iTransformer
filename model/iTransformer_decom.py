"""Channel-independent TimeMixer with sparse cross-variate residual repair.

The self path follows the original channel-independent TimeMixer forecasting
backbone: each scale is normalized separately, every PDM block uses centered
moving-average decomposition, seasonal information is mixed bottom-up, and
trend information is mixed top-down. Cross-variate information never enters
that self path. It is routed only after the last PDM block and can change the
forecast solely through a small, gated residual correction.
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


class MovingAverageDecomposition(nn.Module):
    """Original TimeMixer centered moving-average decomposition."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError('decomp_moving_avg must be a positive odd integer')

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Any number of leading dimensions is allowed; time is always last.
        length = x.shape[-1]
        flattened = x.reshape(-1, 1, length)
        half_window = (self.kernel_size - 1) // 2
        padded = F.pad(
            flattened, (half_window, half_window), mode='replicate'
        )
        trend = F.avg_pool1d(
            padded, kernel_size=self.kernel_size, stride=1
        ).reshape_as(x)
        seasonal = x - trend
        return trend, seasonal


class ChannelIndependentEmbedding(nn.Module):
    """TimeMixer value/temporal embedding shared by every variate."""

    def __init__(self, hidden: int, freq: str, dropout: float):
        super().__init__()
        self.hidden = int(hidden)
        self.value_embedding = nn.Conv1d(
            1, hidden, kernel_size=3, padding=1,
            padding_mode='circular', bias=False
        )
        nn.init.kaiming_normal_(
            self.value_embedding.weight, mode='fan_in', nonlinearity='leaky_relu'
        )
        feature_dims = {
            'h': 4, 't': 5, 's': 6, 'ms': 7, 'm': 1,
            'a': 1, 'w': 2, 'd': 3, 'b': 3,
        }
        self.temporal_embedding = nn.Linear(
            feature_dims.get(str(freq).lower(), 4), hidden, bias=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, x: torch.Tensor, time_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # x: [B,C,T]. Fold C into the batch so convolution remains independent.
        batch, variates, length = x.shape
        values = x.reshape(batch * variates, 1, length)
        values = self.value_embedding(values).transpose(1, 2)
        values = values.reshape(batch, variates, length, self.hidden)
        if time_features is not None:
            temporal = self.temporal_embedding(time_features).unsqueeze(1)
            values = values + temporal
        return self.dropout(values)


class ScaleNormalize(nn.Module):
    """TimeMixer Normalize layer for tensors shaped [B,C,T]."""

    def __init__(self, num_features: int, enabled: bool, eps: float = 1e-5):
        super().__init__()
        self.enabled = bool(enabled)
        self.eps = float(eps)
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self.mean: Optional[torch.Tensor] = None
        self.stdev: Optional[torch.Tensor] = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = x.mean(dim=-1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        if not self.enabled:
            return x
        weight = self.affine_weight.view(1, -1, 1)
        bias = self.affine_bias.view(1, -1, 1)
        return (x - self.mean) / self.stdev * weight + bias

    def _affine_shape(self, x: torch.Tensor) -> List[int]:
        return [1, self.affine_weight.shape[0]] + [1] * (x.ndim - 2)

    def _stat_shape(self, x: torch.Tensor) -> List[int]:
        if self.mean is None:
            raise RuntimeError('normalize() must be called before denormalize()')
        return [self.mean.shape[0], self.mean.shape[1]] + [1] * (x.ndim - 2)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.mean is None or self.stdev is None:
            raise RuntimeError('normalize() must be called before denormalize()')
        weight = self.affine_weight.reshape(self._affine_shape(x))
        bias = self.affine_bias.reshape(self._affine_shape(x))
        mean = self.mean.reshape(self._stat_shape(x))
        stdev = self.stdev.reshape(self._stat_shape(x))
        return (x - bias) / (weight + self.eps * self.eps) * stdev + mean

    def rescale_residual(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.stdev is None:
            raise RuntimeError('normalize() must be called before rescaling')
        weight = self.affine_weight.reshape(self._affine_shape(x))
        stdev = self.stdev.reshape(self._stat_shape(x))
        return x / (weight + self.eps * self.eps) * stdev


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing from fine seasonal patterns to coarse scales."""

    def __init__(self, scale_lengths: List[int]):
        super().__init__()
        self.down_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scale_lengths[index], scale_lengths[index + 1]),
                nn.GELU(),
                nn.Linear(
                    scale_lengths[index + 1], scale_lengths[index + 1]
                ),
            )
            for index in range(len(scale_lengths) - 1)
        ])

    def forward(self, season_list: List[torch.Tensor]) -> List[torch.Tensor]:
        # Each item: [B, C, D, T_scale].
        outputs = [season_list[0]]
        current = season_list[0]
        for index, layer in enumerate(self.down_layers):
            current = season_list[index + 1] + layer(current)
            outputs.append(current)
        return outputs


class MultiScaleTrendMixing(nn.Module):
    """Top-down mixing from coarse trends to fine temporal scales."""

    def __init__(self, scale_lengths: List[int]):
        super().__init__()
        self.up_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scale_lengths[index + 1], scale_lengths[index]),
                nn.GELU(),
                nn.Linear(scale_lengths[index], scale_lengths[index]),
            )
            for index in range(len(scale_lengths) - 1)
        ])

    def forward(self, trend_list: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs: List[Optional[torch.Tensor]] = [None] * len(trend_list)
        current = trend_list[-1]
        outputs[-1] = current
        for index in reversed(range(len(self.up_layers))):
            current = trend_list[index] + self.up_layers[index](current)
            outputs[index] = current
        return outputs  # type: ignore[return-value]


class PastDecomposableMixing(nn.Module):
    """One multi-scale TimeMixer PDM block with channel-shared weights."""

    def __init__(
            self, scale_lengths: List[int], hidden: int, d_ff: int,
            moving_avg: int
    ):
        super().__init__()
        self.decomposition = MovingAverageDecomposition(moving_avg)
        self.season_mixing = MultiScaleSeasonMixing(scale_lengths)
        self.trend_mixing = MultiScaleTrendMixing(scale_lengths)
        self.output_ffn = nn.Sequential(
            nn.Linear(hidden, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, hidden),
        )

    def forward(
            self, x_list: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        trend_list = []
        season_list = []
        for x in x_list:
            # [B, C, T, D] -> [B, C, D, T] for temporal decomposition.
            component_first = x.permute(0, 1, 3, 2)
            trend, season = self.decomposition(component_first)
            trend_list.append(trend)
            season_list.append(season)

        mixed_season = self.season_mixing(season_list)
        mixed_trend = self.trend_mixing(trend_list)

        outputs = []
        trend_sequences = []
        season_sequences = []
        for index, (original, trend, season) in enumerate(zip(
                x_list, mixed_trend, mixed_season
        )):
            trend_time_first = trend.permute(0, 1, 3, 2)
            season_time_first = season.permute(0, 1, 3, 2)
            mixed = trend_time_first + season_time_first
            outputs.append(original + self.output_ffn(mixed))
            trend_sequences.append(trend_time_first)
            season_sequences.append(season_time_first)
        return outputs, trend_sequences, season_sequences


class ChannelIndependentTimeMixer(nn.Module):
    """Full multi-scale TimeMixer self predictor for each variate."""

    def __init__(
            self, seq_len: int, pred_len: int, hidden: int, d_ff: int,
            mixing_layers: int, down_sampling_layers: int,
            down_sampling_window: int, down_sampling_method: str,
            num_variates: int, use_norm: bool, moving_avg: int,
            lags: Iterable[int], freq: str, dropout: float
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.hidden = int(hidden)
        self.down_sampling_layers = int(down_sampling_layers)
        self.down_sampling_window = int(down_sampling_window)
        self.down_sampling_method = str(down_sampling_method).lower()
        self.lags = [int(lag) for lag in lags]

        if int(mixing_layers) < 1:
            raise ValueError('decomp_mixing_layers must be at least 1')
        if self.down_sampling_layers < 1:
            raise ValueError('decomp_down_sampling_layers must be at least 1')
        if self.down_sampling_window < 2:
            raise ValueError('decomp_down_sampling_window must be at least 2')
        if self.down_sampling_method not in {'avg', 'max'}:
            raise ValueError(
                'decomp_down_sampling_method must be either avg or max'
            )

        self.scale_lengths = [self.seq_len]
        for _ in range(self.down_sampling_layers):
            next_length = self.scale_lengths[-1] // self.down_sampling_window
            if next_length < 2:
                raise ValueError(
                    'Downsampling produces a scale shorter than two steps'
                )
            self.scale_lengths.append(next_length)
        if max(self.lags) >= self.scale_lengths[-1]:
            raise ValueError(
                'Every local lag must be smaller than the coarsest scale: '
                f'max lag {max(self.lags)}, coarsest length '
                f'{self.scale_lengths[-1]}'
            )

        self.num_scales = len(self.scale_lengths)
        self.normalize_layers = nn.ModuleList([
            ScaleNormalize(num_variates, enabled=use_norm)
            for _ in self.scale_lengths
        ])
        self.raw_decomposition = MovingAverageDecomposition(moving_avg)
        self.input_embedding = ChannelIndependentEmbedding(
            hidden, freq, dropout
        )
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                self.scale_lengths, hidden, d_ff, moving_avg
            )
            for _ in range(int(mixing_layers))
        ])

        self.predict_layers = nn.ModuleList([
            nn.Linear(length, pred_len) for length in self.scale_lengths
        ])
        self.output_projection = nn.Linear(hidden, 1)

        # Component tokens retain trend/season identity and scale identity.
        self.component_poolers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden),
            )
            for _ in range(2)
        ])
        self.scale_embedding = nn.Embedding(self.num_scales, hidden)
        self.target_scale_scorers = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(2)
        ])

    def _downsample(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = [x]
        current = x
        for _ in range(self.down_sampling_layers):
            if self.down_sampling_method == 'avg':
                current = F.avg_pool1d(
                    current,
                    kernel_size=self.down_sampling_window,
                    stride=self.down_sampling_window,
                )
            else:
                current = F.max_pool1d(
                    current,
                    kernel_size=self.down_sampling_window,
                    stride=self.down_sampling_window,
                )
            outputs.append(current)
        return outputs

    def _downsample_time_features(
            self, time_features: Optional[torch.Tensor]
    ) -> List[Optional[torch.Tensor]]:
        if time_features is None:
            return [None] * self.num_scales
        outputs: List[Optional[torch.Tensor]] = [time_features]
        current = time_features
        for scale in range(1, self.num_scales):
            current = current[:, ::self.down_sampling_window, :]
            current = current[:, :self.scale_lengths[scale], :]
            outputs.append(current)
        return outputs

    @staticmethod
    def _lag_sequence(sequence: torch.Tensor, lag: int) -> torch.Tensor:
        # sequence: [B, C, T, D]. Missing history repeats the first value.
        if lag == 0:
            return sequence
        prefix = sequence[:, :, :1, :].expand(-1, -1, lag, -1)
        return torch.cat([prefix, sequence[:, :, :-lag, :]], dim=2)

    def _pool_component(
            self, sequence: torch.Tensor, component: int, scale: int
    ) -> torch.Tensor:
        pooled = torch.cat([
            sequence[:, :, -1, :], sequence.mean(dim=2)
        ], dim=-1)
        token = self.component_poolers[component](pooled)
        return token + self.scale_embedding.weight[scale].view(1, 1, -1)

    def forward(
            self, x: torch.Tensor,
            time_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # x: [B, C, L]. Downsampling never mixes the C dimension.
        scale_inputs = self._downsample(x)
        normalized_scales = [
            normalizer.normalize(scale)
            for normalizer, scale in zip(self.normalize_layers, scale_inputs)
        ]
        scale_time_features = self._downsample_time_features(time_features)
        encoded = [
            self.input_embedding(scale, scale_mark)
            for scale, scale_mark in zip(
                normalized_scales, scale_time_features
            )
        ]

        trend_sequences: List[torch.Tensor] = []
        season_sequences: List[torch.Tensor] = []
        for block in self.pdm_blocks:
            encoded, trend_sequences, season_sequences = block(encoded)

        scale_predictions = []
        for index, representation in enumerate(encoded):
            prediction = self.predict_layers[index](
                representation.permute(0, 1, 3, 2)
            ).permute(0, 1, 3, 2)
            scale_predictions.append(
                self.output_projection(prediction).squeeze(-1)
            )
        scale_predictions_tensor = torch.stack(scale_predictions, dim=2)
        self_prediction = scale_predictions_tensor.sum(dim=2)

        # Component order is [trend, seasonal].
        component_sequences = [trend_sequences, season_sequences]
        component_tokens = []
        source_tokens = []
        for component, sequences in enumerate(component_sequences):
            scale_tokens = []
            scale_sources = []
            for scale, sequence in enumerate(sequences):
                scale_tokens.append(self._pool_component(
                    sequence, component, scale
                ))
                lag_tokens = [
                    self._pool_component(
                        self._lag_sequence(sequence, lag), component, scale
                    )
                    for lag in self.lags
                ]
                scale_sources.append(torch.stack(lag_tokens, dim=2))
            component_tokens.append(torch.stack(scale_tokens, dim=2))
            source_tokens.append(torch.stack(scale_sources, dim=2))

        # [B, C, A=2, S, D] and [B, C, A=2, S, M, D].
        component_tokens_tensor = torch.stack(component_tokens, dim=2)
        source_tokens_tensor = torch.stack(source_tokens, dim=2)

        scale_scores = []
        for component in range(2):
            score = self.target_scale_scorers[component](
                component_tokens_tensor[:, :, component, :, :]
            ).squeeze(-1)
            scale_scores.append(score)
        scale_scores_tensor = torch.stack(scale_scores, dim=2)
        target_scale_weights = torch.softmax(scale_scores_tensor, dim=-1)
        query_tokens = (
            target_scale_weights.unsqueeze(-1) * component_tokens_tensor
        ).sum(dim=3)

        raw_trend, raw_season = self.raw_decomposition(normalized_scales[0])
        return {
            'prediction': self_prediction,
            'scale_predictions': scale_predictions_tensor,
            'query_tokens': query_tokens,
            'source_tokens': source_tokens_tensor,
            'component_tokens': component_tokens_tensor,
            'target_scale_weights': target_scale_weights,
            'input_trend': raw_trend,
            'input_seasonal': raw_season,
        }

    def denormalize(self, prediction: torch.Tensor) -> torch.Tensor:
        return self.normalize_layers[0].denormalize(prediction)

    def rescale_residual(self, residual: torch.Tensor) -> torch.Tensor:
        return self.normalize_layers[0].rescale_residual(residual)


class MultiScaleComponentLagRouter(nn.Module):
    """Sparse routing over source variable, component, scale and local lag."""

    def __init__(
            self, hidden: int, lags: Iterable[int], num_scales: int,
            down_sampling_window: int, top_k: int, temperature: float,
            dropout: float, variate_top_k: int = 8
    ):
        super().__init__()
        self.lags = [int(lag) for lag in lags]
        self.num_scales = int(num_scales)
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

        self.register_buffer(
            'lag_values', torch.tensor(self.lags, dtype=torch.long)
        )
        self.register_buffer(
            'scale_factors', torch.tensor([
                int(down_sampling_window) ** scale
                for scale in range(self.num_scales)
            ], dtype=torch.long)
        )
        self.component_embedding = nn.Embedding(2, hidden)
        self.scale_embedding = nn.Embedding(self.num_scales, hidden)
        self.lag_embedding = nn.Embedding(len(self.lags), hidden)
        self.query_projection = nn.Linear(hidden, hidden, bias=False)
        self.key_projection = nn.Linear(hidden, hidden, bias=False)
        self.value_projection = nn.Linear(hidden, hidden, bias=False)
        self.query_norm = nn.LayerNorm(hidden)
        self.source_norm = nn.LayerNorm(hidden)
        self.component_bias = nn.Parameter(torch.zeros(2, 2))
        self.scale_bias = nn.Parameter(torch.zeros(self.num_scales))
        self.lag_bias = nn.Parameter(torch.zeros(len(self.lags)))
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, query_tokens: torch.Tensor, source_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # query: [B,C,A,D], sources: [B,C,A,S,M,D].
        batch, variates, components, hidden = query_tokens.shape
        source_components = source_tokens.shape[2]
        num_scales = source_tokens.shape[3]
        num_lags = source_tokens.shape[4]
        if (
                components != 2 or source_components != 2
                or num_scales != self.num_scales
                or num_lags != len(self.lags)
        ):
            raise ValueError('Unexpected component, scale or lag token shape')

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
                'scores': query_tokens.new_empty(
                    batch, variates, components, 0
                ),
                'variable_scores': query_tokens.new_empty(
                    batch, variates, variates
                ),
                'candidate_variates': empty_index,
                'source_variates': empty_index,
                'source_components': empty_index,
                'source_scales': empty_index,
                'source_lags': empty_index,
                'source_effective_lags': empty_index,
            }

        component_ids = torch.arange(components, device=query_tokens.device)
        scale_ids = torch.arange(num_scales, device=query_tokens.device)
        lag_ids = torch.arange(num_lags, device=query_tokens.device)

        # Stage 1: shortlist external variables before expanding component,
        # scale and lag candidates.
        variable_queries = query_tokens.mean(dim=2)
        variable_sources = source_tokens.mean(dim=(2, 3, 4))
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
        )

        source_bank = source_tokens.unsqueeze(1).expand(
            -1, variates, -1, -1, -1, -1, -1
        )
        source_gather = variable_indices.unsqueeze(-1).unsqueeze(-1)
        source_gather = source_gather.unsqueeze(-1).unsqueeze(-1)
        source_gather = source_gather.expand(
            -1, -1, -1, components, num_scales, num_lags, hidden
        )
        candidates = torch.gather(source_bank, dim=2, index=source_gather)
        candidates = candidates + self.component_embedding(component_ids).view(
            1, 1, 1, components, 1, 1, hidden
        )
        candidates = candidates + self.scale_embedding(scale_ids).view(
            1, 1, 1, 1, num_scales, 1, hidden
        )
        candidates = candidates + self.lag_embedding(lag_ids).view(
            1, 1, 1, 1, 1, num_lags, hidden
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
            query.unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
            * candidate_keys.unsqueeze(2)
        ).sum(dim=-1) / math.sqrt(hidden)
        scores = scores + self.component_bias.view(
            1, 1, components, 1, components, 1, 1
        )
        scores = scores + self.scale_bias.view(
            1, 1, 1, 1, 1, num_scales, 1
        )
        scores = scores + self.lag_bias.view(
            1, 1, 1, 1, 1, 1, num_lags
        )

        candidate_count = (
            selected_variates * components * num_scales * num_lags
        )
        scores = scores.reshape(
            batch, variates, components, candidate_count
        )
        values = candidate_values.unsqueeze(2).expand(
            -1, -1, components, -1, -1, -1, -1, -1
        ).reshape(batch, variates, components, candidate_count, hidden)

        selected_k = min(self.top_k, candidate_count)
        top_scores, top_indices = torch.topk(scores, k=selected_k, dim=-1)
        top_weights = torch.softmax(
            top_scores / self.temperature, dim=-1
        )
        selected_values = torch.gather(
            values,
            dim=3,
            index=top_indices.unsqueeze(-1).expand(
                -1, -1, -1, -1, hidden
            ),
        )

        source_variate_candidates = variable_indices.unsqueeze(-1)
        source_variate_candidates = source_variate_candidates.unsqueeze(-1)
        source_variate_candidates = source_variate_candidates.unsqueeze(-1)
        source_variate_candidates = source_variate_candidates.expand(
            -1, -1, -1, components, num_scales, num_lags
        ).reshape(batch, variates, candidate_count)
        source_variate_candidates = source_variate_candidates.unsqueeze(2).expand(
            -1, -1, components, -1
        )

        metadata_shape = (
            batch, variates, components, selected_variates,
            components, num_scales, num_lags
        )
        source_component_candidates = component_ids.view(
            1, 1, 1, 1, components, 1, 1
        ).expand(*metadata_shape).reshape(
            batch, variates, components, candidate_count
        )
        source_scale_candidates = scale_ids.view(
            1, 1, 1, 1, 1, num_scales, 1
        ).expand(*metadata_shape).reshape(
            batch, variates, components, candidate_count
        )
        source_lag_candidates = self.lag_values.view(
            1, 1, 1, 1, 1, 1, num_lags
        ).expand(*metadata_shape).reshape(
            batch, variates, components, candidate_count
        )
        effective_lags = (
            self.scale_factors.view(num_scales, 1)
            * self.lag_values.view(1, num_lags)
        )
        source_effective_lag_candidates = effective_lags.view(
            1, 1, 1, 1, 1, num_scales, num_lags
        ).expand(*metadata_shape).reshape(
            batch, variates, components, candidate_count
        )

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
            'source_scales': torch.gather(
                source_scale_candidates, dim=3, index=top_indices
            ),
            'source_lags': torch.gather(
                source_lag_candidates, dim=3, index=top_indices
            ),
            'source_effective_lags': torch.gather(
                source_effective_lag_candidates, dim=3, index=top_indices
            ),
        }


class CrossMessageRefiner(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        # A missing external message must map to exactly zero so the correction
        # path cannot create an unconditional forecast on its own.
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        return self.norm(message + self.ffn(message))


class Model(nn.Module):
    """TimeMixer self forecast plus sparse multi-scale cross correction."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.output_attention = bool(configs.output_attention)
        self.use_norm = bool(configs.use_norm)
        self.hidden = int(getattr(
            configs, 'decomp_hidden', 16
        ))
        lags = _parse_int_list(
            getattr(configs, 'decomp_lags', None), (0, 1, 2, 4, 8)
        )
        self.lags = lags
        down_sampling_layers = int(getattr(
            configs, 'decomp_down_sampling_layers', 3
        ))
        down_sampling_window = int(getattr(
            configs, 'decomp_down_sampling_window', 2
        ))
        down_sampling_method = getattr(
            configs, 'decomp_down_sampling_method', 'avg'
        )

        self.self_backbone = ChannelIndependentTimeMixer(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            hidden=self.hidden,
            d_ff=int(getattr(configs, 'decomp_d_ff', 32)),
            mixing_layers=int(getattr(
                configs, 'decomp_mixing_layers', configs.e_layers
            )),
            down_sampling_layers=down_sampling_layers,
            down_sampling_window=down_sampling_window,
            down_sampling_method=down_sampling_method,
            num_variates=int(configs.enc_in),
            use_norm=self.use_norm,
            moving_avg=int(getattr(configs, 'decomp_moving_avg', 25)),
            lags=lags,
            freq=str(getattr(configs, 'freq', 'h')),
            dropout=float(configs.dropout),
        )
        self.router = MultiScaleComponentLagRouter(
            hidden=self.hidden,
            lags=lags,
            num_scales=self.self_backbone.num_scales,
            down_sampling_window=down_sampling_window,
            top_k=int(getattr(configs, 'decomp_top_k', 3)),
            temperature=float(getattr(
                configs, 'decomp_router_temperature', 1.0
            )),
            dropout=float(configs.dropout),
            variate_top_k=int(getattr(
                configs, 'decomp_variate_top_k', 8
            )),
        )
        self.message_refiner = CrossMessageRefiner(
            self.hidden, float(configs.dropout)
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
            'self': float(getattr(configs, 'decomp_self_loss', 0.1)),
            'utility': float(getattr(configs, 'decomp_utility_loss', 0.05)),
            'safe': float(getattr(configs, 'decomp_safe_loss', 0.05)),
            'entropy': float(getattr(configs, 'decomp_entropy_loss', 1e-3)),
        }
        self._aux_state: Optional[Dict[str, torch.Tensor]] = None

    def _component_corrections(
            self, self_tokens: torch.Tensor, messages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deltas = []
        gates = []
        contributions = []
        self_context = self_tokens.flatten(start_dim=2)
        for component in range(2):
            message = messages[:, :, component, :]
            delta = self.delta_heads[component](message)
            gate = torch.sigmoid(self.gate_heads[component](torch.cat([
                self_context, message
            ], dim=-1)))
            deltas.append(delta)
            gates.append(gate)
            contributions.append(gate * delta)
        return (
            torch.stack(deltas, dim=2),
            torch.stack(gates, dim=2),
            torch.stack(contributions, dim=2),
        )

    def forecast(
            self, x_enc: torch.Tensor, x_mark_enc=None,
            x_dec=None, x_mark_dec=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x_enc.shape[1] != self.seq_len:
            raise ValueError(
                f'Expected seq_len={self.seq_len}, got {x_enc.shape[1]}'
            )
        # The entire TimeMixer self path is channel independent.
        features = self.self_backbone(x_enc.transpose(1, 2), x_mark_enc)
        self_prediction = features['prediction']
        self_tokens = features['query_tokens']
        source_tokens = features['source_tokens']

        if x_enc.shape[2] > 1:
            routed = self.router(self_tokens, source_tokens)
            weighted_message = (
                routed['weights'].unsqueeze(-1) * routed['values']
            ).sum(dim=-2)
            messages = self.message_refiner(weighted_message)
            deltas, gates, contributions = self._component_corrections(
                self_tokens, messages
            )
            cross_correction = contributions.sum(dim=2)
        else:
            routed = self.router(self_tokens, source_tokens)
            messages = torch.zeros_like(self_tokens)
            deltas = self_prediction.new_zeros(
                *self_prediction.shape[:2], 2, self.pred_len
            )
            gates = torch.zeros_like(deltas)
            contributions = torch.zeros_like(deltas)
            cross_correction = torch.zeros_like(self_prediction)

        prediction = self_prediction + cross_correction
        final_output = self.self_backbone.denormalize(
            prediction
        ).transpose(1, 2)
        self_output = self.self_backbone.denormalize(
            self_prediction
        ).transpose(1, 2)
        cross_correction_output = self.self_backbone.rescale_residual(
            cross_correction
        ).transpose(1, 2)

        if routed['weights'].numel():
            entropy_loss = -(
                routed['weights'] * torch.log(routed['weights'] + 1e-8)
            ).sum(dim=-1).mean()
        else:
            entropy_loss = self_prediction.new_zeros(())

        self._aux_state = {
            'final': final_output,
            'self': self_output,
            'cross_correction': cross_correction_output,
            'self_normalized': self_prediction,
            'self_tokens': self_tokens,
            'full_contributions': contributions,
            'routed_values': routed['values'],
            'routed_weights': routed['weights'],
            'entropy': entropy_loss,
        } if self.training else None

        diagnostics = {
            'trend': features['input_trend'],
            'seasonal': features['input_seasonal'],
            'fluctuation': features['input_seasonal'],
            'normalized_scale_predictions': features['scale_predictions'],
            'target_scale_weights': features['target_scale_weights'],
            'multi_scale_component_tokens': features['component_tokens'],
            'self_prediction': self_output,
            'selected_sources': routed['source_variates'],
            'selected_components': routed['source_components'],
            'selected_scales': routed['source_scales'],
            'selected_scale_lags': routed['source_lags'],
            'selected_effective_lags': routed['source_effective_lags'],
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
        """Train the self baseline and cross correction as separate roles."""
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
        cross_correction = state['cross_correction'][..., channel_slice]
        target = target[..., -target_channels:]

        self_loss = F.mse_loss(self_output, target)
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
            leave_one_out_messages = torch.stack(
                leave_one_out_messages, dim=3
            )

            self_context = state['self_tokens'].flatten(start_dim=2)
            self_context = self_context.unsqueeze(2).expand(
                -1, -1, selected_k, -1
            )
            loo_contributions = []
            for component in range(2):
                message = leave_one_out_messages[:, :, component, :, :]
                delta = self.delta_heads[component](message)
                gate = torch.sigmoid(self.gate_heads[component](
                    torch.cat([self_context, message], dim=-1)
                ))
                loo_contributions.append(gate * delta)
            loo_contributions = torch.stack(
                loo_contributions, dim=2
            )

            full_contributions = state['full_contributions']
            total_full = full_contributions.sum(dim=2)
            loo_prediction = (
                state['self_normalized'].unsqueeze(2).unsqueeze(3)
                + total_full.unsqueeze(2).unsqueeze(3)
                - full_contributions.unsqueeze(3)
                + loo_contributions
            )
            loo_prediction = self.self_backbone.denormalize(loo_prediction)
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
