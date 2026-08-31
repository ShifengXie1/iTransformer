import math

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


class PredictionAwareDynamicFusion(nn.Module):
    """Fuse experts using both latent states and their forecast disagreement."""

    def __init__(self, d_model, pred_len, hidden_size, dropout):
        super().__init__()
        self.pred_len = int(pred_len)
        self.state_gate = nn.Sequential(
            nn.Linear(2 * d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2 * self.pred_len),
        )
        prediction_hidden = max(8, min(int(hidden_size), 64))
        self.prediction_gate = nn.Sequential(
            nn.Linear(3, prediction_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prediction_hidden, 2),
        )
        # Both forecasters contribute equally before the gate has learned a
        # preference. Both state- and prediction-conditioned logits start at 0.
        nn.init.zeros_(self.state_gate[-1].weight)
        nn.init.zeros_(self.state_gate[-1].bias)
        nn.init.zeros_(self.prediction_gate[-1].weight)
        nn.init.zeros_(self.prediction_gate[-1].bias)

    def forward(self, patch_prediction, joint_prediction,
                patch_state, joint_state):
        batch_size, n_vars, _ = patch_state.shape
        state_logits = self.state_gate(
            torch.cat([patch_state, joint_state], dim=-1)
        ).reshape(batch_size, n_vars, self.pred_len, 2)
        state_logits = state_logits.permute(0, 2, 1, 3)

        disagreement = torch.abs(patch_prediction - joint_prediction)
        prediction_features = torch.stack(
            [patch_prediction, joint_prediction, disagreement], dim=-1
        )
        prediction_logits = self.prediction_gate(prediction_features)
        weights = torch.softmax(
            state_logits + prediction_logits, dim=-1
        )
        candidates = torch.stack(
            [patch_prediction, joint_prediction], dim=-1
        )
        base_prediction = (weights * candidates).sum(dim=-1)
        return base_prediction, weights, disagreement


class SparseVariateRouter(nn.Module):
    """Select the most relevant source variables for every sample and target."""

    def __init__(self, d_model, top_k, temperature, dropout):
        super().__init__()
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        if self.top_k < 1:
            raise ValueError('three_refiner_top_k must be at least 1')
        if self.temperature <= 0:
            raise ValueError('three_router_temperature must be positive')

        self.norm = nn.LayerNorm(d_model)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        # tokens: [B, C, D]
        batch_size, n_vars, d_model = tokens.shape
        if n_vars < 2:
            empty_indices = torch.empty(
                batch_size, n_vars, 0, dtype=torch.long,
                device=tokens.device,
            )
            empty_values = tokens.new_empty(batch_size, n_vars, 0)
            empty_scores = tokens.new_zeros(batch_size, n_vars, n_vars)
            return empty_indices, empty_values, empty_scores

        router_tokens = self.dropout(self.norm(tokens))
        queries = self.query_projection(router_tokens)
        keys = self.key_projection(router_tokens)
        scores = torch.matmul(queries, keys.transpose(-1, -2))
        scores = scores / math.sqrt(d_model)
        diagonal = torch.eye(
            n_vars, dtype=torch.bool, device=tokens.device
        ).unsqueeze(0)
        scores = scores.masked_fill(diagonal, -torch.inf)

        selected_k = min(self.top_k, n_vars - 1)
        selected_scores, selected_sources = torch.topk(
            scores, k=selected_k, dim=-1
        )
        selected_weights = torch.softmax(
            selected_scores / self.temperature, dim=-1
        )
        return selected_sources, selected_weights, scores


class PredictionAwareRefinementStep(nn.Module):
    """One shared, history-anchored and dynamically gated refinement step."""

    def __init__(self, n_vars, pred_len, d_model, n_heads, d_ff,
                 dropout, activation, output_attention, top_k,
                 router_temperature, step_init, cross_gate_init):
        super().__init__()
        self.n_vars = int(n_vars)
        self.pred_len = int(pred_len)
        self.n_heads = int(n_heads)
        self.output_attention = bool(output_attention)
        self.trajectory_embedding = nn.Linear(4 * pred_len, d_model)
        self.state_projection = nn.Linear(2 * d_model, d_model)
        self.state_norm = nn.LayerNorm(d_model)
        self.variate_embedding = nn.Embedding(self.n_vars, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.self_norm = nn.LayerNorm(d_model)
        self.self_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.router = SparseVariateRouter(
            d_model, top_k, router_temperature, dropout
        )
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_ffn_norm = nn.LayerNorm(d_model)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.self_projector = nn.Linear(d_model, pred_len, bias=True)
        self.cross_projector = nn.Linear(d_model, pred_len, bias=True)
        gate_input_size = 3 * d_model
        self.cross_gate = nn.Linear(gate_input_size, pred_len, bias=True)
        self.step_gate = nn.Linear(gate_input_size, pred_len, bias=True)

        if not 0 <= step_init <= 1:
            raise ValueError('three_gamma_init must be in [0, 1]')
        if not 0 <= cross_gate_init <= 1:
            raise ValueError('three_cross_gate_init must be in [0, 1]')
        step_probability = min(max(float(step_init), 1e-4), 1 - 1e-4)
        cross_probability = min(
            max(float(cross_gate_init), 1e-4), 1 - 1e-4
        )

        # Small residual heads and conservative gates make the initial model
        # remain close to the dynamically fused forecast.
        nn.init.xavier_uniform_(self.self_projector.weight, gain=0.05)
        nn.init.zeros_(self.self_projector.bias)
        nn.init.xavier_uniform_(self.cross_projector.weight, gain=0.05)
        nn.init.zeros_(self.cross_projector.bias)
        nn.init.zeros_(self.cross_gate.weight)
        nn.init.constant_(
            self.cross_gate.bias,
            math.log(cross_probability / (1 - cross_probability)),
        )
        nn.init.zeros_(self.step_gate.weight)
        nn.init.constant_(
            self.step_gate.bias,
            math.log(step_probability / (1 - step_probability)),
        )

    def _cross_context(self, tokens, source_indices, source_weights):
        batch_size, n_vars, d_model = tokens.shape
        selected_k = source_indices.shape[-1]
        source_bank = tokens.unsqueeze(1).expand(-1, n_vars, -1, -1)
        gather_indices = source_indices.unsqueeze(-1).expand(
            -1, -1, -1, d_model
        )
        selected_tokens = torch.gather(
            source_bank, dim=2, index=gather_indices
        )
        selected_tokens = selected_tokens * (
            source_weights.unsqueeze(-1) * selected_k
        )
        memory = selected_tokens.reshape(
            batch_size * n_vars, selected_k, d_model
        )
        queries = tokens.reshape(batch_size * n_vars, 1, d_model)
        cross_value, attention = self.attention(
            queries,
            memory,
            memory,
            need_weights=self.output_attention,
            average_attn_weights=False,
        )
        cross_value = self.cross_norm(self.dropout(cross_value))
        cross_value = self.cross_ffn_norm(
            cross_value + self.dropout(self.cross_ffn(cross_value))
        ).reshape(batch_size, n_vars, d_model)
        if attention is not None:
            attention = attention.squeeze(-2).reshape(
                batch_size, n_vars, self.n_heads, selected_k
            )
        return cross_value, attention

    def forward(self, current_prediction, patch_prediction,
                joint_prediction, patch_state, joint_state):
        # Forecasts: [B, S, C]. States: [B, C, D].
        batch_size, pred_len, n_vars = current_prediction.shape
        if pred_len != self.pred_len:
            raise ValueError(
                f'Refiner expected pred_len={self.pred_len}, got {pred_len}'
            )
        if n_vars != self.n_vars:
            raise ValueError(
                f'Expected {self.n_vars} variables, got {n_vars}'
            )

        disagreement = torch.abs(patch_prediction - joint_prediction)
        trajectory_features = torch.cat(
            [
                current_prediction.permute(0, 2, 1),
                patch_prediction.permute(0, 2, 1),
                joint_prediction.permute(0, 2, 1),
                disagreement.permute(0, 2, 1),
            ],
            dim=-1,
        )
        state_anchor = self.state_norm(self.state_projection(
            torch.cat([patch_state, joint_state], dim=-1)
        ))
        variable_ids = torch.arange(
            n_vars, device=current_prediction.device
        )
        tokens = self.trajectory_embedding(trajectory_features)
        tokens = tokens + state_anchor
        tokens = tokens + self.variate_embedding(variable_ids).unsqueeze(0)
        tokens = self.input_dropout(self.input_norm(tokens))
        self_context = self.self_norm(
            tokens + self.dropout(self.self_ffn(tokens))
        )

        source_indices, source_weights, route_scores = self.router(
            self_context
        )
        if n_vars > 1:
            cross_context, cross_attention = self._cross_context(
                self_context, source_indices, source_weights
            )
        else:
            cross_context = torch.zeros_like(self_context)
            cross_attention = None

        gate_features = torch.cat(
            [self_context, cross_context, state_anchor], dim=-1
        )
        step_gate = torch.sigmoid(
            self.step_gate(gate_features)
        ).permute(0, 2, 1)
        if n_vars > 1:
            cross_gate = torch.sigmoid(
                self.cross_gate(gate_features)
            ).permute(0, 2, 1)
        else:
            cross_gate = torch.zeros_like(step_gate)

        self_residual = self.self_projector(
            self_context
        ).permute(0, 2, 1)
        cross_residual = self.cross_projector(
            cross_context
        ).permute(0, 2, 1)
        # Forecasts are standardized before refinement. Bounding the proposal
        # prevents a recurrent step from producing an unbounded update.
        proposal = torch.tanh(
            self_residual + cross_gate * cross_residual
        )
        scaled_residual = step_gate * proposal
        updated_prediction = current_prediction + scaled_residual

        selection_mask = torch.ones(
            batch_size, n_vars, n_vars, dtype=torch.bool,
            device=current_prediction.device,
        )
        if n_vars > 1:
            selection_mask.scatter_(2, source_indices, False)

        diagnostics = None
        if self.output_attention:
            diagnostics = {
                'selected_sources': source_indices.detach(),
                'selected_weights': source_weights.detach(),
                'route_scores': route_scores.detach(),
                'selection_mask': selection_mask,
                'cross_attention': (
                    None if cross_attention is None
                    else cross_attention.detach()
                ),
                'cross_gate': cross_gate.detach(),
                'step_gate': step_gate.detach(),
                'scaled_residual': scaled_residual.detach(),
            }
        return updated_prediction, scaled_residual, diagnostics


class Model(nn.Module):
    """
    Prediction-aware three-stage forecaster:

    1. channel-independent PatchTST temporal modeling;
    2. iTransformer historical cross-variate modeling and prediction-aware
       expert fusion;
    3. shared iterative refinement with self-history anchoring, sparse routed
       cross-variate messages, and sample/variable/horizon-specific gates.
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
        self.dynamic_fusion = PredictionAwareDynamicFusion(
            configs.d_model,
            self.pred_len,
            fusion_hidden,
            configs.dropout,
        )
        self.use_refinement = bool(int(getattr(
            configs, 'three_use_refinement', 1
        )))
        self.refinement_steps = int(getattr(
            configs, 'three_refinement_steps', 2
        ))
        if self.refinement_steps < 1:
            raise ValueError('three_refinement_steps must be at least 1')
        if self.use_refinement:
            self.prediction_refiner = PredictionAwareRefinementStep(
                self.n_vars,
                self.pred_len,
                configs.d_model,
                configs.n_heads,
                configs.d_ff,
                configs.dropout,
                configs.activation,
                configs.output_attention,
                int(getattr(configs, 'three_refiner_top_k', 3)),
                float(getattr(configs, 'three_router_temperature', 1.0)),
                float(getattr(configs, 'three_gamma_init', 0.1)),
                float(getattr(configs, 'three_cross_gate_init', 0.1)),
            )
        else:
            self.prediction_refiner = None

        self.patch_loss_weight = float(getattr(
            configs, 'three_patch_loss_weight', 0.2
        ))
        self.joint_loss_weight = float(getattr(
            configs, 'three_joint_loss_weight', 0.2
        ))
        self.base_loss_weight = float(getattr(
            configs, 'three_base_loss_weight', 0.1
        ))
        self.refinement_loss_weight = float(getattr(
            configs, 'three_refinement_loss_weight', 0.1
        ))
        self.monotonic_loss_weight = float(getattr(
            configs, 'three_monotonic_loss_weight', 0.05
        ))
        loss_weights = {
            'three_patch_loss_weight': self.patch_loss_weight,
            'three_joint_loss_weight': self.joint_loss_weight,
            'three_base_loss_weight': self.base_loss_weight,
            'three_refinement_loss_weight': self.refinement_loss_weight,
            'three_monotonic_loss_weight': self.monotonic_loss_weight,
        }
        for name, weight in loss_weights.items():
            if weight < 0:
                raise ValueError(f'{name} must be non-negative')
        self._aux_state = None

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
        base_prediction, fusion_weights, prediction_disagreement = (
            self.dynamic_fusion(
                patch_prediction,
                joint_prediction,
                patch_state,
                joint_state,
            )
        )

        prediction = base_prediction
        refinement_predictions = []
        refinement_diagnostics = []
        if self.prediction_refiner is not None:
            for _ in range(self.refinement_steps):
                prediction, _, step_diagnostics = self.prediction_refiner(
                    prediction,
                    patch_prediction,
                    joint_prediction,
                    patch_state,
                    joint_state,
                )
                refinement_predictions.append(prediction)
                if step_diagnostics is not None:
                    refinement_diagnostics.append(step_diagnostics)

        total_correction = prediction - base_prediction

        if self.use_norm:
            scale = stdev[:, 0, :].unsqueeze(1)
            location = means[:, 0, :].unsqueeze(1)

            def restore(forecast):
                return forecast * scale + location

            patch_output = restore(patch_prediction)
            joint_output = restore(joint_prediction)
            base_output = restore(base_prediction)
            refinement_outputs = [
                restore(refined) for refined in refinement_predictions
            ]
            correction_output = total_correction * scale
            disagreement_output = prediction_disagreement * scale
            prediction = restore(prediction)
        else:
            patch_output = patch_prediction
            joint_output = joint_prediction
            base_output = base_prediction
            refinement_outputs = refinement_predictions
            correction_output = total_correction
            disagreement_output = prediction_disagreement

        # Keep every supervised forecast in the target's physical scale. The
        # experiment runner calls auxiliary_loss(target) after this forward.
        if self.training:
            auxiliary_state = {
                'patch': patch_output,
                'joint': joint_output,
                'base': base_output,
                'correction': correction_output,
                'final': prediction,
            }
            for index, refined in enumerate(refinement_outputs):
                auxiliary_state[f'refine_{index}'] = refined
            self._aux_state = auxiliary_state
        else:
            self._aux_state = None

        last_refinement = (
            refinement_diagnostics[-1] if refinement_diagnostics else None
        )
        diagnostics = {
            'patchtst_attention': patch_attentions,
            'itransformer_attention': joint_attentions,
            'fusion_weights': fusion_weights,
            'prediction_disagreement': disagreement_output,
            'refinement_enabled': self.use_refinement,
            'refinement_iterations': (
                self.refinement_steps if self.use_refinement else 0
            ),
            'refinement_steps': refinement_diagnostics,
            'refinement_total_correction': correction_output,
            # Compatibility aliases for callers that inspected the old stage.
            'masked_cross_attention': (
                None if last_refinement is None
                else last_refinement['cross_attention']
            ),
            'masked_cross_attention_mask': (
                None if last_refinement is None
                else last_refinement['selection_mask']
            ),
            'refinement_step_gate': (
                None if last_refinement is None
                else last_refinement['step_gate']
            ),
            'patch_len': self.patchtst.patch_len,
            'stride': self.patchtst.stride,
            'num_patches': self.patchtst.patch_num,
        }
        return prediction, diagnostics

    def _aligned_auxiliary_state(self, target):
        """Align full multivariate forecasts with M/S/MS task targets."""
        if self._aux_state is None:
            raise RuntimeError(
                'A training forward pass is required before computing loss'
            )
        if target.ndim != 3:
            raise ValueError('target must have shape [batch, pred_len, channels]')
        if target.shape[1] < self.pred_len:
            raise ValueError(
                f'Target length must be at least {self.pred_len}, '
                f'got {target.shape[1]}'
            )
        target = target[:, -self.pred_len:, :]
        target_channels = target.shape[-1]
        if target_channels > self.n_vars:
            raise ValueError('Target has more channels than model prediction')

        # The data pipeline places the target variable last for MS tasks,
        # matching the slicing convention used by the experiment runner.
        channel_slice = slice(self.n_vars - target_channels, self.n_vars)
        aligned = {
            name: prediction[..., channel_slice]
            for name, prediction in self._aux_state.items()
        }
        return aligned, target

    def compute_loss(self, target):
        """Return branch, fusion, refinement, and safety objectives."""
        state, target = self._aligned_auxiliary_state(target)
        patch_loss = F.mse_loss(state['patch'], target)
        joint_loss = F.mse_loss(state['joint'], target)
        base_loss = F.mse_loss(state['base'], target)
        final_loss = F.mse_loss(state['final'], target)
        refinement_names = sorted(
            (name for name in state if name.startswith('refine_')),
            key=lambda name: int(name.split('_')[-1]),
        )
        zero = target.new_zeros(())
        intermediate_names = refinement_names[:-1]
        if intermediate_names:
            refinement_loss = torch.stack([
                F.mse_loss(state[name], target)
                for name in intermediate_names
            ]).mean()
        else:
            refinement_loss = zero

        monotonic_terms = []
        previous_prediction = state['base']
        for name in refinement_names:
            previous_error = torch.mean(
                (previous_prediction - target) ** 2, dim=(1, 2)
            ).detach()
            current_error = torch.mean(
                (state[name] - target) ** 2, dim=(1, 2)
            )
            monotonic_terms.append(
                F.relu(current_error - previous_error).mean()
            )
            previous_prediction = state[name]
        monotonic_loss = (
            torch.stack(monotonic_terms).mean()
            if monotonic_terms else zero
        )
        auxiliary = (
            self.patch_loss_weight * patch_loss
            + self.joint_loss_weight * joint_loss
            + self.base_loss_weight * base_loss
            + self.refinement_loss_weight * refinement_loss
            + self.monotonic_loss_weight * monotonic_loss
        )
        return {
            'total': final_loss + auxiliary,
            'final_loss': final_loss,
            'patch_loss': patch_loss,
            'joint_loss': joint_loss,
            'base_loss': base_loss,
            'refinement_loss': refinement_loss,
            'monotonic_loss': monotonic_loss,
            'auxiliary_loss': auxiliary,
        }

    def auxiliary_loss(self, target):
        """Adapter for Exp_Long_Term_Forecast's auxiliary-loss hook."""
        if self._aux_state is None:
            zero = target.new_zeros(())
            return {'total': zero}
        losses = self.compute_loss(target)
        # The runner already adds the final forecast MSE. Only return the
        # weighted branch/base terms as `total` to avoid counting it twice.
        return {
            'total': losses['auxiliary_loss'],
            'total_loss': losses['total'],
            'final_loss': losses['final_loss'],
            'patch_loss': losses['patch_loss'],
            'joint_loss': losses['joint_loss'],
            'base_loss': losses['base_loss'],
            'refinement_loss': losses['refinement_loss'],
            'monotonic_loss': losses['monotonic_loss'],
        }

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        prediction, diagnostics = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        prediction = prediction[:, -self.pred_len:, :]
        if self.output_attention:
            return prediction, diagnostics
        return prediction
