"""iTransformer with prediction-feedback fixed-point output refinement.

The backbone is exactly one iTransformer.  After producing a direct
multi-horizon forecast, prefixes of that forecast are appended to the observed
history and passed through the *same* backbone again.  A temporally coherent
forecast should be a fixed point of these re-forecast operations.  At inference
time we therefore optimize only the forecast tensor (never model parameters) to
reduce the disagreement between the direct and rolled forecasts.

All fixed-point calculations happen in the original data scale.  The update is
preconditioned and bounded with the input standard deviation so that variables
with different units can share one set of refinement hyperparameters.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


def _inverse_softplus(value):
    """Return x such that softplus(x) is approximately ``value``."""
    value = float(value)
    if value <= 0:
        raise ValueError('refuture_step_size must be positive')
    return math.log(math.expm1(value))


class Model(nn.Module):
    """Single-backbone iTransformer with output-space fixed-point updates."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.output_attention = bool(configs.output_attention)
        self.use_norm = bool(configs.use_norm)
        self.class_strategy = configs.class_strategy

        # The forecasting backbone intentionally matches model/iTransformer.py.
        self.enc_embedding = DataEmbedding_inverted(
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
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(
            configs.d_model, self.pred_len, bias=True
        )

        # Output-space optimization configuration.
        self.refinement_steps = int(
            getattr(configs, 'refuture_steps', 2)
        )
        if self.refinement_steps < 0:
            raise ValueError('refuture_steps must be non-negative')

        self.feedback_splits = self._parse_feedback_splits(
            getattr(configs, 'refuture_splits', '0.25,0.5,0.75')
        )
        self.anchor_weight = float(
            getattr(configs, 'refuture_anchor_weight', 0.1)
        )
        self.max_update_ratio = float(
            getattr(configs, 'refuture_max_update', 0.5)
        )
        self.differentiable_refinement = bool(int(getattr(
            configs, 'refuture_differentiable', 0
        )))
        self.learnable_step = bool(int(getattr(
            configs, 'refuture_learnable_step', 1
        )))
        if self.anchor_weight < 0:
            raise ValueError('refuture_anchor_weight must be non-negative')
        if self.max_update_ratio <= 0:
            raise ValueError('refuture_max_update must be positive')

        initial_step = float(getattr(
            configs, 'refuture_step_size', 0.2
        ))
        step_values = torch.full(
            (max(1, self.refinement_steps),), initial_step
        )
        if self.learnable_step:
            self.step_logits = nn.Parameter(torch.full_like(
                step_values, _inverse_softplus(initial_step)
            ))
            self.register_buffer('fixed_step_sizes', None)
        else:
            self.register_parameter('step_logits', None)
            self.register_buffer('fixed_step_sizes', step_values)

        # Optional model-owned losses.  The experiment runner already applies
        # the final forecast MSE, so auxiliary_loss returns only these terms.
        self.base_loss_weight = float(getattr(
            configs, 'refuture_base_loss_weight', 0.2
        ))
        self.consistency_loss_weight = float(getattr(
            configs, 'refuture_consistency_loss_weight', 0.05
        ))
        self.safe_loss_weight = float(getattr(
            configs, 'refuture_safe_loss_weight', 0.1
        ))
        self.update_loss_weight = float(getattr(
            configs, 'refuture_update_loss_weight', 0.01
        ))
        auxiliary_weights = {
            'refuture_base_loss_weight': self.base_loss_weight,
            'refuture_consistency_loss_weight': (
                self.consistency_loss_weight
            ),
            'refuture_safe_loss_weight': self.safe_loss_weight,
            'refuture_update_loss_weight': self.update_loss_weight,
        }
        for name, weight in auxiliary_weights.items():
            if weight < 0:
                raise ValueError(f'{name} must be non-negative')

        self._aux_state = None

    def _parse_feedback_splits(self, specification):
        """Parse fractions or absolute horizons into sorted split indices."""
        if self.pred_len <= 1:
            return tuple()
        if isinstance(specification, str):
            raw_values = [
                item.strip() for item in specification.split(',')
                if item.strip()
            ]
        elif isinstance(specification, (tuple, list)):
            raw_values = list(specification)
        else:
            raw_values = [specification]

        splits = set()
        for raw_value in raw_values:
            value = float(raw_value)
            if 0 < value < 1:
                split = int(round(value * self.pred_len))
            elif value >= 1 and value.is_integer():
                split = int(value)
            else:
                raise ValueError(
                    'refuture_splits entries must be fractions in (0, 1) '
                    'or positive integer horizons'
                )
            if not 1 <= split < self.pred_len:
                raise ValueError(
                    f'Refuture split {split} must be in '
                    f'[1, {self.pred_len - 1}]'
                )
            splits.add(split)
        if self.refinement_steps > 0 and not splits:
            raise ValueError(
                'refuture_splits must contain at least one valid split when '
                'refuture_steps is positive'
            )
        return tuple(sorted(splits))

    def _forecast_once(self, x_enc, x_mark_enc):
        """Run the shared iTransformer once in the input's physical scale."""
        if x_enc.ndim != 3:
            raise ValueError(
                'x_enc must have shape [batch, seq_len, variables]'
            )
        if x_enc.shape[1] != self.seq_len:
            raise ValueError(
                f'Expected seq_len={self.seq_len}, got {x_enc.shape[1]}'
            )

        n_vars = x_enc.shape[-1]
        if self.use_norm:
            # This deliberately follows the official iTransformer behavior:
            # the location is detached, while scale remains differentiable.
            means = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(
                torch.var(
                    centered, dim=1, keepdim=True, unbiased=False
                ) + 1e-5
            )
            model_input = centered / stdev
        else:
            means = None
            stdev = None
            model_input = x_enc

        tokens = self.enc_embedding(model_input, x_mark_enc)
        encoded, attentions = self.encoder(tokens, attn_mask=None)
        prediction = self.projector(encoded).permute(0, 2, 1)
        prediction = prediction[:, :, :n_vars]

        if self.use_norm:
            prediction = prediction * stdev[:, 0, :].unsqueeze(1)
            prediction = prediction + means[:, 0, :].unsqueeze(1)
        return prediction, encoded[:, :n_vars, :], attentions

    def _future_marks(self, x_mark_dec):
        if x_mark_dec is None:
            return None
        if x_mark_dec.shape[1] < self.pred_len:
            return None
        return x_mark_dec[:, -self.pred_len:, :]

    def _feedback_window(self, history, candidate, split,
                         history_marks, future_marks):
        """Append a forecast prefix and retain one lookback-length window."""
        feedback = torch.cat(
            [history, candidate[:, :split, :]], dim=1
        )[:, -self.seq_len:, :]

        if history_marks is None:
            return feedback, None
        if future_marks is None:
            # This fallback preserves the embedding contract for callers that
            # supply encoder marks but omit decoder/future marks.
            prefix_marks = history_marks[:, -1:, :].expand(
                -1, split, -1
            )
        else:
            prefix_marks = future_marks[:, :split, :]
        feedback_marks = torch.cat(
            [history_marks, prefix_marks], dim=1
        )[:, -self.seq_len:, :]
        return feedback, feedback_marks

    def _consistency_energy(self, history, history_marks, future_marks,
                            candidate, base_anchor, scale):
        """Measure departure from rolling re-forecast fixed points."""
        consistency_energy = candidate.new_zeros(())
        compared_points = 0
        split_scores = []

        for split in self.feedback_splits:
            feedback, feedback_marks = self._feedback_window(
                history,
                candidate,
                split,
                history_marks,
                future_marks,
            )
            rolled, _, _ = self._forecast_once(
                feedback, feedback_marks
            )
            remaining = self.pred_len - split
            residual = (
                rolled[:, :remaining, :]
                - candidate[:, split:, :]
            ) / scale
            # Accumulate the energy in FP32 under AMP; gradients still flow
            # through the cast to the forecast and shared backbone.
            split_energy = 0.5 * residual.float().square().sum()
            consistency_energy = consistency_energy + split_energy
            compared_points += residual.numel()
            split_scores.append(residual.square().mean().detach())

        consistency_energy = (
            consistency_energy / len(self.feedback_splits)
        )
        anchor_residual = (candidate - base_anchor) / scale
        anchor_energy = 0.5 * self.anchor_weight * (
            anchor_residual.float().square().sum()
        )
        total_energy = consistency_energy + anchor_energy

        # A scale-free scalar suitable for logging and a low-weight auxiliary
        # objective.  The optimization energy itself intentionally uses a sum
        # so its gradient does not depend on batch/horizon averaging factors.
        mean_consistency = (
            consistency_energy
            * len(self.feedback_splits)
            / max(1, compared_points)
        )
        return total_energy, mean_consistency, split_scores

    def _step_size(self, index):
        if self.step_logits is not None:
            return F.softplus(self.step_logits[index])
        return self.fixed_step_sizes[index]

    def _fixed_point_refine(self, history, history_marks, future_marks,
                            base_prediction, input_scale):
        """Optimize only the forecast tensor using fixed-point consistency."""
        if self.refinement_steps == 0 or not self.feedback_splits:
            diagnostics = {
                'feedback_splits': self.feedback_splits,
                'energy_trace': [],
                'update_rms_trace': [],
                'step_sizes': [],
                'iterations': 0,
            }
            return base_prediction, None, diagnostics

        outer_grad_enabled = torch.is_grad_enabled()
        differentiable = (
            self.training
            and outer_grad_enabled
            and self.differentiable_refinement
        )
        energy_trace = []
        update_rms_trace = []
        split_score_trace = []
        step_size_trace = []
        last_mean_consistency = None

        # Validation and testing wrap the whole model in no_grad().  Output
        # optimization still needs dE/dY, so locally enable autograd and make
        # the forecast a leaf.  No optimizer step or parameter mutation occurs.
        with torch.enable_grad():
            if outer_grad_enabled and base_prediction.requires_grad:
                candidate = base_prediction
            else:
                candidate = base_prediction.detach().requires_grad_(True)
            base_anchor = base_prediction.detach()

            for step_index in range(self.refinement_steps):
                energy, mean_consistency, split_scores = (
                    self._consistency_energy(
                        history,
                        history_marks,
                        future_marks,
                        candidate,
                        base_anchor,
                        input_scale,
                    )
                )
                gradient = torch.autograd.grad(
                    energy,
                    candidate,
                    create_graph=differentiable,
                    retain_graph=outer_grad_enabled or differentiable,
                    only_inputs=True,
                )[0]

                # First-order mode avoids second-order backbone derivatives but
                # still lets the outer loss learn the positive step schedule.
                direction = gradient if differentiable else gradient.detach()
                preconditioned = direction * input_scale.square()
                step_size = self._step_size(step_index)
                raw_update = step_size * preconditioned
                update_limit = self.max_update_ratio * input_scale
                bounded_update = update_limit * torch.tanh(
                    raw_update / (update_limit + 1e-6)
                )
                candidate = candidate - bounded_update

                energy_trace.append(mean_consistency.detach())
                update_rms_trace.append(
                    torch.sqrt(bounded_update.detach().square().mean())
                )
                split_score_trace.append(split_scores)
                step_size_trace.append(step_size.detach())
                last_mean_consistency = mean_consistency

            refined = candidate if outer_grad_enabled else candidate.detach()

        diagnostics = {
            'feedback_splits': self.feedback_splits,
            'energy_trace': energy_trace,
            'split_score_trace': split_score_trace,
            'update_rms_trace': update_rms_trace,
            'step_sizes': step_size_trace,
            'iterations': self.refinement_steps,
            'differentiable_refinement': differentiable,
        }
        return refined, last_mean_consistency, diagnostics

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        base_prediction, base_tokens, base_attentions = self._forecast_once(
            x_enc, x_mark_enc
        )
        input_scale = torch.sqrt(torch.var(
            x_enc, dim=1, keepdim=True, unbiased=False
        ) + 1e-5).detach()
        future_marks = self._future_marks(x_mark_dec)
        prediction, consistency, refinement_diagnostics = (
            self._fixed_point_refine(
                x_enc,
                x_mark_enc,
                future_marks,
                base_prediction,
                input_scale,
            )
        )

        if self.training:
            self._aux_state = {
                'base': base_prediction,
                'final': prediction,
                'update': prediction - base_prediction,
                'scale': input_scale,
                'consistency': consistency,
            }
        else:
            self._aux_state = None

        diagnostics = {
            'base_attention': base_attentions,
            'base_tokens': base_tokens.detach(),
            'fixed_point': refinement_diagnostics,
            'base_prediction': base_prediction.detach(),
            'total_update': (prediction - base_prediction).detach(),
        }
        return prediction, diagnostics

    def _aligned_auxiliary_state(self, target):
        if self._aux_state is None:
            raise RuntimeError(
                'A training forward pass is required before auxiliary_loss'
            )
        target = target[:, -self.pred_len:, :]
        target_channels = target.shape[-1]
        prediction_channels = self._aux_state['final'].shape[-1]
        if target_channels > prediction_channels:
            raise ValueError('Target has more channels than model prediction')

        # This matches the experiment runner's MS convention: the target is
        # the final channel of the multivariate model output.
        channel_slice = slice(
            prediction_channels - target_channels, prediction_channels
        )
        state = {
            name: value[..., channel_slice]
            for name, value in self._aux_state.items()
            if name in ('base', 'final', 'update', 'scale')
        }
        state['consistency'] = self._aux_state['consistency']
        return state, target

    def auxiliary_loss(self, target):
        """Return base, consistency, safety, and update regularizers."""
        if self._aux_state is None:
            return {'total': target.new_zeros(())}

        state, target = self._aligned_auxiliary_state(target)
        base_loss = F.mse_loss(state['base'], target)
        base_error = torch.mean(
            (state['base'] - target).square(), dim=(1, 2)
        ).detach()
        final_error = torch.mean(
            (state['final'] - target).square(), dim=(1, 2)
        )
        safe_loss = F.relu(final_error - base_error).mean()
        update_loss = torch.mean(torch.abs(
            state['update'] / (state['scale'] + 1e-6)
        ))
        consistency_loss = state['consistency']
        if consistency_loss is None:
            consistency_loss = target.new_zeros(())

        total = (
            self.base_loss_weight * base_loss
            + self.consistency_loss_weight * consistency_loss
            + self.safe_loss_weight * safe_loss
            + self.update_loss_weight * update_loss
        )
        return {
            'total': total,
            'base_loss': base_loss,
            'consistency_loss': consistency_loss,
            'safe_loss': safe_loss,
            'update_loss': update_loss,
        }

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        prediction, diagnostics = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        prediction = prediction[:, -self.pred_len:, :]
        if self.output_attention:
            return prediction, diagnostics
        return prediction
