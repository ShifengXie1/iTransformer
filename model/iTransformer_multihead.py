"""Horizon-conditioned explicit multi-head iTransformer.

Each variate is still represented by one token, as in iTransformer. A
standard iTransformer encoder first builds contextual variate tokens. An
additional explicit relation-attention layer then learns one variate-to-
variate graph per head. Forecast-horizon gates decide how much every future
step uses each graph, so the cross-variate relation used for a near forecast
does not have to be the same as the one used for a distant forecast.

This module deliberately does not decompose the input into trend/frequency
components and does not pre-assign variables to groups.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class HorizonConditionedRelationAttention(nn.Module):
    """Explicit variate-relation heads selected by forecast horizon.

    The relation heads produce H different [N, N] attention maps. A learned
    horizon embedding and one prototype per head produce a [P, H] routing
    matrix. A weak Gaussian prior anchors the otherwise permutation-
    exchangeable heads to ordered forecast ranges while still allowing the
    learned routing logits to move the boundaries.
    """

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        num_heads: int,
        head_dim: int,
        gate_dim: int,
        dropout: float,
        temperature: float,
        prior_strength: float,
        exclude_self: bool,
    ) -> None:
        super().__init__()
        if pred_len < 1:
            raise ValueError("pred_len must be at least 1.")
        if num_heads < 1:
            raise ValueError("mh_relation_heads must be at least 1.")
        if head_dim < 0:
            raise ValueError("mh_head_dim must be non-negative.")
        if gate_dim < 1:
            raise ValueError("mh_gate_dim must be at least 1.")
        if temperature <= 0.0:
            raise ValueError("mh_horizon_temperature must be positive.")
        if prior_strength < 0.0:
            raise ValueError("mh_horizon_prior_strength cannot be negative.")

        self.pred_len = pred_len
        self.num_heads = num_heads
        self.head_dim = head_dim or math.ceil(d_model / num_heads)
        self.temperature = temperature
        self.prior_strength = prior_strength
        self.exclude_self = exclude_self

        self.input_norm = nn.LayerNorm(d_model)
        projection_dim = num_heads * self.head_dim
        self.query_projection = nn.Linear(d_model, projection_dim)
        self.key_projection = nn.Linear(d_model, projection_dim)
        self.value_projection = nn.Linear(d_model, projection_dim)
        self.attention_dropout = nn.Dropout(dropout)

        # e_tau and r_h from the horizon-conditioned routing formulation.
        self.horizon_embeddings = nn.Parameter(
            torch.empty(pred_len, gate_dim)
        )
        self.head_prototypes = nn.Parameter(
            torch.empty(num_heads, gate_dim)
        )
        self.horizon_to_readout = nn.Linear(
            gate_dim, num_heads * self.head_dim
        )

        # The prior gives every head an identifiable initial horizon range.
        horizon_positions = torch.linspace(0.0, 1.0, pred_len).unsqueeze(1)
        if num_heads == 1 or pred_len == 1:
            horizon_prior = torch.zeros(pred_len, num_heads)
        else:
            head_centers = torch.linspace(0.0, 1.0, num_heads).unsqueeze(0)
            bandwidth = 1.0 / (num_heads - 1)
            horizon_prior = -0.5 * (
                (horizon_positions - head_centers) / bandwidth
            ).square()
        self.register_buffer(
            "horizon_prior", horizon_prior, persistent=True
        )

        nn.init.normal_(self.horizon_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.head_prototypes, mean=0.0, std=0.02)

    def _horizon_gates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        learned_logits = torch.matmul(
            self.horizon_embeddings,
            self.head_prototypes.transpose(0, 1),
        ) / math.sqrt(self.horizon_embeddings.shape[-1])
        routing_logits = (
            learned_logits + self.prior_strength * self.horizon_prior
        ) / self.temperature
        return torch.softmax(routing_logits, dim=-1), routing_logits

    def forward(
        self,
        tokens: torch.Tensor,
        return_head_forecasts: bool = False,
    ) -> Dict[str, object]:
        if tokens.ndim != 3:
            raise ValueError(
                "tokens must have shape [B, N, D], "
                f"got {tuple(tokens.shape)}."
            )

        batch_size, num_variables, _ = tokens.shape
        normalized = self.input_norm(tokens)

        def split_heads(projection: nn.Linear) -> torch.Tensor:
            projected = projection(normalized)
            return projected.view(
                batch_size,
                num_variables,
                self.num_heads,
                self.head_dim,
            ).permute(0, 2, 1, 3)

        query = split_heads(self.query_projection)
        key = split_heads(self.key_projection)
        value = split_heads(self.value_projection)

        scores = torch.matmul(
            query, key.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)
        if self.exclude_self and num_variables > 1:
            diagonal_mask = torch.eye(
                num_variables, device=scores.device, dtype=torch.bool
            ).view(1, 1, num_variables, num_variables)
            scores = scores.masked_fill(
                diagonal_mask, torch.finfo(scores.dtype).min
            )

        # Keep the pre-dropout maps for visualization and regularization.
        attention_maps = torch.softmax(scores, dim=-1)
        contexts = torch.matmul(
            self.attention_dropout(attention_maps), value
        )

        horizon_gates, routing_logits = self._horizon_gates()
        readout = self.horizon_to_readout(
            self.horizon_embeddings
        ).view(
            self.pred_len,
            self.num_heads,
            self.head_dim,
        )

        # Contract routing and readout directly, avoiding a potentially large
        # [B, P, H, N] tensor during ordinary training.
        routed_readout = horizon_gates.unsqueeze(-1) * readout
        correction = torch.einsum(
            "bhnd,phd->bpn", contexts, routed_readout
        ) / math.sqrt(self.head_dim)

        head_forecasts: Optional[torch.Tensor] = None
        if return_head_forecasts:
            # Materialize per-head proposals only for explicit diagnostics.
            head_forecasts = torch.einsum(
                "bhnd,phd->bphn", contexts, readout
            ) / math.sqrt(self.head_dim)

        return {
            "correction": correction,
            "attention_maps": attention_maps,
            "head_contexts": contexts,
            "head_forecasts": head_forecasts,
            "horizon_gates": horizon_gates,
            "routing_logits": routing_logits,
        }


class Model(nn.Module):
    """iTransformer with horizon-conditioned explicit relation heads."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_variables = configs.enc_in
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.class_strategy = configs.class_strategy

        relation_heads = getattr(configs, "mh_relation_heads", 0)
        if relation_heads == 0:
            relation_heads = configs.n_heads
        self.num_relation_heads = relation_heads
        self.diversity_loss_weight = getattr(
            configs, "mh_diversity_loss_weight", 0.01
        )
        self.balance_loss_weight = getattr(
            configs, "mh_balance_loss_weight", 0.001
        )
        if self.diversity_loss_weight < 0.0:
            raise ValueError(
                "mh_diversity_loss_weight cannot be negative."
            )
        if self.balance_loss_weight < 0.0:
            raise ValueError("mh_balance_loss_weight cannot be negative.")

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

        # The direct iTransformer forecast is a stable base. Explicit heads
        # learn a horizon-conditioned cross-variate correction on top of it.
        self.projector = nn.Linear(
            configs.d_model, configs.pred_len, bias=True
        )
        self.relation_attention = HorizonConditionedRelationAttention(
            d_model=configs.d_model,
            pred_len=configs.pred_len,
            num_heads=relation_heads,
            head_dim=getattr(configs, "mh_head_dim", 0),
            gate_dim=getattr(configs, "mh_gate_dim", 32),
            dropout=configs.dropout,
            temperature=getattr(
                configs, "mh_horizon_temperature", 1.0
            ),
            prior_strength=getattr(
                configs, "mh_horizon_prior_strength", 1.0
            ),
            exclude_self=bool(getattr(configs, "mh_exclude_self", 1)),
        )

        residual_init = getattr(configs, "mh_residual_init", 0.1)
        if not -1.0 < residual_init < 1.0:
            raise ValueError("mh_residual_init must lie in (-1, 1).")
        self.residual_logit = nn.Parameter(
            torch.tensor(
                math.atanh(residual_init), dtype=torch.float32
            )
        )
        self._last_auxiliary: Optional[Dict[str, torch.Tensor]] = None

    def _head_regularization(
        self,
        attention_maps: torch.Tensor,
        horizon_gates: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.num_relation_heads == 1:
            diversity = attention_maps.new_zeros(())
        else:
            flattened = attention_maps.flatten(start_dim=2)
            flattened = F.normalize(flattened, p=2, dim=-1)
            similarity = torch.matmul(
                flattened, flattened.transpose(1, 2)
            )
            off_diagonal = ~torch.eye(
                self.num_relation_heads,
                device=similarity.device,
                dtype=torch.bool,
            ).unsqueeze(0)
            diversity = similarity.masked_select(off_diagonal).mean()

        average_usage = horizon_gates.mean(dim=0)
        uniform_usage = torch.full_like(
            average_usage, 1.0 / self.num_relation_heads
        )
        balance = self.num_relation_heads * F.mse_loss(
            average_usage, uniform_usage
        )
        total = (
            self.diversity_loss_weight * diversity
            + self.balance_loss_weight * balance
        )
        return {
            "total": total,
            "attention_diversity": diversity,
            "horizon_gate_balance": balance,
        }

    def auxiliary_loss(
        self, target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Return regularizers consumed by the shared training loop."""
        del target
        if self._last_auxiliary is not None:
            return self._last_auxiliary
        zero = self.residual_logit.new_zeros(())
        return {
            "total": zero,
            "attention_diversity": zero,
            "horizon_gate_balance": zero,
        }

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        return_head_forecasts: bool = False,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        if x_enc.ndim != 3:
            raise ValueError(
                f"x_enc must have shape [B, L, N], got {tuple(x_enc.shape)}."
            )
        if x_enc.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, got {x_enc.shape[1]}."
            )
        if x_enc.shape[2] != self.num_variables:
            raise ValueError(
                f"Expected enc_in={self.num_variables}, "
                f"got {x_enc.shape[2]}."
            )
        if x_mark_enc is not None and x_mark_enc.shape[:2] != x_enc.shape[:2]:
            raise ValueError(
                "x_mark_enc must share batch and temporal dimensions with x_enc."
            )

        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            normalized = x_enc - means
            stdev = torch.sqrt(
                torch.var(
                    normalized,
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                )
                + 1e-5
            )
            normalized = normalized / stdev
        else:
            means = None
            stdev = None
            normalized = x_enc

        embedded = self.enc_embedding(normalized, x_mark_enc)
        encoded, encoder_attentions = self.encoder(
            embedded, attn_mask=None
        )
        # Time-feature tokens, when present, participate in the base encoder,
        # but only real variable tokens enter the explicit relation heads.
        variable_tokens = encoded[:, : self.num_variables]

        base_prediction = self.projector(variable_tokens).permute(0, 2, 1)
        relation_outputs = self.relation_attention(
            variable_tokens,
            return_head_forecasts=return_head_forecasts,
        )
        residual_scale = torch.tanh(self.residual_logit)
        scaled_correction = (
            residual_scale * relation_outputs["correction"]
        )
        normalized_prediction = (
            base_prediction + scaled_correction
        )

        self._last_auxiliary = self._head_regularization(
            relation_outputs["attention_maps"],
            relation_outputs["horizon_gates"],
        )

        prediction = normalized_prediction
        base_prediction_output = base_prediction
        correction_output = scaled_correction
        if self.use_norm:
            prediction = prediction * stdev[:, 0, :].unsqueeze(1)
            prediction = prediction + means[:, 0, :].unsqueeze(1)
            base_prediction_output = (
                base_prediction * stdev[:, 0, :].unsqueeze(1)
                + means[:, 0, :].unsqueeze(1)
            )
            correction_output = (
                scaled_correction * stdev[:, 0, :].unsqueeze(1)
            )

        outputs: Dict[str, object] = {
            "prediction": prediction,
            "base_prediction": base_prediction_output,
            "relation_correction": correction_output,
            "variable_tokens": variable_tokens,
            "relation_attention_maps": relation_outputs["attention_maps"],
            "head_contexts": relation_outputs["head_contexts"],
            "horizon_gates": relation_outputs["horizon_gates"],
            "routing_logits": relation_outputs["routing_logits"],
            "horizon_prior": self.relation_attention.horizon_prior,
            "residual_scale": residual_scale,
        }
        if relation_outputs["head_forecasts"] is not None:
            outputs["head_forecasts"] = relation_outputs["head_forecasts"]
        attentions: Dict[str, object] = {
            "encoder": encoder_attentions,
            "relation_heads": relation_outputs["attention_maps"],
            "horizon_gates": relation_outputs["horizon_gates"],
        }
        return outputs, attentions

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: Optional[torch.Tensor],
        x_mark_dec: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        return_auxiliary: bool = False,
    ):
        del x_dec, x_mark_dec, mask
        outputs, attentions = self.forecast(
            x_enc,
            x_mark_enc,
            return_head_forecasts=return_auxiliary,
        )
        prediction = outputs["prediction"][:, -self.pred_len :, :]
        if return_auxiliary:
            result = dict(outputs)
            result["prediction"] = prediction
            result["auxiliary_loss"] = self._last_auxiliary
            if self.output_attention:
                result["attentions"] = attentions
            return result
        if self.output_attention:
            return prediction, attentions
        return prediction
