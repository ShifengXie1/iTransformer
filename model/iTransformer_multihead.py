"""Residual multi-view token-head iTransformer.

The original iTransformer already uses multi-head self-attention across
variates. This variant puts the extra heads before that encoder: each variate
is decomposed into complementary temporal views, the views interact inside
the variate, and the resulting correction is added to the original inverted
token. The correction scale is initialized to zero, so the model starts from
exactly the original iTransformer.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


def _odd_window(window: int, seq_len: int) -> int:
    """Return a positive odd pooling window no larger than seq_len."""
    window = max(1, min(int(window), int(seq_len)))
    if window % 2 == 0:
        window = window - 1 if window == seq_len else window + 1
    return max(1, window)


def _automatic_scales(seq_len: int, count: int) -> Tuple[int, ...]:
    """Construct log-spaced moving-average boundaries."""
    if count <= 0:
        return ()
    largest = _odd_window(min(25, seq_len), seq_len)
    if count == 1:
        return (largest,)

    raw = torch.logspace(math.log10(3), math.log10(max(3, largest)), count)
    scales: List[int] = []
    for value in raw.tolist():
        candidate = _odd_window(round(value), seq_len)
        if not scales or candidate > scales[-1]:
            scales.append(candidate)

    for candidate in range(3, largest + 1, 2):
        if candidate not in scales:
            scales.append(candidate)
        if len(scales) == count:
            break
    scales = sorted(scales)[:count]

    if len(scales) < count:
        scales = [
            _odd_window(
                1 + round((largest - 1) * index / (count - 1)),
                seq_len,
            )
            for index in range(1, count + 1)
        ]
    return tuple(scales)


def _parse_scales(
    scales: Sequence[int] | str,
    seq_len: int,
    adaptive_heads: int,
) -> Tuple[int, ...]:
    """Parse temporal pyramid boundaries for adaptive heads."""
    if adaptive_heads <= 0:
        return ()

    boundary_count = 1 if adaptive_heads == 1 else adaptive_heads - 1
    if isinstance(scales, str):
        if scales.strip().lower() == "auto":
            parsed = _automatic_scales(seq_len, boundary_count)
        else:
            parsed = tuple(
                _odd_window(int(item.strip()), seq_len)
                for item in scales.split(",")
                if item.strip()
            )
    else:
        parsed = tuple(_odd_window(value, seq_len) for value in scales)

    if len(parsed) != boundary_count:
        raise ValueError(
            f"Expected {boundary_count} temporal scale(s) for "
            f"{adaptive_heads} adaptive view(s), but got {parsed}."
        )
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError(f"Temporal scales must be strictly increasing: {parsed}.")
    return parsed


class AdaptiveTemporalViewGenerator(nn.Module):
    """Create complementary temporal components and embed each as a token."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        adaptive_heads: int,
        scales: Sequence[int] | str,
        use_dynamic_mask: bool,
        mask_hidden: int,
        temperature: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("token_temperature must be positive.")

        self.seq_len = seq_len
        self.d_model = d_model
        self.adaptive_heads = adaptive_heads
        self.use_dynamic_mask = use_dynamic_mask
        self.temperature = temperature
        self.scales = _parse_scales(scales, seq_len, adaptive_heads)

        self.mask_generators = nn.ModuleList()
        self.projections = nn.ModuleList()
        for _ in range(adaptive_heads):
            if use_dynamic_mask:
                generator = nn.Sequential(
                    nn.Linear(seq_len, mask_hidden),
                    nn.GELU(),
                    nn.Linear(mask_hidden, seq_len),
                )
                nn.init.zeros_(generator[-1].weight)
                nn.init.zeros_(generator[-1].bias)
                self.mask_generators.append(generator)
            self.projections.append(nn.Linear(seq_len, d_model))

        self.head_embedding = nn.Parameter(
            torch.zeros(1, adaptive_heads, 1, d_model)
        )
        if adaptive_heads:
            nn.init.normal_(self.head_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _moving_average(x: torch.Tensor, window: int) -> torch.Tensor:
        padding = window // 2
        padded = F.pad(x, (padding, padding), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=window, stride=1)

    def _temporal_components(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.adaptive_heads == 0:
            return []
        if self.adaptive_heads == 1:
            return [self._moving_average(x, self.scales[0])]

        smooth = [self._moving_average(x, window) for window in self.scales]
        components = [x - smooth[0]]
        components.extend(
            smooth[index] - smooth[index + 1]
            for index in range(len(smooth) - 1)
        )
        components.append(smooth[-1])
        return components

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return adaptive tokens, masks, and additive temporal components."""
        batch_size, token_count, seq_len = x.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f"Expected temporal length {self.seq_len}, got {seq_len}."
            )
        if self.adaptive_heads == 0:
            return (
                x.new_empty(batch_size, 0, token_count, self.d_model),
                x.new_empty(batch_size, 0, token_count, seq_len),
                x.new_empty(batch_size, 0, token_count, seq_len),
            )

        components = self._temporal_components(x)
        tokens: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for head_index, component in enumerate(components):
            if self.use_dynamic_mask:
                logits = self.mask_generators[head_index](x)
                mask = torch.softmax(
                    logits / self.temperature, dim=-1
                ) * self.seq_len
            else:
                mask = torch.ones_like(component)
            token = self.projections[head_index](component * mask)
            tokens.append(token)
            masks.append(mask)

        token_tensor = torch.stack(tokens, dim=1)
        token_tensor = self.dropout(token_tensor + self.head_embedding)
        return (
            token_tensor,
            torch.stack(masks, dim=1),
            torch.stack(components, dim=1),
        )


class ResidualViewMixer(nn.Module):
    """Fuse views within each variate and form a gated residual correction."""

    def __init__(
        self,
        d_model: int,
        num_views: int,
        attention_heads: int,
        fusion_type: str,
        gate_temperature: float,
        residual_init: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if fusion_type not in {"dynamic", "learnable_global", "mean"}:
            raise ValueError(
                "fusion_type must be one of dynamic, learnable_global, or mean."
            )
        if d_model % attention_heads != 0:
            raise ValueError(
                "d_model must be divisible by view_attention_heads."
            )
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")
        if not -1.0 < residual_init < 1.0:
            raise ValueError("view_residual_init must lie in (-1, 1).")

        self.num_views = num_views
        self.fusion_type = fusion_type
        self.gate_temperature = gate_temperature
        self.view_norm = nn.LayerNorm(d_model)
        self.view_attention = nn.MultiheadAttention(
            d_model,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.global_view_logits = nn.Parameter(torch.zeros(num_views))
        self.correction = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.residual_logit = nn.Parameter(
            torch.tensor(math.atanh(residual_init), dtype=torch.float32)
        )

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix B-H-N-D tokens into one token per variate."""
        batch_size, num_views, token_count, d_model = tokens.shape
        anchor = tokens[:, 0]
        if num_views == 1:
            weights = tokens.new_ones(batch_size, token_count, 1, 1)
            gate = tokens.new_zeros(batch_size, token_count, 1)
            scale = torch.tanh(self.residual_logit)
            return anchor, weights, gate, scale

        per_variate = tokens.permute(0, 2, 1, 3).reshape(
            batch_size * token_count, num_views, d_model
        )
        normalized = self.view_norm(per_variate)

        if self.fusion_type == "dynamic":
            query = normalized[:, :1]
            mixed, weights = self.view_attention(
                query,
                normalized,
                normalized,
                need_weights=True,
                average_attn_weights=False,
            )
            mixed = mixed[:, 0]
            weights = weights[:, :, 0, :].reshape(
                batch_size, token_count, -1, num_views
            )
        else:
            if self.fusion_type == "mean":
                view_weights = tokens.new_full(
                    (num_views,), 1.0 / num_views
                )
            else:
                view_weights = torch.softmax(self.global_view_logits, dim=0)
            mixed = torch.sum(
                normalized * view_weights.view(1, num_views, 1), dim=1
            )
            weights = view_weights.view(1, 1, 1, num_views).expand(
                batch_size, token_count, 1, num_views
            )

        correction = self.correction(mixed).reshape(
            batch_size, token_count, d_model
        )
        gate = torch.sigmoid(
            self.gate(torch.cat([anchor, correction], dim=-1))
            / self.gate_temperature
        )
        scale = torch.tanh(self.residual_logit)
        fused = anchor + scale * gate * self.dropout(correction)
        return fused, weights, gate, scale


class Model(nn.Module):
    """iTransformer with residual, intra-variate temporal view attention."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.num_token_heads = getattr(configs, "num_token_heads", 4)
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        if self.num_token_heads < 1:
            raise ValueError("num_token_heads must be at least 1.")

        # Keep this order identical to the original model so the baseline path
        # has identical initial parameters under the same random seed.
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.class_strategy = configs.class_strategy
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
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        adaptive_heads = self.num_token_heads - 1
        self.view_generator = AdaptiveTemporalViewGenerator(
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            adaptive_heads=adaptive_heads,
            scales=getattr(configs, "token_scales", "auto"),
            use_dynamic_mask=bool(getattr(configs, "use_dynamic_mask", 1)),
            mask_hidden=getattr(configs, "token_mask_hidden", 64),
            temperature=getattr(configs, "token_temperature", 1.0),
            dropout=configs.dropout,
        )
        self.view_mixer = ResidualViewMixer(
            d_model=configs.d_model,
            num_views=self.num_token_heads,
            attention_heads=getattr(configs, "view_attention_heads", 4),
            fusion_type=getattr(configs, "fusion_type", "dynamic"),
            gate_temperature=getattr(configs, "gate_temperature", 1.0),
            residual_init=getattr(configs, "view_residual_init", 0.0),
            dropout=configs.dropout,
        )

        self.lambda_redundancy = getattr(configs, "lambda_redundancy", 0.0)
        self.lambda_mask_diversity = getattr(
            configs, "lambda_mask_diversity", 0.0
        )
        if self.lambda_redundancy < 0 or self.lambda_mask_diversity < 0:
            raise ValueError("Diversity loss weights must be non-negative.")
        self._aux_state: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _diversity_loss(features: torch.Tensor) -> torch.Tensor:
        """Squared off-diagonal cosine similarity among adaptive views."""
        head_count = features.shape[1]
        if head_count <= 1:
            return features.new_zeros(())
        flattened = F.normalize(features.flatten(2), dim=-1)
        similarity = torch.matmul(flattened, flattened.transpose(1, 2))
        identity = torch.eye(
            head_count, device=features.device, dtype=features.dtype
        ).unsqueeze(0)
        return ((similarity - identity) ** 2).sum() / (
            features.shape[0] * head_count * (head_count - 1)
        )

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len:
            raise ValueError(
                f"x_enc must have shape [B, {self.seq_len}, N], "
                f"got {tuple(x_enc.shape)}."
            )
        if x_mark_enc is not None and x_mark_enc.shape[:2] != x_enc.shape[:2]:
            raise ValueError(
                "x_mark_enc must share batch and temporal dimensions with x_enc."
            )

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            normalized = x_enc - means
            stdev = torch.sqrt(
                torch.var(
                    normalized, dim=1, keepdim=True, unbiased=False
                ) + 1e-5
            )
            normalized = normalized / stdev
        else:
            means = None
            stdev = None
            normalized = x_enc

        anchor_tokens = self.enc_embedding(normalized, x_mark_enc)
        view_input = normalized.transpose(1, 2)
        if x_mark_enc is not None:
            view_input = torch.cat(
                [view_input, x_mark_enc.transpose(1, 2)], dim=1
            )

        adaptive_tokens, adaptive_masks, temporal_components = (
            self.view_generator(view_input)
        )
        all_tokens = torch.cat(
            [anchor_tokens.unsqueeze(1), adaptive_tokens], dim=1
        )
        fused_tokens, view_weights, view_gate, residual_scale = self.view_mixer(
            all_tokens
        )

        encoded, attentions = self.encoder(fused_tokens, attn_mask=None)
        prediction = self.projector(encoded).permute(0, 2, 1)
        target_count = x_enc.shape[-1]
        prediction = prediction[:, :, :target_count]
        if self.use_norm:
            prediction = prediction * stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
            prediction = prediction + means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )

        anchor_mask = torch.ones_like(view_input).unsqueeze(1)
        dynamic_masks = torch.cat([anchor_mask, adaptive_masks], dim=1)
        outputs = {
            "prediction": prediction,
            "tokens": all_tokens,
            "anchor_tokens": anchor_tokens,
            "adaptive_tokens": adaptive_tokens,
            "fused_tokens": fused_tokens,
            "temporal_components": temporal_components,
            "dynamic_masks": dynamic_masks,
            "adaptive_masks": adaptive_masks,
            "view_weights": view_weights,
            "view_gate": view_gate,
            "residual_scale": residual_scale,
        }
        if self.training and (
            self.lambda_redundancy > 0
            or self.lambda_mask_diversity > 0
        ):
            self._aux_state = outputs
        else:
            self._aux_state = None
        return outputs, attentions

    def auxiliary_loss(
        self, target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Return optional weak diversity regularizers."""
        del target
        if self._aux_state is None:
            parameter = next(self.parameters())
            zero = parameter.new_zeros(())
            return {
                "total": zero,
                "token_redundancy": zero,
                "mask_diversity": zero,
            }

        adaptive_tokens = self._aux_state["adaptive_tokens"]
        adaptive_masks = self._aux_state["adaptive_masks"]
        token_redundancy = self._diversity_loss(adaptive_tokens)
        mask_diversity = self._diversity_loss(adaptive_masks)
        total = (
            self.lambda_redundancy * token_redundancy
            + self.lambda_mask_diversity * mask_diversity
        )
        return {
            "total": total,
            "token_redundancy": token_redundancy,
            "mask_diversity": mask_diversity,
        }

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
        outputs, attentions = self.forecast(x_enc, x_mark_enc)
        prediction = outputs["prediction"][:, -self.pred_len :, :]
        if return_auxiliary:
            result = dict(outputs)
            result["prediction"] = prediction
            if self.output_attention:
                result["attentions"] = attentions
            return result
        if self.output_attention:
            return prediction, attentions
        return prediction
