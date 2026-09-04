"""Explicit grouped multi-head iTransformer.

iTransformer represents the complete history of each variable as one token
and applies attention across variable tokens. This variant partitions the
variables into disjoint groups. Every group owns an independent embedding and
iTransformer encoder, then acts as one explicit attention head. The head
contexts are concatenated and decoded by one shared prediction head.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


VariableGroups = Tuple[Tuple[int, ...], ...]


def _balanced_groups(
    num_variables: int, num_groups: int
) -> VariableGroups:
    """Split consecutive variable indices into balanced, non-empty groups."""
    if num_variables < 1:
        raise ValueError("enc_in must be at least 1.")
    if num_groups < 1:
        raise ValueError("num_variable_groups must be at least 1.")
    if num_groups > num_variables:
        raise ValueError(
            "num_variable_groups cannot exceed the number of input variables."
        )

    base_size, remainder = divmod(num_variables, num_groups)
    groups: List[Tuple[int, ...]] = []
    start = 0
    for group_index in range(num_groups):
        size = base_size + int(group_index < remainder)
        groups.append(tuple(range(start, start + size)))
        start += size
    return tuple(groups)


def parse_variable_groups(
    specification: str | Sequence[Sequence[int]],
    num_variables: int,
    num_groups: int,
) -> VariableGroups:
    """Parse auto or semicolon-separated disjoint variable groups.

    Example: 0,2,4;1,3,5,6 creates two groups. Every index from 0 to
    num_variables - 1 must occur exactly once.
    """
    if isinstance(specification, str):
        normalized = specification.strip().lower()
        if normalized == "auto":
            return _balanced_groups(num_variables, num_groups)
        if not normalized:
            raise ValueError("variable_groups cannot be empty.")
        groups = tuple(
            tuple(
                int(item.strip())
                for item in group_text.split(",")
                if item.strip()
            )
            for group_text in specification.split(";")
        )
    else:
        groups = tuple(
            tuple(int(index) for index in group)
            for group in specification
        )

    if len(groups) != num_groups:
        raise ValueError(
            f"num_variable_groups={num_groups}, but the explicit "
            f"specification defines {len(groups)} groups."
        )
    if any(len(group) == 0 for group in groups):
        raise ValueError("Variable groups must be non-empty.")

    flattened = [index for group in groups for index in group]
    expected = list(range(num_variables))
    if sorted(flattened) != expected:
        raise ValueError(
            "variable_groups must contain every variable index exactly once; "
            f"expected {expected}, got {flattened}."
        )
    return groups


class GroupITransformer(nn.Module):
    """One independent iTransformer encoder for one variable group."""

    def __init__(self, configs) -> None:
        super().__init__()
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
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
    def forward(
        self,
        group_input: torch.Tensor,
        time_features: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        group_size = group_input.shape[-1]
        tokens = self.embedding(group_input, time_features)
        encoded, attentions = self.encoder(tokens, attn_mask=None)
        # Time-feature tokens have already provided context through attention.
        return encoded[:, :group_size], attentions


class ExplicitGroupMultiHeadAttention(nn.Module):
    """Use each encoded variable group as one explicit attention head."""

    def __init__(
        self,
        d_model: int,
        num_groups: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if head_dim < 0:
            raise ValueError("group_head_dim must be non-negative.")
        if head_dim == 0:
            head_dim = math.ceil(d_model / num_groups)

        self.num_groups = num_groups
        self.head_dim = head_dim
        self.query_norm = nn.LayerNorm(d_model)
        self.group_norms = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in range(num_groups)
        )
        self.query_projections = nn.ModuleList(
            nn.Linear(d_model, head_dim) for _ in range(num_groups)
        )
        self.key_projections = nn.ModuleList(
            nn.Linear(d_model, head_dim) for _ in range(num_groups)
        )
        self.value_projections = nn.ModuleList(
            nn.Linear(d_model, head_dim) for _ in range(num_groups)
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(
            num_groups * head_dim, d_model
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        all_variable_tokens: torch.Tensor,
        encoded_groups: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(encoded_groups) != self.num_groups:
            raise ValueError(
                f"Expected {self.num_groups} encoded groups, "
                f"got {len(encoded_groups)}."
            )

        normalized_queries = self.query_norm(all_variable_tokens)
        head_contexts: List[torch.Tensor] = []
        attention_maps: List[torch.Tensor] = []
        for group_index, group_tokens in enumerate(encoded_groups):
            normalized_group = self.group_norms[group_index](group_tokens)
            query = self.query_projections[group_index](
                normalized_queries
            )
            key = self.key_projections[group_index](normalized_group)
            value = self.value_projections[group_index](normalized_group)

            scores = torch.matmul(
                query, key.transpose(-1, -2)
            ) / math.sqrt(self.head_dim)
            attention = torch.softmax(scores, dim=-1)
            context = torch.matmul(
                self.attention_dropout(attention), value
            )
            head_contexts.append(context)
            attention_maps.append(attention)

        concatenated = torch.cat(head_contexts, dim=-1)
        context = self.output_dropout(
            self.output_projection(concatenated)
        )
        return context, tuple(attention_maps)


class Model(nn.Module):
    """Encode variable groups, explicitly attend to them, and predict once."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.num_variables = configs.enc_in
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.class_strategy = configs.class_strategy
        self.num_variable_groups = getattr(
            configs,
            "num_variable_groups",
            getattr(configs, "num_token_heads", 2),
        )
        self.variable_groups = parse_variable_groups(
            getattr(configs, "variable_groups", "auto"),
            self.num_variables,
            self.num_variable_groups,
        )

        self.group_models = nn.ModuleList(
            GroupITransformer(configs) for _ in self.variable_groups
        )
        # One shared head predicts all variables after group-head fusion.
        # It is created before the extra attention modules so the one-group,
        # zero-residual case preserves original iTransformer initialization.
        self.projector = nn.Linear(
            configs.d_model, configs.pred_len, bias=True
        )
        self.group_attention = ExplicitGroupMultiHeadAttention(
            d_model=configs.d_model,
            num_groups=self.num_variable_groups,
            head_dim=getattr(configs, "group_head_dim", 0),
            dropout=configs.dropout,
        )
        self.context_gate = nn.Sequential(
            nn.LayerNorm(2 * configs.d_model),
            nn.Linear(2 * configs.d_model, 1),
        )
        residual_init = getattr(configs, "group_residual_init", 0.1)
        if not -1.0 < residual_init < 1.0:
            raise ValueError("group_residual_init must lie in (-1, 1).")
        self.residual_logit = nn.Parameter(
            torch.tensor(math.atanh(residual_init), dtype=torch.float32)
        )

        grouped_order = [
            variable
            for group in self.variable_groups
            for variable in group
        ]
        restore_order = torch.tensor(
            grouped_order, dtype=torch.long
        ).argsort()
        self.register_buffer(
            "restore_order", restore_order, persistent=False
        )

        assignment = torch.zeros(
            self.num_variable_groups, self.num_variables
        )
        for group_index, group in enumerate(self.variable_groups):
            assignment[group_index, list(group)] = 1.0
        self.register_buffer(
            "group_assignment", assignment, persistent=False
        )

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
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
                f"Expected enc_in={self.num_variables} variables, "
                f"got {x_enc.shape[2]}."
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

        group_tokens: List[torch.Tensor] = []
        group_attentions: List[List[Optional[torch.Tensor]]] = []
        for group, group_model in zip(
            self.variable_groups, self.group_models
        ):
            indices = torch.as_tensor(
                group, device=x_enc.device, dtype=torch.long
            )
            group_input = normalized.index_select(2, indices)
            tokens, attentions = group_model(group_input, x_mark_enc)
            group_tokens.append(tokens)
            group_attentions.append(attentions)

        # Concatenation follows group order; restore variable order before the
        # common-query attention and shared prediction head.
        grouped_tokens = torch.cat(group_tokens, dim=1)
        local_tokens = grouped_tokens.index_select(
            1, self.restore_order
        )
        cross_group_context, group_attention_maps = self.group_attention(
            local_tokens, group_tokens
        )
        context_gate = torch.sigmoid(
            self.context_gate(
                torch.cat([local_tokens, cross_group_context], dim=-1)
            )
        )
        residual_scale = torch.tanh(self.residual_logit)
        fused_tokens = (
            local_tokens
            + residual_scale * context_gate * cross_group_context
        )
        prediction = self.projector(fused_tokens).permute(0, 2, 1)

        if self.use_norm:
            prediction = prediction * stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
            prediction = prediction + means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )

        outputs: Dict[str, object] = {
            "prediction": prediction,
            "group_encoded_tokens": tuple(group_tokens),
            "local_tokens": local_tokens,
            "cross_group_context": cross_group_context,
            "fused_tokens": fused_tokens,
            "group_attention_maps": group_attention_maps,
            "context_gate": context_gate,
            "residual_scale": residual_scale,
            "group_indices": self.variable_groups,
            "group_assignment": self.group_assignment,
        }
        attentions: Dict[str, object] = {
            "within_group": tuple(group_attentions),
            "between_groups": group_attention_maps,
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
