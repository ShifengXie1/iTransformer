"""Grouped-variable multi-branch iTransformer.

iTransformer represents the complete history of each variable as one token
and applies attention across variable tokens. This variant partitions the
variables into disjoint groups. Every group owns an independent embedding,
iTransformer encoder, and prediction head. Group forecasts are finally
restored to the original variable order.
"""

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
    """One independent iTransformer expert for one variable group."""

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
        self.projector = nn.Linear(
            configs.d_model, configs.pred_len, bias=True
        )

    def forward(
        self,
        group_input: torch.Tensor,
        time_features: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        group_size = group_input.shape[-1]
        tokens = self.embedding(group_input, time_features)
        encoded, attentions = self.encoder(tokens, attn_mask=None)
        prediction = self.projector(encoded).permute(0, 2, 1)
        # Time-feature tokens provide context but are not prediction targets.
        prediction = prediction[:, :, :group_size]
        return prediction, encoded, attentions


class Model(nn.Module):
    """Partition variables and forecast every group with its own iTransformer."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
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
    ) -> Tuple[
        Dict[str, object],
        List[List[Optional[torch.Tensor]]],
    ]:
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

        group_predictions: List[torch.Tensor] = []
        group_tokens: List[torch.Tensor] = []
        group_attentions: List[List[Optional[torch.Tensor]]] = []
        for group, group_model in zip(
            self.variable_groups, self.group_models
        ):
            indices = torch.as_tensor(
                group, device=x_enc.device, dtype=torch.long
            )
            group_input = normalized.index_select(2, indices)
            prediction, tokens, attentions = group_model(
                group_input, x_mark_enc
            )
            group_predictions.append(prediction)
            group_tokens.append(tokens)
            group_attentions.append(attentions)

        # Concatenation follows group order; restore_order returns [0, ..., N-1].
        grouped_prediction = torch.cat(group_predictions, dim=-1)
        prediction = grouped_prediction.index_select(
            2, self.restore_order
        )

        if self.use_norm:
            prediction = prediction * stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
            prediction = prediction + means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )

        outputs: Dict[str, object] = {
            "prediction": prediction,
            "group_predictions": tuple(group_predictions),
            "group_tokens": tuple(group_tokens),
            "group_indices": self.variable_groups,
            "group_assignment": self.group_assignment,
        }
        return outputs, group_attentions

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
