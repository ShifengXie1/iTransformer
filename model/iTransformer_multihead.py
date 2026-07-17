"""Multi-token-head iTransformer with one shared variate encoder."""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class DynamicTokenGenerator(nn.Module):
    """Generate H independent tokens for every input variate."""

    def __init__(
            self, seq_len: int, d_model: int, num_heads: int,
            mask_hidden: int, temperature: float, use_dynamic_mask: bool
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError('num_token_heads must be at least 1')
        if mask_hidden < 1:
            raise ValueError('token_mask_hidden must be at least 1')
        if temperature <= 0:
            raise ValueError('token_temperature must be positive')

        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)
        self.temperature = float(temperature)
        self.use_dynamic_mask = bool(use_dynamic_mask)

        self.mask_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len, mask_hidden),
                nn.GELU(),
                nn.Linear(mask_hidden, seq_len),
            )
            for _ in range(num_heads)
        ])
        self.projections = nn.ModuleList([
            nn.Linear(seq_len, d_model) for _ in range(num_heads)
        ])
        self.normalizations = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_heads)
        ])
        self.head_embedding = nn.Parameter(torch.empty(num_heads, d_model))
        nn.init.normal_(self.head_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, L]
        if x.ndim != 3 or x.shape[-1] != self.seq_len:
            raise ValueError(
                f'Expected tokenizer input [B, N, {self.seq_len}], '
                f'got {tuple(x.shape)}'
            )

        tokens: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for head in range(self.num_heads):
            if self.use_dynamic_mask:
                mask_logits = self.mask_generators[head](x)
                mask = self.seq_len * torch.softmax(
                    mask_logits / self.temperature, dim=-1
                )
                projection_input = mask * x
            else:
                # Keeping an explicit all-one mask makes diagnostics retain the
                # same [B,H,N,L] contract in the static projection ablation.
                mask = torch.ones_like(x)
                projection_input = x

            token = self.projections[head](projection_input)
            token = token + self.head_embedding[head].view(1, 1, -1)
            token = self.normalizations[head](token)
            tokens.append(token)
            masks.append(mask)

        return torch.stack(tokens, dim=1), torch.stack(masks, dim=1)


class Model(nn.Module):
    """iTransformer with multiple token branches and a shared Encoder."""

    _FUSION_TYPES = {'mean', 'learnable_global', 'dynamic'}

    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.d_model = int(configs.d_model)
        self.num_token_heads = int(getattr(configs, 'num_token_heads', 4))
        self.output_attention = bool(configs.output_attention)
        self.use_norm = bool(configs.use_norm)
        self.fusion_type = str(
            getattr(configs, 'fusion_type', 'dynamic')
        ).lower()
        self.share_prediction_head = bool(
            getattr(configs, 'share_prediction_head', False)
        )
        self.gate_temperature = float(
            getattr(configs, 'gate_temperature', 1.0)
        )

        if self.num_token_heads < 1:
            raise ValueError('num_token_heads must be at least 1')
        if self.fusion_type not in self._FUSION_TYPES:
            raise ValueError(
                'fusion_type must be one of: mean, learnable_global, dynamic'
            )
        if self.gate_temperature <= 0:
            raise ValueError('gate_temperature must be positive')

        self.tokenizer = DynamicTokenGenerator(
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.num_token_heads,
            mask_hidden=int(getattr(configs, 'token_mask_hidden', 64)),
            temperature=float(getattr(configs, 'token_temperature', 1.0)),
            use_dynamic_mask=bool(
                getattr(configs, 'use_dynamic_mask', True)
            ),
        )

        # This is deliberately one Encoder instance. Token branches are folded
        # into the batch dimension and all reuse these exact parameters.
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
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )

        if self.share_prediction_head:
            self.prediction_head = nn.Linear(self.d_model, self.pred_len)
            self.prediction_heads = None
        else:
            self.prediction_head = None
            self.prediction_heads = nn.ModuleList([
                nn.Linear(self.d_model, self.pred_len)
                for _ in range(self.num_token_heads)
            ])

        self.global_gate_logits = nn.Parameter(
            torch.zeros(self.num_token_heads)
        )
        self.gate_network = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1),
        )

        self.lambda_branch = float(getattr(configs, 'lambda_branch', 0.2))
        self.lambda_redundancy = float(
            getattr(configs, 'lambda_redundancy', 1e-3)
        )
        self.lambda_contribution = float(
            getattr(configs, 'lambda_contribution', 0.05)
        )
        self.lambda_balance = float(
            getattr(configs, 'lambda_balance', 0.01)
        )
        self.contribution_margin = float(
            getattr(configs, 'contribution_margin', 1e-4)
        )
        for name in (
                'lambda_branch', 'lambda_redundancy',
                'lambda_contribution', 'lambda_balance',
                'contribution_margin'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be non-negative')

        self._aux_state: Optional[Dict[str, torch.Tensor]] = None

    def _predict_branches(self, representations: torch.Tensor) -> torch.Tensor:
        # representations: [B,H,N,D] -> predictions: [B,H,P,N]
        if self.share_prediction_head:
            predictions = self.prediction_head(representations)
        else:
            predictions = torch.stack([
                self.prediction_heads[head](representations[:, head])
                for head in range(self.num_token_heads)
            ], dim=1)
        return predictions.permute(0, 1, 3, 2).contiguous()

    def _fusion_weights(self, representations: torch.Tensor) -> torch.Tensor:
        batch, _, variates, _ = representations.shape
        if self.fusion_type == 'mean':
            return representations.new_full(
                (batch, self.num_token_heads, variates),
                1.0 / self.num_token_heads,
            )
        if self.fusion_type == 'learnable_global':
            weights = torch.softmax(
                self.global_gate_logits / self.gate_temperature, dim=0
            )
            return weights.view(1, -1, 1).expand(batch, -1, variates)

        gate_logits = self.gate_network(representations).squeeze(-1)
        return torch.softmax(
            gate_logits / self.gate_temperature, dim=1
        )

    @staticmethod
    def _reshape_attentions(
            attns: List[Optional[torch.Tensor]], batch: int, heads: int
    ) -> List[Optional[torch.Tensor]]:
        return [
            attention.reshape(batch, heads, *attention.shape[1:])
            if attention is not None else None
            for attention in attns
        ]

    def forecast(
            self, x_enc: torch.Tensor, x_mark_enc=None,
            x_dec=None, x_mark_dec=None
    ) -> Tuple[Dict[str, torch.Tensor], List[Optional[torch.Tensor]]]:
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len:
            raise ValueError(
                f'Expected x_enc [B, {self.seq_len}, N], '
                f'got {tuple(x_enc.shape)}'
            )

        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            normalized = x_enc - means
            stdev = torch.sqrt(
                normalized.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            normalized = normalized / stdev
        else:
            normalized = x_enc
            means = None
            stdev = None

        batch, _, variates = normalized.shape
        tokens, dynamic_masks = self.tokenizer(normalized.transpose(1, 2))
        encoder_input = tokens.reshape(
            batch * self.num_token_heads, variates, self.d_model
        )
        encoder_output, attns = self.encoder(
            encoder_input, attn_mask=None
        )
        representations = encoder_output.reshape(
            batch, self.num_token_heads, variates, self.d_model
        )

        branch_predictions = self._predict_branches(representations)
        if self.use_norm:
            branch_predictions = (
                branch_predictions * stdev[:, 0, :].unsqueeze(1).unsqueeze(1)
                + means[:, 0, :].unsqueeze(1).unsqueeze(1)
            )

        gate_weights = self._fusion_weights(representations)
        prediction = (
            branch_predictions * gate_weights.unsqueeze(2)
        ).sum(dim=1)

        outputs = {
            'prediction': prediction,
            'branch_predictions': branch_predictions,
            'tokens': tokens,
            'representations': representations,
            'gate_weights': gate_weights,
            'dynamic_masks': dynamic_masks,
        }
        self._aux_state = outputs if self.training else None
        return outputs, self._reshape_attentions(
            attns, batch, self.num_token_heads
        )

    def _aligned_state(
            self, target: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if self._aux_state is None:
            raise RuntimeError(
                'No training forward state is available. Call the model in '
                'training mode before computing multihead losses.'
            )
        state = self._aux_state
        model_channels = state['prediction'].shape[-1]
        target_channels = target.shape[-1]
        if target_channels > model_channels:
            raise ValueError('Target has more channels than model prediction')
        channel_slice = slice(model_channels - target_channels, model_channels)
        aligned = {
            'prediction': state['prediction'][..., channel_slice],
            'branch_predictions': state['branch_predictions'][..., channel_slice],
            'gate_weights': state['gate_weights'][..., channel_slice],
            # Redundancy is defined across every encoded input variable, not
            # just the selected output channel in an MS forecasting task.
            'representations': state['representations'],
        }
        return aligned, target[..., -target_channels:]

    def _redundancy_loss(self, representations: torch.Tensor) -> torch.Tensor:
        if self.num_token_heads == 1:
            return representations.new_zeros(())

        # Pearson-style cross-correlation between flattened Encoder branch
        # representations. This is linear in N*D and remains practical for
        # high-dimensional datasets such as Traffic.
        features = representations.flatten(start_dim=2)
        features = features - features.mean(dim=-1, keepdim=True)
        features = F.normalize(features, p=2, dim=-1, eps=1e-8)
        correlation = torch.matmul(features, features.transpose(1, 2))
        off_diagonal = ~torch.eye(
            self.num_token_heads,
            dtype=torch.bool,
            device=representations.device,
        ).unsqueeze(0)
        return correlation.pow(2).masked_select(off_diagonal).mean()

    def _contribution_loss(
            self, prediction: torch.Tensor,
            branch_predictions: torch.Tensor,
            gate_weights: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_token_heads == 1:
            return prediction.new_zeros(())

        # removed_weights: [B, removed_head, active_head, N]
        keep = 1.0 - torch.eye(
            self.num_token_heads,
            device=gate_weights.device,
            dtype=gate_weights.dtype,
        )
        removed_weights = (
            gate_weights.unsqueeze(1) * keep.view(
                1, self.num_token_heads, self.num_token_heads, 1
            )
        )
        removed_weights = removed_weights / removed_weights.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-8)
        leave_one_out = torch.einsum(
            'brhn,bhpn->brpn', removed_weights, branch_predictions
        )

        full_error = (prediction - target).pow(2).mean(dim=1)
        leave_one_out_error = (
            leave_one_out - target.unsqueeze(1)
        ).pow(2).mean(dim=2)
        return F.relu(
            self.contribution_margin
            + full_error.unsqueeze(1)
            - leave_one_out_error
        ).mean()

    def _balance_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        if self.num_token_heads == 1:
            return gate_weights.new_zeros(())
        # KL(gate || uniform), averaged over samples and variates.
        return (
            gate_weights * (
                torch.log(gate_weights.clamp_min(1e-8))
                + math.log(self.num_token_heads)
            )
        ).sum(dim=1).mean()

    def compute_loss(self, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return the complete forecast and diversity objective."""
        state, target = self._aligned_state(target)
        forecast_loss = F.mse_loss(state['prediction'], target)
        branch_loss = F.mse_loss(
            state['branch_predictions'],
            target.unsqueeze(1).expand_as(state['branch_predictions']),
        )
        redundancy_loss = self._redundancy_loss(state['representations'])
        contribution_loss = self._contribution_loss(
            state['prediction'],
            state['branch_predictions'],
            state['gate_weights'],
            target,
        )
        balance_loss = self._balance_loss(state['gate_weights'])
        auxiliary = (
            self.lambda_branch * branch_loss
            + self.lambda_redundancy * redundancy_loss
            + self.lambda_contribution * contribution_loss
            + self.lambda_balance * balance_loss
        )
        return {
            'total': forecast_loss + auxiliary,
            'forecast_loss': forecast_loss,
            'branch_loss': branch_loss,
            'redundancy_loss': redundancy_loss,
            'contribution_loss': contribution_loss,
            'balance_loss': balance_loss,
            'auxiliary_loss': auxiliary,
        }

    def auxiliary_loss(self, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Adapter for Exp_Long_Term_Forecast's model-owned loss hook."""
        if self._aux_state is None:
            zero = target.new_zeros(())
            return {'total': zero}
        losses = self.compute_loss(target)
        return {
            'total': losses['auxiliary_loss'],
            'total_loss': losses['total'],
            'forecast_loss': losses['forecast_loss'],
            'branch_loss': losses['branch_loss'],
            'redundancy_loss': losses['redundancy_loss'],
            'contribution_loss': losses['contribution_loss'],
            'balance_loss': losses['balance_loss'],
        }

    def forward(
            self, x_enc: torch.Tensor, x_mark_enc=None,
            x_dec=None, x_mark_dec=None, mask=None,
            return_auxiliary: bool = False,
    ):
        outputs, attns = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        if return_auxiliary:
            return outputs
        if self.output_attention:
            return outputs['prediction'], attns
        return outputs['prediction']
