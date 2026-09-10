"""Transferability-aware dual retrieval for multivariate forecasting.

The model deliberately differs from residual-only post-processing.  A frozen
iTransformer supplies the base forecast and stable final-encoder retrieval
keys.  One variable-wise, causally filtered neighbor set produces two forms of
evidence:

* a query-adapted historical continuation (future expert), and
* a scale-transported historical base-model error (residual expert).

Local variable keys determine which cases are retained.  Pooled multivariate
context can only reweight those cases, so a global state cannot discard a
variable-specific neighbor.  A three-way gate selects base/future/residual for
every sample, variable and horizon.  Nonnegative uncertainty and disagreement
penalties make an expert's logit monotonically decrease as its evidence becomes
less reliable.  An optional smooth horizon basis changes confidence without
introducing H independent gate heads.

Training is intentionally staged.  Train the backbone normally, freeze it,
build the memory from that exact checkpoint, and then train only the retrieval
projections, continuation transport and gate.  Replacing the backbone invalidates
both residual values and keys and therefore requires rebuilding the memory.
"""

import math
from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from model.iTransformer import Model as OriginalITransformer


class DualRetrievedEvidence(NamedTuple):
    future: torch.Tensor          # [B,N,H], query-scaled historical continuation
    residual: torch.Tensor        # [B,N,H], correction added to the base forecast
    indices: torch.Tensor         # [B,N,K], -1 for unavailable neighbors
    weights: torch.Tensor         # [B,N,K]
    similarity: torch.Tensor      # [B,N,1], weighted local/global score
    similarity_stats: torch.Tensor  # [B,N,4]: mean,max,std,range
    future_variance: torch.Tensor   # [B,N,H]
    residual_variance: torch.Tensor # [B,N,H]
    candidate_count: torch.Tensor   # [B,N]


class Model(OriginalITransformer):
    def __init__(self, configs):
        super().__init__(configs)
        self.dual_top_k = int(getattr(configs, 'dual_top_k', 64))
        self.dual_temperature = float(getattr(configs, 'dual_temperature', 0.1))
        self.dual_memory_size = int(getattr(configs, 'dual_memory_size', 4096))
        self.dual_stride = int(getattr(configs, 'dual_stride', 1))
        self.dual_chunk_size = int(getattr(configs, 'dual_chunk_size', 128))
        self.dual_variable_chunk_size = int(getattr(configs, 'dual_variable_chunk_size', 32))
        self.dual_memory_batch_size = int(getattr(configs, 'dual_memory_batch_size', 128))
        self.dual_search_metric = str(getattr(configs, 'dual_search_metric', 'l2')).lower()
        self.dual_use_future = bool(getattr(configs, 'dual_use_future', True))
        self.dual_use_residual = bool(getattr(configs, 'dual_use_residual', True))
        self.dual_use_global = bool(getattr(configs, 'dual_use_global', True))
        self.dual_horizon_gate = bool(getattr(configs, 'dual_horizon_gate', True))
        self.dual_scale_residual = bool(getattr(configs, 'dual_scale_residual', True))
        causal_gap = int(getattr(configs, 'dual_causal_gap', -1))
        self.dual_causal_gap = self.pred_len if causal_gap < 0 else causal_gap

        positive = (self.dual_top_k, self.dual_stride, self.dual_chunk_size,
                    self.dual_variable_chunk_size, self.dual_memory_batch_size)
        if min(positive) < 1 or self.dual_memory_size < 0 or self.dual_causal_gap < 0:
            raise ValueError('Dual retrieval sizes must be positive; memory_size and causal_gap may be zero')
        if not math.isfinite(self.dual_temperature) or self.dual_temperature <= 0:
            raise ValueError('dual_temperature must be finite and positive')
        if self.dual_search_metric not in ('l2', 'cosine'):
            raise ValueError("dual_search_metric must be 'l2' or 'cosine'")
        if not (self.dual_use_future or self.dual_use_residual):
            raise ValueError('At least one dual retrieval expert must be enabled')

        d_model = configs.d_model
        self.local_projection = nn.Linear(d_model, d_model, bias=False)
        self.global_projection = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.local_projection.weight)
        nn.init.eye_(self.global_projection.weight)

        global_weight = float(getattr(configs, 'dual_global_weight', 0.5))
        if not math.isfinite(global_weight) or not 0 <= global_weight <= 1:
            raise ValueError('dual_global_weight must lie in [0, 1]')
        clipped_weight = min(max(global_weight, 1e-4), 1 - 1e-4)
        self.global_weight_logit = nn.Parameter(torch.tensor(math.log(clipped_weight / (1 - clipped_weight))))

        # A zero-initialized transfer preserves the retrieved continuation at
        # initialization.  It is shared across variables and memory cases.
        self.continuation_adapter = nn.Linear(self.seq_len, self.pred_len, bias=False)
        nn.init.zeros_(self.continuation_adapter.weight)

        # Bounded residual gain and scale exponent both initialize to one and
        # one half respectively.  The exponent transports residual magnitude
        # without forcing full proportional scaling.
        self.residual_gain_logit = nn.Parameter(torch.zeros(()))
        self.residual_scale_power_logit = nn.Parameter(torch.zeros(()))

        self.expert_gate = nn.Sequential(
            nn.Linear(d_model + 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        nn.init.normal_(self.expert_gate[-1].weight, std=0.02)
        nn.init.constant_(self.expert_gate[-1].bias, -1.5)
        # Rows: future/residual; columns: variance/base-disagreement.
        self.reliability_penalty = nn.Parameter(torch.zeros(2, 2))

        if self.dual_horizon_gate:
            self.horizon_gate = nn.Linear(d_model, 6)
            nn.init.zeros_(self.horizon_gate.weight)
            nn.init.zeros_(self.horizon_gate.bias)
            position = torch.linspace(-1., 1., self.pred_len)
            basis = torch.stack((position,
                                 (3 * position.square() - 1) / 2,
                                 (5 * position.pow(3) - 3 * position) / 2))
            self.register_buffer('horizon_basis', basis, persistent=False)

        self.register_buffer('memory_keys', torch.empty(0, 0, 0))
        self.register_buffer('memory_global_keys', torch.empty(0, 0))
        self.register_buffer('memory_past', torch.empty(0, 0, 0))
        self.register_buffer('memory_future', torch.empty(0, 0, 0))
        self.register_buffer('memory_residual', torch.empty(0, 0, 0))
        self.register_buffer('memory_past_std', torch.empty(0, 0))
        self.register_buffer('memory_starts', torch.empty(0, dtype=torch.long))
        self.register_buffer('memory_mean', torch.empty(0))
        self.register_buffer('memory_scale', torch.empty(0))
        self.register_buffer('memory_ready', torch.tensor(False))
        self.register_buffer('dual_gamma', torch.tensor(1.0))

    @property
    def retrieval_ready(self):
        return bool(self.memory_ready.item()) and self.memory_starts.numel() > 0

    def _backbone_modules(self):
        return self.enc_embedding, self.encoder, self.projector

    def set_base_stage(self):
        """Enable ordinary iTransformer training and invalidate old memory."""
        self.memory_ready.fill_(False)
        for module in self._backbone_modules():
            module.train(self.training)
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        for name, parameter in self.named_parameters():
            if not name.startswith(('enc_embedding.', 'encoder.', 'projector.')):
                parameter.requires_grad_(False)

    def set_retrieval_stage(self):
        """Freeze the exact backbone used to construct residual memory."""
        if not self.memory_starts.numel():
            raise RuntimeError('Cannot start retrieval training before building memory')
        self.memory_ready.fill_(True)
        for module in self._backbone_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if not name.startswith(('enc_embedding.', 'encoder.', 'projector.')):
                parameter.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.retrieval_ready:
            # A frozen base must remain deterministic while the retrieval
            # modules train with model.train().
            for module in self._backbone_modules():
                module.eval()
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in ('memory_keys', 'memory_global_keys', 'memory_past',
                     'memory_future', 'memory_residual', 'memory_past_std',
                     'memory_starts', 'memory_mean', 'memory_scale'):
            value = state_dict.get(prefix + name)
            if value is not None:
                current = getattr(self, name)
                setattr(self, name, current.new_empty(value.shape))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if self.retrieval_ready:
            for module in self._backbone_modules():
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def _encode_backbone(self, x_enc, x_mark_enc):
        """Return base [B,H,N], final variable keys and normalization state."""
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len:
            raise ValueError('x_enc must have shape [B,seq_len,N]')
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            centered = x_enc - means
            stdev = (centered.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
            normalized = centered / stdev
        else:
            means = x_enc.new_zeros(x_enc.size(0), 1, x_enc.size(2))
            stdev = x_enc.new_ones(x_enc.size(0), 1, x_enc.size(2))
            normalized = x_enc
        inputs = normalized.permute(0, 2, 1)
        if x_mark_enc is not None:
            inputs = torch.cat((inputs, x_mark_enc.permute(0, 2, 1)), dim=1)
        tokens = self.enc_embedding.value_embedding(inputs)
        encoded, attention = self.encoder(self.enc_embedding.dropout(tokens), attn_mask=None)
        variables = x_enc.size(2)
        keys = encoded[:, :variables]
        base = self.projector(keys).permute(0, 2, 1)
        if self.use_norm:
            base = base * stdev + means
        return base, keys, normalized.permute(0, 2, 1), means, stdev, attention

    @torch.no_grad()
    def build_memory(self, memory_x, memory_y, memory_x_mark=None, starts=None):
        """Encode a frozen-backbone memory from aligned training samples.

        memory_x is [M,L,N].  memory_y may contain label_len context; its last H
        positions are used.  starts are the corresponding training-window starts.
        """
        device = self.projector.weight.device
        dtype = self.projector.weight.dtype
        memory_x = torch.as_tensor(memory_x)
        memory_y = torch.as_tensor(memory_y)
        if memory_x.ndim != 3 or memory_x.size(1) != self.seq_len:
            raise ValueError('memory_x must be [M,seq_len,N]')
        if memory_y.ndim != 3 or memory_y.size(0) != memory_x.size(0) or memory_y.size(1) < self.pred_len:
            raise ValueError('memory_y must align with memory_x and contain pred_len targets')
        if memory_y.size(2) != memory_x.size(2):
            raise ValueError('Memory input and target variable counts must match')
        if memory_x_mark is not None:
            memory_x_mark = torch.as_tensor(memory_x_mark)
            if memory_x_mark.shape[:2] != memory_x.shape[:2]:
                raise ValueError('memory_x_mark must align with memory_x in batch and time')
        if starts is None:
            starts = torch.arange(memory_x.size(0))
        starts = torch.as_tensor(starts, dtype=torch.long)
        if starts.shape != (memory_x.size(0),) or (starts[1:] < starts[:-1]).any():
            raise ValueError('starts must be a sorted integer vector with one entry per memory sample')

        self.eval()
        for module in self._backbone_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        keys, pasts, futures, residuals, past_stds = [], [], [], [], []
        for first in range(0, memory_x.size(0), self.dual_memory_batch_size):
            last = min(first + self.dual_memory_batch_size, memory_x.size(0))
            x = memory_x[first:last].to(device=device, dtype=dtype)
            y = memory_y[first:last, -self.pred_len:].to(device=device, dtype=dtype)
            marks = (None if memory_x_mark is None else
                     memory_x_mark[first:last].to(device=device, dtype=dtype))
            base, key, normalized, means, stdev, _ = self._encode_backbone(x, marks)
            future = ((y - means) / stdev).permute(0, 2, 1)
            residual = (y - base).permute(0, 2, 1)
            keys.append(key.detach())
            pasts.append(normalized.detach())
            futures.append(future.detach())
            residuals.append(residual.detach())
            past_stds.append(stdev.squeeze(1).detach())

        self.memory_keys = torch.cat(keys, 0)
        self.memory_global_keys = self.memory_keys.mean(1)
        self.memory_past = torch.cat(pasts, 0)
        self.memory_future = torch.cat(futures, 0)
        self.memory_residual = torch.cat(residuals, 0)
        self.memory_past_std = torch.cat(past_stds, 0)
        self.memory_starts = starts.to(device=device)
        self.memory_ready.fill_(True)
        self.set_retrieval_stage()

    def set_gamma(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('dual gamma must lie in [0, 1]')
        self.dual_gamma.fill_(value)

    def _score(self, query, keys):
        if self.dual_search_metric == 'cosine':
            return (F.normalize(query.float(), dim=-1) *
                    F.normalize(keys.float(), dim=-1)).sum(-1)
        return -(query.float() - keys.float()).square().mean(-1)

    @torch.no_grad()
    def _select_neighbors(self, query_keys, query_end, first_var):
        batch, variables, _ = query_keys.shape
        count = self.memory_starts.numel()
        k = min(self.dual_top_k, count)
        best_scores = query_keys.new_empty(batch, variables, 0, dtype=torch.float32)
        best_indices = torch.empty(batch, variables, 0, dtype=torch.long, device=query_keys.device)
        for start in range(0, count, self.dual_chunk_size):
            stop = min(start + self.dual_chunk_size, count)
            keys = self.memory_keys[start:stop, first_var:first_var + variables]
            if self.dual_search_metric == 'cosine':
                scores = torch.einsum('bnd,mnd->bnm', F.normalize(query_keys.float(), dim=-1),
                                      F.normalize(keys.float(), dim=-1))
            else:
                scores = -(query_keys.float().unsqueeze(2) - keys.float().permute(1, 0, 2).unsqueeze(0)).square().mean(-1)
            if query_end is not None:
                availability = (self.memory_starts[start:stop] + self.seq_len +
                                self.pred_len + self.dual_causal_gap)
                scores = scores.masked_fill(
                    availability[None, None, :] > query_end[:, None, None], -torch.inf)
            indices = torch.arange(start, stop, device=query_keys.device).expand(batch, variables, -1)
            scores = torch.cat((best_scores, scores), dim=-1)
            indices = torch.cat((best_indices, indices), dim=-1)
            best_scores, positions = scores.topk(min(k, scores.size(-1)), dim=-1)
            best_indices = indices.gather(-1, positions)
        return best_indices, torch.isfinite(best_scores)

    def retrieve(self, query_keys, query_past, query_means, query_stds, query_end=None):
        if not self.retrieval_ready:
            raise RuntimeError('Dual retrieval memory is not ready')
        if query_keys.ndim != 3 or query_keys.size(1) != self.memory_keys.size(1):
            raise ValueError('Query variables must match memory variables')
        if query_end is not None:
            query_end = torch.as_tensor(query_end, device=query_keys.device)
            if query_end.shape != (query_keys.size(0),) or query_end.dtype not in (torch.int32, torch.int64):
                raise ValueError('query_end must be an integer tensor of shape [B]')

        futures, residuals, indices_out, weights_out = [], [], [], []
        similarities, statistics, future_variances, residual_variances = [], [], [], []
        counts = []
        global_query = query_keys.mean(1)
        mix = (torch.sigmoid(self.global_weight_logit.float()) if self.dual_use_global
               else query_keys.new_zeros((), dtype=torch.float32))
        query_mean = query_means.squeeze(1)
        query_std = query_stds.squeeze(1)

        with torch.autocast(device_type=query_keys.device.type, enabled=False):
            for first in range(0, query_keys.size(1), self.dual_variable_chunk_size):
                last = min(first + self.dual_variable_chunk_size, query_keys.size(1))
                query = query_keys[:, first:last].float()
                indices, valid = self._select_neighbors(query, query_end, first)
                safe_indices = indices.clamp_min(0)
                channels = torch.arange(first, last, device=query.device)[None, :, None]

                selected_keys = self.memory_keys[safe_indices, channels]
                local_scores = self._score(
                    self.local_projection(query).unsqueeze(2),
                    self.local_projection(selected_keys))
                selected_global = self.memory_global_keys[safe_indices]
                global_scores = self._score(
                    self.global_projection(global_query.float())[:, None, None, :],
                    self.global_projection(selected_global.float()))
                scores = (1 - mix) * local_scores + mix * global_scores
                logits = (scores / self.dual_temperature).masked_fill(~valid, -torch.inf)
                logits = torch.where(valid.any(-1, keepdim=True), logits, torch.zeros_like(logits))
                weights = logits.softmax(-1) * valid

                memory_past = self.memory_past[safe_indices, channels]
                future_candidates = self.memory_future[safe_indices, channels]
                adapted = future_candidates + self.continuation_adapter(
                    query_past[:, first:last].float().unsqueeze(2) - memory_past.float())
                future_candidates = (adapted * query_std[:, first:last, None, None].float() +
                                     query_mean[:, first:last, None, None].float())

                residual_candidates = self.memory_residual[safe_indices, channels].float()
                if self.dual_scale_residual:
                    ratio = (query_std[:, first:last, None].float() /
                             self.memory_past_std[safe_indices, channels].float().clamp_min(1e-5))
                    power = torch.sigmoid(self.residual_scale_power_logit.float())
                    residual_candidates = residual_candidates * ratio.clamp(0.25, 4).pow(power).unsqueeze(-1)
                gain = 2 * torch.sigmoid(self.residual_gain_logit.float())
                residual_candidates = gain * residual_candidates

                future = (weights.unsqueeze(-1) * future_candidates).sum(2)
                residual = (weights.unsqueeze(-1) * residual_candidates).sum(2)
                future_variance = (weights.unsqueeze(-1) *
                                   (future_candidates - future.unsqueeze(2)).square()).sum(2)
                residual_variance = (weights.unsqueeze(-1) *
                                     (residual_candidates - residual.unsqueeze(2)).square()).sum(2)

                valid_count = valid.sum(-1, keepdim=True).clamp_min(1)
                has_history = valid.any(-1, keepdim=True)
                mean_score = (scores * valid).sum(-1, keepdim=True) / valid_count
                score_var = ((scores - mean_score).square() * valid).sum(-1, keepdim=True) / valid_count
                score_std = torch.where(score_var > 0, score_var.clamp_min(1e-12).sqrt(), 0.)
                maximum = torch.where(has_history,
                                      scores.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True), 0.)
                minimum = torch.where(has_history,
                                      scores.masked_fill(~valid, torch.inf).amin(-1, keepdim=True), 0.)

                if query_end is None:
                    candidate_count = torch.full(
                        (query.size(0), last - first), self.memory_starts.numel(),
                        device=query.device, dtype=torch.long)
                else:
                    availability = (self.memory_starts + self.seq_len + self.pred_len +
                                    self.dual_causal_gap)
                    per_batch = torch.searchsorted(availability, query_end, right=True)
                    candidate_count = per_batch[:, None].expand(-1, last - first)

                futures.append(future)
                residuals.append(residual)
                indices_out.append(indices.masked_fill(~valid, -1))
                weights_out.append(weights)
                similarities.append((weights * scores).sum(-1, keepdim=True))
                statistics.append(torch.cat((mean_score, maximum, score_std, maximum - minimum), -1))
                future_variances.append(future_variance)
                residual_variances.append(residual_variance)
                counts.append(candidate_count)

        return DualRetrievedEvidence(
            torch.cat(futures, 1), torch.cat(residuals, 1),
            torch.cat(indices_out, 1), torch.cat(weights_out, 1),
            torch.cat(similarities, 1), torch.cat(statistics, 1),
            torch.cat(future_variances, 1), torch.cat(residual_variances, 1),
            torch.cat(counts, 1))

    def fuse_prediction(self, base, query_keys, evidence):
        available = (evidence.indices >= 0).any(-1, keepdim=True)
        context = torch.cat((query_keys.detach(), evidence.similarity_stats.detach().to(query_keys.dtype)), -1)
        hidden = self.expert_gate[1](self.expert_gate[0](context))
        confidence = self.expert_gate[2](hidden).float().unsqueeze(2).expand(-1, -1, self.pred_len, -1)
        if self.dual_horizon_gate:
            horizon_coefficients = self.horizon_gate(hidden.float()).view(*hidden.shape[:2], 2, 3)
            curve = torch.einsum('bnrc,ch->bnhr', horizon_coefficients, self.horizon_basis.float())
            confidence = confidence + curve

        base_bnh = base.permute(0, 2, 1).float()
        future_disagreement = (evidence.future.detach().float() - base_bnh.detach()).abs()
        residual_disagreement = evidence.residual.detach().float().abs()
        penalties = F.softplus(self.reliability_penalty.float())
        future_logit = (confidence[..., 0]
                        - penalties[0, 0] * torch.log1p(evidence.future_variance.detach().float().clamp_min(0))
                        - penalties[0, 1] * torch.log1p(future_disagreement))
        residual_logit = (confidence[..., 1]
                          - penalties[1, 0] * torch.log1p(evidence.residual_variance.detach().float().clamp_min(0))
                          - penalties[1, 1] * torch.log1p(residual_disagreement))
        if not self.dual_use_future:
            future_logit = torch.full_like(future_logit, -torch.inf)
        if not self.dual_use_residual:
            residual_logit = torch.full_like(residual_logit, -torch.inf)
        logits = torch.stack((torch.zeros_like(future_logit), future_logit, residual_logit), -1)
        gates = logits.softmax(-1)
        fallback = torch.zeros_like(gates)
        fallback[..., 0] = 1
        gates = torch.where(available.unsqueeze(2), gates, fallback)

        future = torch.where(available, evidence.future, base_bnh)
        residual_prediction = torch.where(available, base_bnh + evidence.residual, base_bnh)
        mixture = (gates[..., 0] * base_bnh + gates[..., 1] * future +
                   gates[..., 2] * residual_prediction)
        gamma = self.dual_gamma.float()
        prediction = base_bnh + gamma * (mixture - base_bnh)

        effective_future = gamma * gates[..., 1]
        effective_residual = gamma * gates[..., 2]
        effective_base = 1 - effective_future - effective_residual
        return (prediction.permute(0, 2, 1), mixture.permute(0, 2, 1),
                future.permute(0, 2, 1), residual_prediction.permute(0, 2, 1),
                effective_base.permute(0, 2, 1), effective_future.permute(0, 2, 1),
                effective_residual.permute(0, 2, 1), available.squeeze(-1))

    def _base_components(self, base):
        batch, horizon, variables = base.shape
        zeros = base.new_zeros(batch, horizon, variables)
        return dict(prediction=base, base=base, retrieval=base, future=base, residual=base,
                    gate=zeros, base_gate=torch.ones_like(base), future_gate=zeros,
                    residual_gate=zeros, available=zeros[:, 0].bool(),
                    candidate_count=zeros[:, 0], similarity=zeros[:, 0],
                    scale_ratio_mean=torch.ones_like(zeros[:, 0]),
                    scale_ratio_max=torch.ones_like(zeros[:, 0]),
                    future_variance=zeros, residual_variance=zeros,
                    gamma=self.dual_gamma.detach().clone())

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                 query_end=None, return_components=False):
        base, keys, query_past, means, stdev, attention = self._encode_backbone(x_enc, x_mark_enc)
        if not self.retrieval_ready:
            if return_components:
                return base, attention, self._base_components(base)
            return base, attention

        evidence = self.retrieve(keys.detach(), query_past.detach(), means, stdev, query_end)
        (prediction, mixture, future, residual, base_gate, future_gate,
         residual_gate, available) = self.fuse_prediction(base.detach(), keys.detach(), evidence)
        if return_components:
            query_std = stdev.squeeze(1).float()
            safe_indices = evidence.indices.clamp_min(0)
            channels = torch.arange(keys.size(1), device=keys.device)[None, :, None]
            ratios = query_std.unsqueeze(-1) / self.memory_past_std[safe_indices, channels].float().clamp_min(1e-5)
            valid = evidence.indices >= 0
            ratio_mean = (ratios * evidence.weights).sum(-1)
            ratio_max = ratios.masked_fill(~valid, 0.).amax(-1)
            components = dict(
                prediction=prediction, base=base.detach(), retrieval=mixture,
                future=future, residual=residual,
                gate=(future_gate + residual_gate).detach(),
                base_gate=base_gate.detach(), future_gate=future_gate.detach(),
                residual_gate=residual_gate.detach(), available=available,
                candidate_count=evidence.candidate_count,
                similarity=evidence.similarity.detach().squeeze(-1),
                scale_ratio_mean=ratio_mean.detach(), scale_ratio_max=ratio_max.detach(),
                future_variance=evidence.future_variance.detach().permute(0, 2, 1),
                residual_variance=evidence.residual_variance.detach().permute(0, 2, 1),
                gamma=self.dual_gamma.detach().clone())
            return prediction, attention, components
        return prediction, attention

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                query_end=None, return_components=False):
        if return_components:
            _, _, components = self.forecast(
                x_enc, x_mark_enc, x_dec, x_mark_dec, query_end, True)
            return components
        prediction, attention = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec, query_end)
        prediction = prediction[:, -self.pred_len:]
        return (prediction, attention) if self.output_attention else prediction

    def base_auxiliary_loss(self, components, target):
        # The backbone is trained in its own stage and frozen during retrieval.
        return target.new_zeros(())


class _IndexedTrainingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (*self.dataset[index], index + self.dataset.seq_len)


@torch.no_grad()
def initialize_dual_retrieval_memory(model, dataset, loader, use_time_marks=True):
    """Build stable key/future/residual memory and retain loader sampling."""
    core = model.module if isinstance(model, nn.DataParallel) else model
    if not isinstance(core, Model):
        return loader
    if isinstance(model, nn.DataParallel):
        raise ValueError('Dual retrieval memory currently supports one device; omit --use_multi_gpu')

    # Excluding the final causal-gap starts makes the entire fixed train-only
    # memory admissible at the first validation origin as well.
    stop = len(dataset) - core.dual_causal_gap
    if stop <= 0:
        raise ValueError('Training data is too short for the configured causal gap')
    starts = torch.arange(0, stop, core.dual_stride, dtype=torch.long)
    if core.dual_memory_size and starts.numel() > core.dual_memory_size:
        positions = torch.linspace(0, starts.numel() - 1, core.dual_memory_size).round().long()
        starts = starts[positions]
    memory_loader = DataLoader(
        Subset(dataset, starts.tolist()), batch_size=core.dual_memory_batch_size,
        shuffle=False, drop_last=False, num_workers=loader.num_workers,
        collate_fn=loader.collate_fn, pin_memory=loader.pin_memory)
    xs, ys, marks = [], [], []
    for x, y, x_mark, _ in memory_loader:
        xs.append(x)
        ys.append(y)
        if use_time_marks:
            marks.append(x_mark)
    core.build_memory(torch.cat(xs), torch.cat(ys),
                      torch.cat(marks) if use_time_marks else None, starts)
    if getattr(dataset, 'scale', False):
        core.memory_mean = torch.as_tensor(
            dataset.scaler.mean_, device=core.memory_keys.device,
            dtype=core.memory_keys.dtype).clone()
        core.memory_scale = torch.as_tensor(
            dataset.scaler.scale_, device=core.memory_keys.device,
            dtype=core.memory_keys.dtype).clone()
    return DataLoader(
        _IndexedTrainingDataset(dataset), batch_sampler=loader.batch_sampler,
        num_workers=loader.num_workers, collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory, worker_init_fn=loader.worker_init_fn,
        generator=loader.generator)


def align_dual_prediction_data(model, dataset):
    core = model.module if isinstance(model, nn.DataParallel) else model
    if not isinstance(core, Model) or not dataset.scale:
        return
    if not core.memory_mean.numel():
        raise RuntimeError('Prediction requires the training scaler stored with the dual memory')
    raw = dataset.scaler.inverse_transform(dataset.data_x)
    dataset.scaler.mean_ = core.memory_mean.detach().cpu().numpy()
    dataset.scaler.scale_ = core.memory_scale.detach().cpu().numpy()
    dataset.scaler.var_ = dataset.scaler.scale_ ** 2
    dataset.data_x = dataset.scaler.transform(raw)
    dataset.data_y = raw if dataset.inverse else dataset.data_x.copy()


def dual_retrieval_setting_suffix(configs):
    gap = getattr(configs, 'dual_causal_gap', -1)
    return '_dualret_k{}_t{}_m{}_s{}_metric{}_gw{}_ug{}_f{}_r{}_gap{}_hg{}_sr{}'.format(
        getattr(configs, 'dual_top_k', 64),
        getattr(configs, 'dual_temperature', 0.1),
        getattr(configs, 'dual_memory_size', 4096),
        getattr(configs, 'dual_stride', 1),
        getattr(configs, 'dual_search_metric', 'l2'),
        getattr(configs, 'dual_global_weight', 0.5),
        getattr(configs, 'dual_use_global', 1),
        getattr(configs, 'dual_use_future', 1),
        getattr(configs, 'dual_use_residual', 1), gap,
        getattr(configs, 'dual_horizon_gate', 1),
        getattr(configs, 'dual_scale_residual', 1))
