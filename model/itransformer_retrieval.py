"""Global-context-constrained variable retrieval and consensus-aware fusion.

Default retrieval first pools all contextual variable tokens and projects the
pooled state with W_r. Causal global Top-G selects one shared candidate set per
query. Each variable then refines Top-K using its OWN past embedding inside
that set, preserving local trajectory detail instead of replacing it with
mixed encoder tokens. Selected weights use the mean of local/global cosines,
so global compatibility remains a differentiable prior after hard selection.

Query and memory contextual encodings see all data variables, omit timestamp
tokens, and disable dropout in this retrieval path only. Detached encoder
contexts train W_r through the selected weights; the backbone still learns
through the forecasting path. Forecasting retains its covariates and dropout.
The original iTransformer produces Y_base;
historical futures are adapted to the current past before producing Y_ret:

    Y_ret = sum_k w_k * [Y_k + A(X_query - X_k)]

A is a shared, bias-free temporal linear map initialized to zero. It learns
how differences between past windows change their continuations, instead of
assuming similar past embeddings imply interchangeable futures. The same
adapted neighbors supply disagreement statistics. The default gate shares its
parameters across horizons and penalizes each step's uncertainty U and
base/retrieval disagreement D with nonnegative learned coefficients:

    g_h = sigmoid(a(z, similarity) - softplus(b_U)*log1p(U_h)
                                  - softplus(b_D)*log1p(D_h))

    Y_pred = Y_base + g * (Y_ret - Y_base)

Holding context/similarity fixed, increasing U_h or D_h cannot increase g_h
or change another horizon's gate. Detached gate evidence prevents the base
or adapter from changing predictions solely to manipulate confidence.
Consensus is an agreement signal, not a guarantee that retrieved futures
are accurate. Both new mechanisms can be disabled to reproduce ctx1.

Training uses MSE(Y_pred, Y) + retrieval_base_loss_weight * MSE(Y_base, Y).
W_r learns through this forecast loss; a separate future-similarity objective
is not part of this implementation.
forward(..., return_components=True) returns differentiable branch forecasts
and detached diagnostics through the normal DataParallel output path.

Historical futures never enter the embedding or self-attention. With use_norm,
each historical future uses its own past's mean/std and is transferred to the
current query's scale by the final de-normalization. No eligible history means
exactly Y_base. Timestamp tokens do not query the memory.

Usage outside the experiment runner::

    model.build_memory(train_data)  # [T, N], training split in model input units
    y = model(x, x_mark, None, None, query_end=starts + model.seq_len)

``query_end`` is the exclusive input end on the memory's time axis. A case is
eligible only if its entire future ends at or before query_end. It is required
during training; evaluation may omit it ONLY for queries after the training
split. Neither decoder inputs nor evaluation labels ever populate the memory.
The raw series and candidate starts are checkpoint buffers; keys are encoded
with current embedding, encoder and retrieval projection weights, so optimizer
updates cannot leave stale keys.
"""

import math
from contextlib import contextmanager
from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model.iTransformer import Model as OriginalITransformer


class RetrievedFuture(NamedTuple):
    future: torch.Tensor      # [B,N,H], in normalized past units if use_norm
    indices: torch.Tensor     # [B,N,K], positions in memory_starts; -1 is invalid
    weights: torch.Tensor     # [B,N,K], zero for invalid neighbors
    similarity: torch.Tensor  # [B,N,1], weighted cosine similarity
    similarity_stats: Optional[torch.Tensor] = None  # [B,N,4]: mean,max,std,gap
    future_variance: Optional[torch.Tensor] = None   # [B,N,H], valid-K population variance
    past_std: Optional[torch.Tensor] = None          # [B,N,K], historical input std
    global_candidate_count: Optional[torch.Tensor] = None  # [B], after causal/global filtering
    global_similarity: Optional[torch.Tensor] = None       # [B,N,1], selected weighted global cosine


class Model(OriginalITransformer):
    def __init__(self, configs):
        super().__init__(configs)
        self.use_retrieval = bool(getattr(configs, 'use_retrieval', True))
        self.retrieval_top_k = int(getattr(configs, 'retrieval_top_k', 8))
        self.retrieval_temperature = float(getattr(configs, 'retrieval_temperature', 0.1))
        self.retrieval_memory_size = int(getattr(configs, 'retrieval_memory_size', 1024))
        self.retrieval_stride = int(getattr(configs, 'retrieval_stride', 1))
        self.retrieval_chunk_size = int(getattr(configs, 'retrieval_chunk_size', 128))
        self.retrieval_variable_chunk_size = int(getattr(configs, 'retrieval_variable_chunk_size', 32))
        self.retrieval_use_future = bool(getattr(configs, 'retrieval_use_future', True))
        self.retrieval_use_gate = bool(getattr(configs, 'retrieval_use_gate', True))
        self.retrieval_weighted = bool(getattr(configs, 'retrieval_weighted', True))
        self.retrieval_reliability_gate = bool(getattr(configs, 'retrieval_reliability_gate', True))
        self.retrieval_base_loss_weight = float(getattr(configs, 'retrieval_base_loss_weight', 0.2))
        self.retrieval_contextual = bool(getattr(configs, 'retrieval_contextual', True))
        self.retrieval_global_filter = bool(getattr(configs, 'retrieval_global_filter', True))
        self.retrieval_global_top_k = int(getattr(configs, 'retrieval_global_top_k', 64))
        self.retrieval_consensus_gate = bool(getattr(configs, 'retrieval_consensus_gate', True))
        if min(self.seq_len, self.pred_len, self.retrieval_top_k,
               self.retrieval_memory_size, self.retrieval_stride,
               self.retrieval_chunk_size, self.retrieval_variable_chunk_size,
               self.retrieval_global_top_k) < 1:
            raise ValueError('Retrieval lengths, sizes, stride and top_k must be positive')
        if not math.isfinite(self.retrieval_temperature) or self.retrieval_temperature <= 0:
            raise ValueError('retrieval_temperature must be finite and positive')
        if not math.isfinite(self.retrieval_base_loss_weight) or self.retrieval_base_loss_weight < 0:
            raise ValueError('retrieval_base_loss_weight must be finite and nonnegative')

        d_model = configs.d_model
        # Learn reliability in prediction space: one gate per variable/horizon.
        # Nonzero final weights let both MLP layers learn from the first step.
        consensus = self.retrieval_consensus_gate and self.retrieval_reliability_gate
        gate_inputs = d_model + (4 if consensus else
                                 4 + 2 * self.pred_len if self.retrieval_reliability_gate else 1)
        self.prediction_gate = nn.Sequential(
            nn.Linear(gate_inputs, d_model), nn.GELU(),
            nn.Linear(d_model, 1 if consensus else self.pred_len),
        )
        nn.init.normal_(self.prediction_gate[-1].weight, std=0.02)
        nn.init.constant_(self.prediction_gate[-1].bias, -2.0)
        if consensus:
            # Positive penalties enforce decreasing trust as disagreement grows.
            self.consensus_penalty = nn.Parameter(torch.zeros(2))
        # No bias: an identical past must preserve its observed continuation.
        # Zero initialization starts with the existing retrieval forecast.
        self.continuation_adapter = nn.Linear(self.seq_len, self.pred_len, bias=False)
        nn.init.zeros_(self.continuation_adapter.weight)
        self.retrieval_projection = (nn.Linear(d_model, d_model, bias=False)
                                     if self.retrieval_contextual or self.retrieval_global_filter else nn.Identity())
        if self.retrieval_contextual or self.retrieval_global_filter:
            nn.init.eye_(self.retrieval_projection.weight)
        self.register_buffer('memory_series', torch.empty(0, 0))
        self.register_buffer('memory_starts', torch.empty(0, dtype=torch.long))
        self.register_buffer('memory_mean', torch.empty(0))
        self.register_buffer('memory_scale', torch.empty(0))
        # Evaluation cache is transient; version checks also catch optimizer or
        # load_state_dict updates made while a model remains in eval mode.
        self._key_cache = None
        self._context_cache = None
        self._global_key_cache = None
        self._key_cache_version = None

    @torch.no_grad()
    def build_memory(self, train_series):
        """Build deterministic candidates from an ordered training-only [T,N].

        Subsample evenly after applying stride if the bank exceeds its cap.
        Store the series once instead of duplicating overlapping X/Y windows.
        Channel ordering and preprocessing must match the forecasting inputs.
        """
        series = torch.as_tensor(train_series, device=self.projector.weight.device,
                                 dtype=self.projector.weight.dtype)
        if series.ndim != 2 or series.size(1) < 1:
            raise ValueError('train_series must be [T,N] with N positive')
        if series.size(0) < self.seq_len + self.pred_len:
            raise ValueError('Training memory needs at least seq_len + pred_len observations')
        if not torch.isfinite(series).all():
            raise ValueError('Training memory contains NaN or infinity')
        starts = torch.arange(0, series.size(0) - self.seq_len - self.pred_len + 1,
                              self.retrieval_stride, device=series.device)
        if starts.numel() > self.retrieval_memory_size:
            selected = torch.linspace(0, starts.numel() - 1, self.retrieval_memory_size,
                                      device=series.device).round().long()
            starts = starts[selected]
        self.memory_series = series.detach().clone()
        self.memory_starts = starts
        self.memory_mean = series.new_empty(0)
        self.memory_scale = series.new_empty(0)
        self._key_cache = None
        self._context_cache = None
        self._global_key_cache = None
        self._key_cache_version = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # A fresh instance has an empty bank. Resize before normal strict loading,
        # including when this module is nested in DataParallel.
        for name in ('memory_series', 'memory_starts', 'memory_mean', 'memory_scale'):
            value = state_dict.get(prefix + name)
            if value is not None:
                current = getattr(self, name)
                setattr(self, name, current.new_empty(value.shape))
        self._key_cache = None
        self._context_cache = None
        self._global_key_cache = None
        self._key_cache_version = None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def train(self, mode=True):
        if mode:
            self._key_cache = None
            self._context_cache = None
            self._global_key_cache = None
            self._key_cache_version = None
        return super().train(mode)

    def _windows(self, starts, variables, include_future=True, return_std=False):
        """Gather [...,L] past and [...,H] future for matching variable indices."""
        length = self.seq_len + (self.pred_len if include_future else 0)
        times = starts.unsqueeze(-1) + torch.arange(length, device=starts.device)
        windows = self.memory_series[times, variables.unsqueeze(-1)]
        past = windows[..., :self.seq_len]
        future = windows[..., self.seq_len:]
        if self.use_norm or return_std:
            mean = past.mean(-1, keepdim=True)
            std = (past.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        if self.use_norm:
            past = (past - mean) / std
            future = (future - mean) / std
        return (past, future, std.squeeze(-1)) if return_std else (past, future)

    @contextmanager
    def _retrieval_dropout_off(self):
        # Restore individual flags even after an exception. This affects only
        # retrieval passes, not the subsequent forecasting pass or gradients.
        modules = [m for m in self.encoder.modules() if isinstance(m, nn.Dropout)]
        flags = [m.training for m in modules]
        try:
            for module in modules:
                module.training = False
            yield
        finally:
            for module, flag in zip(modules, flags):
                module.training = flag

    def encode_retrieval_context(self, normalized_past):
        """Deterministic [B,N,L] -> [B,N,D] using all current data variables.

        Inputs use per-window normalization when use_norm is enabled. History
        and queries intentionally omit time tokens: memory stores only data_x,
        so adding query-only time tokens would put them in different spaces.
        """
        with torch.autocast(device_type=normalized_past.device.type, enabled=False):
            with self._retrieval_dropout_off():
                tokens = self.enc_embedding.value_embedding(normalized_past.float())
                return self.encoder(tokens, attn_mask=None)[0]

    def _history_context(self, start, stop):
        # Gather the entire multivariate window before extracting variable n.
        # Encoding each selected variable alone would discard its context.
        past, _ = self._windows(
            self.memory_starts[start:stop, None],
            torch.arange(self.memory_series.size(1), device=self.memory_series.device)[None, :],
            include_future=False)
        return self.encode_retrieval_context(past)

    def _encode_keys(self, start, stop, first_var, last_var):
        if self.retrieval_contextual and not self.retrieval_global_filter:
            context = self._history_context(start, stop)
            return F.normalize(self.retrieval_projection(context[:, first_var:last_var]).float(), dim=-1)
        past, _ = self._windows(
            self.memory_starts[start:stop, None],
            torch.arange(first_var, last_var, device=self.memory_series.device)[None, :],
            include_future=False)
        return F.normalize(self.enc_embedding.value_embedding(past).float(), dim=-1)

    @torch.no_grad()
    def _select_global_candidates(self, query_context, query_end):
        """Causal masking precedes global Top-G, shared by all variables."""
        scores = F.normalize(query_context.float(), dim=-1) @ self._global_key_cache.T
        if query_end is not None:
            ends = self.memory_starts + self.seq_len + self.pred_len
            scores = scores.masked_fill(ends[None, :] > query_end[:, None], -torch.inf)
        scores, indices = scores.topk(min(self.retrieval_global_top_k, scores.size(-1)), dim=-1)
        return indices, torch.isfinite(scores)

    @torch.no_grad()
    def _select_neighbors(self, query, query_end, first_var, candidates=None):
        """Chunked cosine Top-K, independently for each [batch, variable]."""
        batch, variables, _ = query.shape
        count = self.memory_starts.numel()
        k = min(self.retrieval_top_k, count)
        best_scores = query.new_empty(batch, variables, 0, dtype=torch.float32)
        best_indices = torch.empty(batch, variables, 0, dtype=torch.long, device=query.device)
        query = F.normalize(query.float(), dim=-1)
        if candidates is not None:
            indices, valid = candidates
            keys = self._key_cache[indices, first_var:first_var + variables]
            scores = torch.einsum('bnd,bgnd->bng', query, keys)
            scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
            scores, positions = scores.topk(min(k, indices.size(-1)), dim=-1)
            indices = indices[:, None, :].expand(-1, variables, -1).gather(-1, positions)
            return indices, torch.isfinite(scores)
        for start in range(0, count, self.retrieval_chunk_size):
            stop = min(start + self.retrieval_chunk_size, count)
            if self._key_cache is None:
                keys = self._encode_keys(start, stop, first_var, first_var + variables)
            else:
                keys = self._key_cache[start:stop, first_var:first_var + variables]
            scores = torch.einsum('bnd,mnd->bnm', query, keys)
            if query_end is not None:
                ends = self.memory_starts[start:stop] + self.seq_len + self.pred_len
                scores = scores.masked_fill(ends[None, None, :] > query_end[:, None, None], -torch.inf)
            indices = torch.arange(start, stop, device=query.device).expand(batch, variables, -1)
            scores = torch.cat((best_scores, scores), dim=-1)
            indices = torch.cat((best_indices, indices), dim=-1)
            best_scores, positions = scores.topk(min(k, scores.size(-1)), dim=-1)
            best_indices = indices.gather(-1, positions)
        return best_indices, torch.isfinite(best_scores)

    def _adapt_futures(self, query_past, past, future):
        """Transfer [B,N,K,H] continuations using past differences only.

        All inputs share the window-normalized coordinates used by retrieval
        (or model input units when normalization is off). The map is shared
        across variables/neighbors and never receives current future labels.
        """
        return future + self.continuation_adapter(query_past.unsqueeze(2) - past)

    def retrieve(self, query_tokens, query_end=None, query_past=None, query_context=None):
        """Return futures, neighbors and reliability statistics (see RetrievedFuture).

        Invalid neighbors have index -1 and weight zero. The discrete Top-K
        search has no gradient; selected keys and similarity weights are
        recomputed with gradients. Global filtering takes local past tokens
        plus W_r-projected pooled query_context [B,D]. Without global filtering,
        contextual mode takes W_r-projected
        encode_retrieval_context outputs; otherwise pass past embeddings. Pass
        normalized query_past [B,N,L] to adapt historical continuations.
        Omitting query_past exposes the raw historical retrieval for inspection.
        """
        if not self.memory_starts.numel():
            raise RuntimeError('Retrieval memory is empty; call build_memory(train_series) first')
        if query_tokens.ndim != 3 or query_tokens.size(1) != self.memory_series.size(1):
            raise ValueError('Query variable count/order must match the training memory')
        if query_past is not None and query_past.shape != (*query_tokens.shape[:2], self.seq_len):
            raise ValueError('query_past must have shape [B,N,seq_len]')
        if self.retrieval_global_filter and (query_context is None or
                                             query_context.shape != (query_tokens.size(0), query_tokens.size(-1))):
            raise ValueError('Global filtering requires projected query_context [B,D]')
        if query_end is None and self.training:
            raise ValueError('Training retrieval requires query_end to prevent future leakage')
        if query_end is not None:
            query_end = torch.as_tensor(query_end, device=query_tokens.device)
            if query_end.shape != (query_tokens.size(0),) or query_end.dtype not in (torch.int32, torch.int64):
                raise ValueError('query_end must be an integer tensor of shape [B]')

        # Float32 similarities and softmax also handle flat windows under AMP.
        with torch.autocast(device_type=query_tokens.device.type, enabled=False):
            projection = self.enc_embedding.value_embedding
            key_parameters = list(projection.parameters())
            if self.retrieval_contextual or self.retrieval_global_filter:
                key_parameters += list(self.encoder.parameters()) + list(self.retrieval_projection.parameters())
            version = (tuple((id(p), p._version) for p in key_parameters),
                       self.memory_series._version, self.memory_starts._version,
                       query_tokens.device, projection.weight.dtype, self.use_norm,
                       self.retrieval_contextual, self.retrieval_global_filter)
            if self.training or self._key_cache is None or self._key_cache_version != version:
                with torch.no_grad():
                    chunks = range(0, self.memory_starts.numel(), self.retrieval_chunk_size)
                    if self.retrieval_global_filter:
                        contexts, local_keys = [], []
                        for start in chunks:
                            stop = min(start + self.retrieval_chunk_size, self.memory_starts.numel())
                            past, _ = self._windows(self.memory_starts[start:stop, None],
                                                   torch.arange(query_tokens.size(1), device=query_tokens.device)[None],
                                                   include_future=False)
                            local = projection(past)
                            context = self.encode_retrieval_context(past) if self.retrieval_contextual else local
                            contexts.append(context.mean(1))
                            local_keys.append(F.normalize(local.float(), dim=-1))
                        self._context_cache = torch.cat(contexts, dim=0)
                        self._global_key_cache = F.normalize(self.retrieval_projection(self._context_cache).float(), dim=-1)
                        self._key_cache = torch.cat(local_keys, dim=0)
                    elif self.retrieval_contextual:
                        self._context_cache = torch.cat([
                            self._history_context(start, min(start + self.retrieval_chunk_size,
                                                             self.memory_starts.numel()))
                            for start in chunks], dim=0)
                        self._key_cache = F.normalize(self.retrieval_projection(self._context_cache).float(), dim=-1)
                    else:
                        self._context_cache = None
                        self._key_cache = torch.cat([
                            self._encode_keys(start, min(start + self.retrieval_chunk_size,
                                                         self.memory_starts.numel()),
                                              0, query_tokens.size(1))
                            for start in chunks], dim=0)
                self._key_cache_version = version
            candidates = (self._select_global_candidates(query_context, query_end)
                          if self.retrieval_global_filter else None)
            futures, all_indices, all_weights, similarities = [], [], [], []
            statistics, variances, past_stds = [], [], []
            global_similarities = []
            for first in range(0, query_tokens.size(1), self.retrieval_variable_chunk_size):
                last = min(first + self.retrieval_variable_chunk_size, query_tokens.size(1))
                query = query_tokens[:, first:last].float()
                indices, valid = self._select_neighbors(query, query_end, first, candidates)
                channels = torch.arange(first, last, device=query.device)[None, :, None]
                past, future, past_std = self._windows(self.memory_starts[indices], channels, return_std=True)
                if query_past is not None:
                    future = self._adapt_futures(query_past[:, first:last].float(), past, future)
                if self.retrieval_global_filter:
                    # Local matching preserves the variable's own trajectory.
                    past_token = projection(past).detach()
                elif self.retrieval_contextual:
                    # Stop-gradient memory contexts avoid retaining the whole
                    # bank's encoder graph. Selected W_r keys remain trainable.
                    past_token = self.retrieval_projection(self._context_cache[indices, channels])
                else:
                    past_token = projection(past)
                scores = (F.normalize(query, dim=-1).unsqueeze(2)
                          * F.normalize(past_token.float(), dim=-1)).sum(-1)
                if self.retrieval_global_filter:
                    global_keys = self.retrieval_projection(self._context_cache[indices])
                    global_scores = (F.normalize(query_context.float(), dim=-1)[:, None, None, :]
                                     * F.normalize(global_keys.float(), dim=-1)).sum(-1)
                    # Global compatibility remains a differentiable prior after
                    # local Top-K selection. The average retains cosine scale.
                    scores = 0.5 * (scores + global_scores)
                if self.retrieval_weighted:
                    logits = (scores / self.retrieval_temperature).masked_fill(~valid, -torch.inf)
                    # All-masked queries must yield zero retrieval, not NaN.
                    logits = torch.where(valid.any(-1, keepdim=True), logits, torch.zeros_like(logits))
                    weights = logits.softmax(-1) * valid
                else:
                    weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
                if self.retrieval_global_filter:
                    global_similarities.append((weights * global_scores).sum(-1, keepdim=True))
                futures.append((weights.unsqueeze(-1) * future).sum(2))
                similarities.append((weights * scores).sum(-1, keepdim=True))
                # Unweighted statistics over VALID neighbors: even a low-weight
                # conflicting future should be visible to the reliability gate.
                count = valid.sum(-1, keepdim=True).clamp_min(1)
                has_history = valid.any(-1, keepdim=True)
                mean_score = (scores * valid).sum(-1, keepdim=True) / count
                score_var = ((scores - mean_score).square() * valid).sum(-1, keepdim=True) / count
                score_std = torch.where(score_var > 0, score_var.clamp_min(1e-12).sqrt(), 0.)
                maximum = torch.where(has_history, scores.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True), 0.)
                minimum = torch.where(has_history, scores.masked_fill(~valid, torch.inf).amin(-1, keepdim=True), 0.)
                statistics.append(torch.cat((mean_score, maximum, score_std, maximum - minimum), -1))
                future_mean = (future * valid.unsqueeze(-1)).sum(2) / count
                variances.append(((future - future_mean.unsqueeze(2)).square()
                                  * valid.unsqueeze(-1)).sum(2) / count)
                past_stds.append(past_std)
                all_indices.append(indices.masked_fill(~valid, -1))
                all_weights.append(weights)
            return RetrievedFuture(torch.cat(futures, dim=1), torch.cat(all_indices, dim=1),
                                   torch.cat(all_weights, dim=1), torch.cat(similarities, dim=1),
                                   torch.cat(statistics, dim=1), torch.cat(variances, dim=1),
                                   torch.cat(past_stds, dim=1),
                                   candidates[1].sum(-1) if candidates is not None else None,
                                   torch.cat(global_similarities, dim=1) if global_similarities else None)

    def fuse_prediction(self, base_prediction, current_tokens, retrieved, return_gate=False):
        """Fuse [B,N,H] forecasts in the same units, with [B,N,D] past tokens.

        Delta is Y_ret - Y_base, so gates near 0 recover the backbone and
        gates near 1 trust the retrieved forecast. Disabling the gate uses
        Y_ret wherever history is available. Unavailable history always
        retains the backbone, including when the gate network has biases.
        """
        if self.retrieval_use_gate:
            if self.retrieval_consensus_gate and self.retrieval_reliability_gate:
                if retrieved.similarity_stats is None or retrieved.future_variance is None:
                    raise ValueError('Consensus gate requires similarity_stats and future_variance')
                context = torch.cat((current_tokens.detach(),
                                     retrieved.similarity_stats.detach().to(current_tokens.dtype)), dim=-1)
                confidence = self.prediction_gate(context).float()
                uncertainty = torch.log1p(retrieved.future_variance.detach().float().clamp_min(0))
                disagreement = torch.log1p((retrieved.future.detach().float() - base_prediction.detach().float()).abs())
                penalties = F.softplus(self.consensus_penalty.float())
                gate = torch.sigmoid(confidence - penalties[0] * uncertainty - penalties[1] * disagreement)
            elif self.retrieval_reliability_gate:
                if retrieved.similarity_stats is None or retrieved.future_variance is None:
                    raise ValueError('Reliability gate requires similarity_stats and future_variance')
                # These features use the same normalized units as both forecasts.
                # log1p limits dynamic range without imposing a clipping policy.
                reliability = torch.cat((retrieved.similarity_stats,
                                         torch.log1p(retrieved.future_variance),
                                         torch.log1p((retrieved.future - base_prediction).abs())), dim=-1)
            else:
                reliability = retrieved.similarity
            if not (self.retrieval_consensus_gate and self.retrieval_reliability_gate):
                context = torch.cat((current_tokens, reliability.to(current_tokens.dtype)), dim=-1)
                gate = torch.sigmoid(self.prediction_gate(context))
        else:
            gate = torch.ones_like(base_prediction)
        available = (retrieved.indices >= 0).any(-1, keepdim=True)
        gate = torch.where(available, gate, 0.)
        prediction = base_prediction + gate * (retrieved.future - base_prediction)
        prediction = torch.where(available, prediction, base_prediction)
        return (prediction, gate) if return_gate else prediction

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, query_end=None, return_components=False):
        if not self.use_retrieval or not self.retrieval_use_future:
            prediction, attention = super().forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if not return_components:
                return prediction, attention
            bn = prediction[:, 0, :]
            zeros = torch.zeros_like(bn)
            components = dict(prediction=prediction, base=prediction, retrieval=prediction,
                              gate=torch.zeros_like(prediction), available=zeros.bool(),
                              candidate_count=zeros, similarity=zeros, scale_ratio_mean=zeros,
                              scale_ratio_max=zeros, future_variance=torch.zeros_like(prediction))
            return prediction, attention, components
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len:
            raise ValueError('x_enc must have shape [B,seq_len,N]')
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            centered = x_enc - means
            stdev = (centered.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
            normalized = centered / stdev
        else:
            normalized = x_enc
        variables = x_enc.size(2)
        inputs = normalized.permute(0, 2, 1)
        if x_mark_enc is not None:
            inputs = torch.cat((inputs, x_mark_enc.permute(0, 2, 1)), dim=1)
        # Match the original embedding's dropout, but use deterministic keys
        # and queries for selecting historical cases during training.
        tokens = self.enc_embedding.value_embedding(inputs)
        encoded = self.enc_embedding.dropout(tokens)
        query_context = None
        if self.retrieval_global_filter:
            with torch.autocast(device_type=x_enc.device.type, enabled=False):
                with torch.no_grad():
                    query_tokens = self.enc_embedding.value_embedding(normalized.permute(0, 2, 1).float())
                    retrieval_context = (self.encode_retrieval_context(normalized.permute(0, 2, 1))
                                         if self.retrieval_contextual else query_tokens)
                query_context = self.retrieval_projection(retrieval_context.mean(1))
        elif self.retrieval_contextual:
            # Keep metric-learning gradients in W_r. Forecast loss still trains
            # the encoder normally through base_prediction below.
            with torch.no_grad():
                retrieval_context = self.encode_retrieval_context(normalized.permute(0, 2, 1))
            with torch.autocast(device_type=x_enc.device.type, enabled=False):
                query_tokens = self.retrieval_projection(retrieval_context)
        else:
            query_tokens = tokens[:, :variables]
        retrieved = self.retrieve(query_tokens, query_end,
                                  query_past=normalized.permute(0, 2, 1), query_context=query_context)
        # Forecast attention sees only the current window and its covariates.
        encoded, attention = self.encoder(encoded, attn_mask=None)
        base_prediction = self.projector(encoded)[:, :variables]
        prediction, gate = self.fuse_prediction(base_prediction, tokens[:, :variables], retrieved, return_gate=True)
        prediction = prediction.permute(0, 2, 1)
        if self.use_norm:
            prediction = prediction * stdev + means
        if return_components:
            base = base_prediction.permute(0, 2, 1)
            ret = retrieved.future.permute(0, 2, 1)
            if self.use_norm:
                base, ret = base * stdev + means, ret * stdev + means
            available = (retrieved.indices >= 0).any(-1)
            ret = torch.where(available[:, None, :], ret, base)
            with torch.no_grad():
                query_std = (x_enc.var(1, unbiased=False) + 1e-5).sqrt()
                ratio = query_std.unsqueeze(-1) / retrieved.past_std
                valid = retrieved.indices >= 0
                ratio_mean = (ratio * retrieved.weights).sum(-1)
                ratio_max = ratio.masked_fill(~valid, 0.).amax(-1)
                ends = self.memory_starts + self.seq_len + self.pred_len
                if query_end is None:
                    counts = ends.new_full((x_enc.size(0),), ends.numel())
                else:
                    query_ends = torch.as_tensor(query_end, device=ends.device)
                    counts = torch.searchsorted(ends, query_ends, right=True)
                counts = counts[:, None].expand(-1, variables)
            components = dict(prediction=prediction, base=base, retrieval=ret,
                              gate=gate.detach().permute(0, 2, 1), available=available,
                              candidate_count=counts, similarity=retrieved.similarity.detach().squeeze(-1),
                              scale_ratio_mean=ratio_mean, scale_ratio_max=ratio_max,
                              future_variance=retrieved.future_variance.detach().permute(0, 2, 1))
            if retrieved.global_candidate_count is not None:
                components['global_candidate_count'] = retrieved.global_candidate_count[:, None].expand(-1, variables)
                components['global_similarity'] = retrieved.global_similarity.detach().squeeze(-1)
            return prediction, attention, components
        return prediction, attention

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, query_end=None, return_components=False):
        if return_components:
            _, _, components = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, query_end, True)
            return components
        prediction, attention = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, query_end)
        prediction = prediction[:, -self.pred_len:, :]
        return (prediction, attention) if self.output_attention else prediction

    def base_auxiliary_loss(self, components, target):
        """Use gathered, non-detached forecasts; supports M, MS and DataParallel."""
        if not self.use_retrieval or not self.retrieval_use_future or self.retrieval_base_loss_weight == 0:
            return target.new_zeros(())
        base = components['base'][:, -target.size(1):, -target.size(2):]
        return self.retrieval_base_loss_weight * F.mse_loss(base.float(), target.float())


class _IndexedTrainingDataset(Dataset):
    """Keep sample positions attached through shuffling and worker processes."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (*self.dataset[index], index + self.dataset.seq_len)


def initialize_retrieval_memory(model, dataset, loader):
    """Build from train data_x only; preserve the loader's sampling policy."""
    model = model.module if isinstance(model, nn.DataParallel) else model
    if not isinstance(model, Model) or not model.use_retrieval or not model.retrieval_use_future:
        return loader
    model.build_memory(dataset.data_x)
    if dataset.scale:
        model.memory_mean = torch.as_tensor(dataset.scaler.mean_, device=model.memory_series.device,
                                           dtype=model.memory_series.dtype).clone()
        model.memory_scale = torch.as_tensor(dataset.scaler.scale_, device=model.memory_series.device,
                                            dtype=model.memory_series.dtype).clone()
    return DataLoader(_IndexedTrainingDataset(dataset), batch_sampler=loader.batch_sampler,
                      num_workers=loader.num_workers, collate_fn=loader.collate_fn,
                      pin_memory=loader.pin_memory, worker_init_fn=loader.worker_init_fn,
                      generator=loader.generator)


def align_retrieval_prediction_data(model, dataset):
    """Dataset_Pred fits its own scaler; use the memory's training units instead."""
    model = model.module if isinstance(model, nn.DataParallel) else model
    if (not isinstance(model, Model) or not model.use_retrieval
            or not model.retrieval_use_future or not dataset.scale):
        return
    if not model.memory_mean.numel():
        raise RuntimeError('Prediction requires the training scaler stored with the memory')
    raw = dataset.scaler.inverse_transform(dataset.data_x)
    dataset.scaler.mean_ = model.memory_mean.detach().cpu().numpy()
    dataset.scaler.scale_ = model.memory_scale.detach().cpu().numpy()
    dataset.scaler.var_ = dataset.scaler.scale_ ** 2
    dataset.data_x = dataset.scaler.transform(raw)
    dataset.data_y = raw if dataset.inverse else dataset.data_x.copy()


def retrieval_setting_suffix(configs):
    """Separate contextual checkpoints while retaining the previous ablation path."""
    defaults = [('use_retrieval', 1), ('retrieval_top_k', 8),
                ('retrieval_temperature', 0.1), ('retrieval_memory_size', 1024),
                ('retrieval_stride', 1), ('retrieval_use_future', 1),
                ('retrieval_use_gate', 1), ('retrieval_weighted', 1),
                ('retrieval_reliability_gate', 1), ('retrieval_base_loss_weight', 0.2)]
    suffix = '_vradapt{}_k{}_t{}_m{}_s{}_f{}_g{}_w{}_rg{}_bl{}'.format(
        *(getattr(configs, name, default) for name, default in defaults))
    if getattr(configs, 'retrieval_contextual', True):
        suffix += '_ctx1'
    if getattr(configs, 'retrieval_global_filter', True):
        suffix += '_gc{}'.format(getattr(configs, 'retrieval_global_top_k', 64))
    if getattr(configs, 'retrieval_consensus_gate', True):
        suffix += '_cg1'
    return suffix
