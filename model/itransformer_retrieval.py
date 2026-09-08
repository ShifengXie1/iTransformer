"""VR-iTransformer: variable-wise historical retrieval before self-attention.

Each variable independently retrieves the same variable's historical (X, Y)
pairs, softmax-aggregates their representations, and gates the result into its
current token. Timestamp tokens do not query the memory.

Usage outside the experiment runner::

    model.build_memory(train_data)  # [T, N], training split in model input units
    y = model(x, x_mark, None, None, query_end=starts + model.seq_len)

``query_end`` is the exclusive input end on the memory's time axis. A case is
eligible only if its entire future ends at or before query_end. It is required
during training; evaluation may omit it ONLY for queries after the training
split. Neither decoder inputs nor evaluation labels ever populate the memory.
The raw series and candidate starts are checkpoint buffers; keys are encoded
with current embedding weights, so optimizer updates cannot leave stale keys.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model.iTransformer import Model as OriginalITransformer


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
        if min(self.seq_len, self.pred_len, self.retrieval_top_k,
               self.retrieval_memory_size, self.retrieval_stride,
               self.retrieval_chunk_size, self.retrieval_variable_chunk_size) < 1:
            raise ValueError('Retrieval lengths, sizes, stride and top_k must be positive')
        if not math.isfinite(self.retrieval_temperature) or self.retrieval_temperature <= 0:
            raise ValueError('retrieval_temperature must be finite and positive')

        d_model = configs.d_model
        self.future_embedding = nn.Linear(self.pred_len, d_model)
        self.retrieval_value = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU())
        self.retrieval_gate = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.retrieval_gate.weight)
        nn.init.constant_(self.retrieval_gate.bias, -2.0)
        self.register_buffer('memory_series', torch.empty(0, 0))
        self.register_buffer('memory_starts', torch.empty(0, dtype=torch.long))
        self.register_buffer('memory_mean', torch.empty(0))
        self.register_buffer('memory_scale', torch.empty(0))
        # Evaluation cache is transient; version checks also catch optimizer or
        # load_state_dict updates made while a model remains in eval mode.
        self._key_cache = None
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
        self._key_cache_version = None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def train(self, mode=True):
        if mode:
            self._key_cache = None
            self._key_cache_version = None
        return super().train(mode)

    def _windows(self, starts, variables, include_future=True):
        """Gather [...,L] past and [...,H] future for matching variable indices."""
        length = self.seq_len + (self.pred_len if include_future else 0)
        times = starts.unsqueeze(-1) + torch.arange(length, device=starts.device)
        windows = self.memory_series[times, variables.unsqueeze(-1)]
        past = windows[..., :self.seq_len]
        future = windows[..., self.seq_len:]
        if self.use_norm:
            mean = past.mean(-1, keepdim=True)
            std = (past.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
            past = (past - mean) / std
            future = (future - mean) / std
        return past, future

    def _encode_keys(self, start, stop, first_var, last_var):
        past, _ = self._windows(
            self.memory_starts[start:stop, None],
            torch.arange(first_var, last_var, device=self.memory_series.device)[None, :],
            include_future=False)
        return F.normalize(self.enc_embedding.value_embedding(past).float(), dim=-1)

    @torch.no_grad()
    def _select_neighbors(self, query, query_end, first_var):
        """Chunked cosine Top-K, independently for each [batch, variable]."""
        batch, variables, _ = query.shape
        count = self.memory_starts.numel()
        k = min(self.retrieval_top_k, count)
        best_scores = query.new_empty(batch, variables, 0, dtype=torch.float32)
        best_indices = torch.empty(batch, variables, 0, dtype=torch.long, device=query.device)
        query = F.normalize(query.float(), dim=-1)
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

    def retrieve(self, query_tokens, query_end=None):
        """Return retrieval tokens [B,N,D], indices and weights [B,N,K].

        Invalid neighbors have index -1 and weight zero. The discrete Top-K
        search has no gradient; selected keys, similarity weights and values
        are recomputed with gradients. Pass pre-dropout variable embeddings.
        """
        if not self.memory_starts.numel():
            raise RuntimeError('Retrieval memory is empty; call build_memory(train_series) first')
        if query_tokens.ndim != 3 or query_tokens.size(1) != self.memory_series.size(1):
            raise ValueError('Query variable count/order must match the training memory')
        if query_end is None and self.training:
            raise ValueError('Training retrieval requires query_end to prevent future leakage')
        if query_end is not None:
            query_end = torch.as_tensor(query_end, device=query_tokens.device)
            if query_end.shape != (query_tokens.size(0),) or query_end.dtype not in (torch.int32, torch.int64):
                raise ValueError('query_end must be an integer tensor of shape [B]')

        # Float32 similarities and softmax also handle flat windows under AMP.
        with torch.autocast(device_type=query_tokens.device.type, enabled=False):
            projection = self.enc_embedding.value_embedding
            version = (projection.weight._version, projection.bias._version,
                       self.memory_series._version, self.memory_starts._version,
                       query_tokens.device, projection.weight.dtype)
            if self.training:
                self._key_cache = None
            elif self._key_cache is None or self._key_cache_version != version:
                with torch.no_grad():
                    self._key_cache = torch.cat([
                        self._encode_keys(start, min(start + self.retrieval_chunk_size,
                                                     self.memory_starts.numel()),
                                          0, query_tokens.size(1))
                        for start in range(0, self.memory_starts.numel(), self.retrieval_chunk_size)
                    ], dim=0)
                self._key_cache_version = version
            representations, all_indices, all_weights = [], [], []
            for first in range(0, query_tokens.size(1), self.retrieval_variable_chunk_size):
                last = min(first + self.retrieval_variable_chunk_size, query_tokens.size(1))
                query = query_tokens[:, first:last].float()
                indices, valid = self._select_neighbors(query, query_end, first)
                channels = torch.arange(first, last, device=query.device)[None, :, None]
                past, future = self._windows(self.memory_starts[indices], channels)
                past_token = projection(past)
                scores = (F.normalize(query, dim=-1).unsqueeze(2)
                          * F.normalize(past_token.float(), dim=-1)).sum(-1)
                if self.retrieval_weighted:
                    logits = (scores / self.retrieval_temperature).masked_fill(~valid, -torch.inf)
                    # All-masked queries must yield zero retrieval, not NaN.
                    logits = torch.where(valid.any(-1, keepdim=True), logits, torch.zeros_like(logits))
                    weights = logits.softmax(-1) * valid
                else:
                    weights = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
                future_token = (self.future_embedding(future) if self.retrieval_use_future
                                else torch.zeros_like(past_token))
                values = self.retrieval_value(torch.cat((past_token, future_token), dim=-1))
                representations.append((weights.unsqueeze(-1) * values).sum(2))
                all_indices.append(indices.masked_fill(~valid, -1))
                all_weights.append(weights)
            return (torch.cat(representations, dim=1), torch.cat(all_indices, dim=1),
                    torch.cat(all_weights, dim=1))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, query_end=None):
        if not self.use_retrieval:
            return super().forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
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
        retrieval, _, _ = self.retrieve(tokens[:, :variables], query_end)
        retrieval = retrieval.to(encoded.dtype)
        current = encoded[:, :variables]
        gate = (torch.sigmoid(self.retrieval_gate(torch.cat((current, retrieval), dim=-1)))
                if self.retrieval_use_gate else 1.0)
        enhanced = current + gate * retrieval
        encoded = torch.cat((enhanced, encoded[:, variables:]), dim=1)
        encoded, attention = self.encoder(encoded, attn_mask=None)
        prediction = self.projector(encoded).permute(0, 2, 1)[:, :, :variables]
        if self.use_norm:
            prediction = prediction * stdev + means
        return prediction, attention

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, query_end=None):
        prediction, attention = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, query_end)
        prediction = prediction[:, -self.pred_len:, :]
        return (prediction, attention) if self.output_attention else prediction


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
    if not isinstance(model, Model) or not model.use_retrieval:
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
    if not isinstance(model, Model) or not model.use_retrieval or not dataset.scale:
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
    """Distinguish retrieval hyperparameters and ablations in checkpoints."""
    defaults = [('use_retrieval', 1), ('retrieval_top_k', 8),
                ('retrieval_temperature', 0.1), ('retrieval_memory_size', 1024),
                ('retrieval_stride', 1), ('retrieval_use_future', 1),
                ('retrieval_use_gate', 1), ('retrieval_weighted', 1)]
    return '_vr{}_k{}_t{}_m{}_s{}_f{}_g{}_w{}'.format(
        *(getattr(configs, name, default) for name, default in defaults))
