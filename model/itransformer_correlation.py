"""Original iTransformer with a fixed, training-label PCA auxiliary objective.

Forward is inherited unchanged. Only the training loss uses the projector.
Statistics are fitted in two streaming passes: per-variate label mean/std,
then temporal and variate covariance. Memory is O(T*T + N*N + batch*T*N).
"""

import math

import torch
from torch import nn
from torch.utils.data import (
    BatchSampler, DataLoader, IterableDataset, RandomSampler, SequentialSampler,
    Subset, SubsetRandomSampler,
)

from model.iTransformer import Model as OriginalITransformer


class _StreamingMoments:
    """Merge centered float64 batch moments without storing the dataset."""

    def __init__(self, width, diagonal=False):
        self.count = 0
        self.diagonal = diagonal
        self.mean = torch.zeros(width, dtype=torch.float64)
        self.scatter = torch.zeros(width if diagonal else (width, width),
                                   dtype=torch.float64)

    def update(self, rows):
        count = rows.size(0)
        if not count:
            return
        mean = rows.mean(0)
        centered = rows - mean
        scatter = (centered.square().sum(0) if self.diagonal
                   else centered.T @ centered)
        delta = mean - self.mean
        total = self.count + count
        correction = delta.square() if self.diagonal else torch.outer(delta, delta)
        self.scatter += scatter + correction * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def covariance(self):
        if not self.count:
            raise ValueError('Cannot initialize correlation basis from an empty training set')
        return self.scatter / self.count


class JointCorrelationProjector(nn.Module):
    """Fixed truncated PCA bases and eigenvalue-weighted alignment loss.

    Label normalization uses one mean/std per variate over all training
    windows and forecast positions. Temporal covariance treats each (window,
    variate) as one T-vector; variate covariance treats each (window, horizon)
    as one N-vector. Both are centered population covariances. Overlapping
    forecast windows retain their training-sample multiplicity.

    Ratios are strictly between 0 and 1 to keep the objective truncated.
    A singleton axis necessarily retains its only component.
    Variates with training variance <= eps and PCA directions with eigenvalue
    <= eps contribute no auxiliary penalty; forecast MSE still trains them.
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(getattr(configs, 'pred_len', 96))
        self.features = getattr(configs, 'features', 'M')
        self.n_variables = (1 if self.features in ('MS', 'S')
                            else int(getattr(configs, 'enc_in', 7)))
        self.alignment_mode = getattr(configs, 'alignment_mode', 'joint')
        self.temporal_keep_ratio = float(getattr(configs, 'temporal_keep_ratio', 0.5))
        self.variate_keep_ratio = float(getattr(configs, 'variate_keep_ratio', 0.5))
        self.joint_weighting = getattr(configs, 'joint_weighting', 'sqrt_eigen_product')
        self.eps = float(getattr(configs, 'correlation_eps', 1e-6))
        self.standardize_labels = bool(getattr(configs, 'corr_standardize_labels', True))
        if self.pred_len < 1 or self.n_variables < 1:
            raise ValueError('pred_len and number of target variables must be positive')
        if self.alignment_mode not in ('none', 'temporal', 'variate', 'joint'):
            raise ValueError('alignment_mode must be none, temporal, variate or joint')
        for name in ('temporal_keep_ratio', 'variate_keep_ratio'):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(name + ' must be in (0, 1) for truncated PCA')
        if self.joint_weighting != 'sqrt_eigen_product':
            raise ValueError('joint_weighting must be sqrt_eigen_product')
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('correlation_eps must be finite and positive')
        kt = max(1, math.floor(self.pred_len * self.temporal_keep_ratio))
        kn = max(1, math.floor(self.n_variables * self.variate_keep_ratio))
        # Fixed shapes allow strict checkpoint loading before initialization.
        self.register_buffer('temporal_basis', torch.zeros(self.pred_len, kt))
        self.register_buffer('variate_basis', torch.zeros(self.n_variables, kn))
        self.register_buffer('temporal_eigenvalues', torch.zeros(kt))
        self.register_buffer('variate_eigenvalues', torch.zeros(kn))
        self.register_buffer('label_mean', torch.zeros(1, 1, self.n_variables))
        self.register_buffer('label_std', torch.ones(1, 1, self.n_variables))
        self.register_buffer('initialized', torch.tensor(False))

    def _check_shape(self, labels):
        if labels.ndim != 3 or labels.shape[1:] != (self.pred_len, self.n_variables):
            raise ValueError('Correlation labels must be [B, {}, {}], got {}'.format(
                self.pred_len, self.n_variables, tuple(labels.shape)))

    def _training_labels(self, loader):
        for batch in loader:
            labels = batch[1]
            if labels.ndim != 3 or labels.size(1) < self.pred_len:
                raise ValueError('Training batch_y must contain the complete forecast horizon')
            f_dim = -1 if self.features == 'MS' else 0
            labels = labels[:, -self.pred_len:, f_dim:].detach().to(
                device='cpu', dtype=torch.float64)
            self._check_shape(labels)
            if not torch.isfinite(labels).all():
                raise ValueError('Training labels contain NaN or Inf')
            yield labels

    @torch.no_grad()
    def initialize(self, train_loader):
        """Fit once from a map-style training DataLoader, including its tail.

        A dedicated sequential loader covers the same training selection in both
        passes, even if the optimization loader shuffles or drops its last batch.
        SubsetRandomSampler indices are preserved. Other restricted/weighted or
        custom batch samplers must be expressed as an explicit dataset Subset.
        Custom datasets must contain training labels only and be deterministic.
        """
        if self.initialized.item() or self.alignment_mode == 'none':
            return
        if not isinstance(train_loader, DataLoader):
            raise TypeError('Expected a training DataLoader with a map-style dataset')
        dataset = train_loader.dataset
        if isinstance(dataset, IterableDataset):
            raise TypeError('Correlation initialization requires a re-iterable map-style dataset')
        split_dataset = dataset
        while isinstance(split_dataset, Subset):
            split_dataset = split_dataset.dataset
        if getattr(split_dataset, 'set_type', 0) != 0 or getattr(
                split_dataset, 'flag', 'train') != 'train':
            raise ValueError('Correlation basis must be initialized from the training split only')
        if train_loader.batch_size is None or type(train_loader.batch_sampler) is not BatchSampler:
            raise ValueError('Correlation initialization requires the standard batch sampler; '
                             'express the training selection as a dataset Subset')
        sampler = train_loader.sampler
        if type(sampler) is SubsetRandomSampler:
            # Snapshot the selection without drawing random samples or reading
            # labels outside the sampler's indices. Reuse it for both passes.
            dataset = Subset(dataset, list(sampler.indices))
        elif type(sampler) is RandomSampler:
            if sampler.replacement or sampler.num_samples != len(dataset):
                raise ValueError('Correlation initialization does not support replacement or '
                                 'partial random sampling; use a dataset Subset')
        elif type(sampler) is SequentialSampler:
            if len(sampler) != len(dataset):
                raise ValueError('Partial sequential sampling is unsupported; use a dataset Subset')
        else:
            raise ValueError('Unsupported correlation sampler; use a dataset Subset or '
                             'SubsetRandomSampler to specify the training selection')
        loader = DataLoader(dataset, batch_size=train_loader.batch_size or 32,
                            shuffle=False, drop_last=False, num_workers=0,
                            collate_fn=train_loader.collate_fn,
                            generator=torch.Generator().manual_seed(0))
        moments = _StreamingMoments(self.n_variables, diagonal=True)
        for labels in self._training_labels(loader):
            moments.update(labels.reshape(-1, self.n_variables))
        mean = moments.mean.reshape(1, 1, -1)
        std = moments.covariance().clamp_min(self.eps).sqrt().reshape(1, 1, -1)
        active = std > math.sqrt(self.eps)
        safe_std = torch.where(active, std, torch.ones_like(std))
        temporal = _StreamingMoments(self.pred_len)
        variate = _StreamingMoments(self.n_variables)
        for labels in self._training_labels(loader):
            if self.standardize_labels:
                labels = (labels - mean) / safe_std
            labels = labels.masked_fill(~active, 0)
            temporal.update(labels.transpose(1, 2).reshape(-1, self.pred_len))
            variate.update(labels.reshape(-1, self.n_variables))
        for name, stats in (('temporal', temporal), ('variate', variate)):
            covariance = stats.covariance()
            eigenvalues, basis = torch.linalg.eigh((covariance + covariance.T) * 0.5)
            destination = getattr(self, name + '_basis')
            indices = torch.arange(basis.size(1) - 1,
                                   basis.size(1) - destination.size(1) - 1, -1)
            destination.copy_(basis[:, indices])
            getattr(self, name + '_eigenvalues').copy_(eigenvalues[indices].clamp_min(0))
        self.label_mean.copy_(mean)
        self.label_std.copy_(std)
        self.initialized.fill_(True)

    def _require_initialized(self):
        if not self.initialized.item():
            raise RuntimeError('Correlation basis is not initialized; call '
                               'initialize_correlation_basis(model, train_loader) before training')

    def _project(self, values):
        if self.alignment_mode in ('temporal', 'joint'):
            values = torch.matmul(self.temporal_basis.float().T, values)
        if self.alignment_mode in ('variate', 'joint'):
            values = torch.matmul(values, self.variate_basis.float())
        return values

    def _prepare_values(self, values, center=False):
        # Derive the mask from the existing std buffer so older checkpoints
        # retain their state_dict format. Avoid dividing inactive channels by
        # sqrt(eps), which would amplify their errors before masking.
        std = self.label_std.float()
        active = std > math.sqrt(self.eps)
        if self.standardize_labels:
            if center:
                values = values - self.label_mean.float()
            values = values / torch.where(active, std, torch.ones_like(std))
        return values.masked_fill(~active, 0)

    def _eigen_weights(self, eigenvalues):
        eigenvalues = eigenvalues.float()
        return (eigenvalues + self.eps).sqrt().masked_fill(eigenvalues <= self.eps, 0)

    def project(self, labels):
        """Project [B,T,N] labels in the fitted coordinate system (float32)."""
        self._check_shape(labels)
        if self.alignment_mode == 'none':
            return labels
        self._require_initialized()
        with torch.autocast(device_type=labels.device.type, enabled=False):
            values = self._prepare_values(labels.float(), center=True)
            return self._project(values)

    def forward(self, pred, true):
        self._check_shape(pred)
        if pred.shape != true.shape:
            raise ValueError('Correlation pred and true must have identical shapes')
        if self.alignment_mode == 'none':
            return pred.sum() * 0.0
        self._require_initialized()
        # Keep PCA matmuls and the weighted reduction in float32 under AMP.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            error = pred.float() - true.float()
            error = self._prepare_values(error)  # shared label mean cancels
            error = self._project(error)
            wt = self._eigen_weights(self.temporal_eigenvalues)
            wn = self._eigen_weights(self.variate_eigenvalues)
            if self.alignment_mode == 'temporal':
                weights = wt[:, None]
            elif self.alignment_mode == 'variate':
                weights = wn[None, :]
            else:
                weights = torch.outer(wt, wn)
            mean_weight = weights.mean()
            # A zero-rank basis yields a differentiable zero auxiliary loss.
            weights = weights / torch.where(mean_weight > 0, mean_weight,
                                             torch.ones_like(mean_weight))
            return (weights * error.square()).mean()


class Model(OriginalITransformer):
    """Unmodified iTransformer forecast with an optional training objective."""

    def __init__(self, configs):
        super().__init__(configs)
        self.lambda_joint = float(getattr(configs, 'lambda_joint', 0.1))
        if not math.isfinite(self.lambda_joint) or self.lambda_joint < 0:
            raise ValueError('lambda_joint must be finite and nonnegative')
        self.correlation_projector = JointCorrelationProjector(configs)

    def compute_correlation_loss(self, pred, true):
        return self.correlation_projector(pred, true)


def initialize_correlation_basis(model, train_loader):
    """Call after obtaining train_loader, before the first optimization epoch.

    Works with DataParallel. Loaded checkpoints retain their fitted buffers;
    validation, test and prediction need no fitting and no label access.
    """
    model = model.module if isinstance(model, nn.DataParallel) else model
    if model.lambda_joint == 0:
        return
    model.correlation_projector.initialize(train_loader)


def correlation_setting_suffix(configs):
    """Keep ablation checkpoints distinct in training and standalone testing."""
    return '_corr_{}_l{}_t{}_n{}_w{}_eps{}_std{}'.format(
        getattr(configs, 'alignment_mode', 'joint'),
        getattr(configs, 'lambda_joint', 0.1),
        getattr(configs, 'temporal_keep_ratio', 0.5),
        getattr(configs, 'variate_keep_ratio', 0.5),
        getattr(configs, 'joint_weighting', 'sqrt_eigen_product'),
        getattr(configs, 'correlation_eps', 1e-6),
        int(getattr(configs, 'corr_standardize_labels', True)))
