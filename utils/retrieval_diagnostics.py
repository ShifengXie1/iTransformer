"""Streaming branch errors and retrieval reliability diagnostics (no training state)."""

import csv
import json
from pathlib import Path

import numpy as np
import torch


class RetrievalDiagnostics:
    def __init__(self, feature_start=0, error_scale=None):
        self.feature_start = feature_start
        self.error_scale = error_scale
        self.sums = {}
        self.counts = {}
        self.channel_sums = {}
        self.channel_count = 0
        self.correlation = np.zeros(6, dtype=np.float64)

    @torch.no_grad()
    def update(self, components, target):
        """Accumulate exact element counts; target already has M/MS slicing."""
        if components is None:
            return None
        start = self.feature_start
        target = target.detach().double().cpu()
        available = components['available'][:, start:].detach().cpu()
        row = {}

        def record(name, values):
            values = values.detach().double().cpu()
            total, count = values.sum().item(), values.numel()
            self.sums[name] = self.sums.get(name, 0.) + total
            self.counts[name] = self.counts.get(name, 0) + count
            row[name] = total / count if count else None

        for name, key in [('base', 'base'), ('retrieval', 'retrieval'), ('fused', 'prediction')]:
            prediction = components[key][:, -target.size(1):, start:].detach().double().cpu()
            error = prediction - target
            if self.error_scale is not None:
                error = error * torch.as_tensor(self.error_scale, dtype=error.dtype)
            square, absolute = error.square(), error.abs()
            record(name + '_mse', square)
            record(name + '_mae', absolute)
            for suffix, values in [('mse', square), ('mae', absolute)]:
                key = name + '_' + suffix
                total = values.sum((0, 1)).numpy()
                self.channel_sums[key] = self.channel_sums.get(key, np.zeros_like(total)) + total
            # Horizon quarters expose a good early forecast hiding a poor tail.
            for segment, values in enumerate(torch.tensor_split(square, 4, dim=1), 1):
                record(name + '_mse_q' + str(segment), values)
            if name == 'retrieval':
                mask = available[:, None, :].expand_as(square)
                record('retrieval_available_mse', square[mask])
                record('retrieval_available_mae', absolute[mask])
                per_query_error = square.mean(1)[available].numpy()

        self.channel_count += target.size(0) * target.size(1)
        record('available_fraction', available)
        record('candidate_count', components['candidate_count'][:, start:])
        if 'global_candidate_count' in components:
            record('global_candidate_count', components['global_candidate_count'][:, start:])
            record('global_similarity', components['global_similarity'][:, start:].detach().cpu()[available])
        record('mean_similarity', components['similarity'][:, start:].detach().cpu()[available])
        record('mean_gate', components['gate'][:, :, start:])
        for segment, values in enumerate(torch.tensor_split(components['gate'][:, :, start:], 4, dim=1), 1):
            record('mean_gate_q' + str(segment), values)
        record('future_variance', components['future_variance'][:, :, start:].detach().cpu()[
            available[:, None, :].expand(-1, components['future_variance'].size(1), -1)])
        ratios = components['scale_ratio_mean'][:, start:].detach().double().cpu()[available]
        record('scale_ratio_mean', ratios)
        record('scale_ratio_max_mean', components['scale_ratio_max'][:, start:].detach().cpu()[available])
        r = ratios.numpy()
        self.correlation += [len(r), r.sum(), per_query_error.sum(), (r * r).sum(),
                             (per_query_error * per_query_error).sum(), (r * per_query_error).sum()]
        return row

    def summary(self):
        result = {key: self.sums[key] / count if count else None for key, count in self.counts.items()}
        result['per_channel'] = {key: (value / self.channel_count).tolist()
                                 for key, value in self.channel_sums.items()}
        n, sx, sy, sxx, syy, sxy = self.correlation
        denominator = max(n * sxx - sx * sx, 0.) * max(n * syy - sy * sy, 0.)
        result['scale_ratio_retrieval_error_correlation'] = (
            float(np.clip((n * sxy - sx * sy) / np.sqrt(denominator), -1, 1)) if denominator > 0 else None)
        result['metric_units'] = 'original_units' if self.error_scale is not None else 'dataset_input_units'
        result['retrieval_metric_policy'] = 'base fallback where no history; *_available_* excludes fallback'
        return result


def print_retrieval_summary(label, summary):
    if not summary or 'base_mse' not in summary:
        return
    values = ['{} MSE={:.6f} MAE={:.6f}'.format(name, summary[name + '_mse'], summary[name + '_mae'])
              for name in ('base', 'retrieval', 'fused')]
    print('Retrieval {}: {}'.format(label, ' | '.join(values)))
    if 'fused_mse_q1' in summary:
        def quarters(prefix):
            return ','.join('n/a' if summary[prefix + str(i)] is None else
                            '{:.4f}'.format(summary[prefix + str(i)]) for i in range(1, 5))
        candidates = (' | global candidates={:.1f}'.format(summary['global_candidate_count'])
                      if 'global_candidate_count' in summary else '')
        print('Retrieval {} horizon quarters: fused MSE=[{}] | gate=[{}]{}'.format(
            label, quarters('fused_mse_q'), quarters('mean_gate_q'), candidates))


def save_retrieval_diagnostics(directory, summaries, rows=None):
    """Save server-readable metrics and batch curves without storing predictions."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'retrieval_diagnostics.json').write_text(
        json.dumps(summaries, indent=2, allow_nan=False), encoding='utf-8')
    if not rows:
        return
    with (directory / 'retrieval_batches.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for axis, key in zip(axes, ('candidate_count', 'mean_similarity', 'mean_gate')):
        axis.plot([row[key] for row in rows], linewidth=0.6, alpha=0.8)
        axis.set_ylabel(key)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel('Training batch (across epochs)')
    figure.tight_layout()
    figure.savefig(directory / 'retrieval_training_curves.png', dpi=140)
    plt.close(figure)
