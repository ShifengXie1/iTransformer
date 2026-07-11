import json
import os
from collections import Counter

import torch


def adjust_long_periods(periods, seq_len):
    """Replace periods longer than half the input with the reliable mode."""
    periods = [int(period) for period in periods]
    reliable = [period for period in periods if period <= (seq_len / 2)]
    if not reliable:
        return periods

    counts = Counter(reliable)
    max_count = max(counts.values())
    replacement = min(
        period for period, count in counts.items() if count == max_count
    )
    return [
        replacement if period > (seq_len / 2) else period
        for period in periods
    ]


@torch.no_grad()
def estimate_channel_periods(train_loader, seq_len, max_batches=0):
    """Estimate one stable dominant period per variable from train windows."""
    amplitude_sum = None
    sample_count = 0

    for batch_idx, (batch_x, _, _, _) in enumerate(train_loader):
        if max_batches and batch_idx >= max_batches:
            break
        x = batch_x.float()
        if x.shape[1] != seq_len:
            raise ValueError(
                f'Period scan expected seq_len={seq_len}, got {x.shape[1]}'
            )
        x = x - x.mean(dim=1, keepdim=True)
        amplitude = torch.fft.rfft(x, dim=1).abs().sum(dim=0)
        amplitude_sum = (
            amplitude if amplitude_sum is None else amplitude_sum + amplitude
        )
        sample_count += x.shape[0]

    if amplitude_sum is None or sample_count == 0:
        raise RuntimeError('Cannot estimate periods from an empty train loader')

    mean_amplitude = amplitude_sum / sample_count
    mean_amplitude[0, :] = -float('inf')
    top_frequency = mean_amplitude.argmax(dim=0).clamp(min=1)
    periods = torch.round(seq_len / top_frequency.float()).long()
    periods = periods.clamp(min=1, max=seq_len)
    raw_periods = periods.cpu().tolist()
    adjusted_periods = adjust_long_periods(raw_periods, seq_len)

    finite_amplitude = mean_amplitude.clone()
    finite_amplitude[0, :] = 0
    spectral_mean = finite_amplitude.mean(dim=0).clamp_min(1e-8)
    peak = mean_amplitude.gather(0, top_frequency.unsqueeze(0)).squeeze(0)
    confidence = peak / spectral_mean
    return (
        adjusted_periods,
        confidence.cpu().tolist(),
        raw_periods,
    )


def save_period_metadata(file_path, periods, confidence, seq_len,
                         data_path, features, enc_in, raw_periods=None):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    metadata = {
        'data_path': data_path,
        'features': features,
        'seq_len': int(seq_len),
        'enc_in': int(enc_in),
        'periods': [int(period) for period in periods],
        'raw_periods': (
            None if raw_periods is None
            else [int(period) for period in raw_periods]
        ),
        'confidence': (
            None if confidence is None
            else [float(value) for value in confidence]
        ),
    }
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_period_metadata(file_path, seq_len, data_path, features, enc_in):
    with open(file_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    expected = {
        'data_path': data_path,
        'features': features,
        'seq_len': int(seq_len),
        'enc_in': int(enc_in),
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f'Period metadata mismatch for {key}: '
                f'expected {expected_value!r}, got {metadata.get(key)!r}'
            )
    periods = [int(period) for period in metadata['periods']]
    confidence = metadata.get('confidence')
    raw_periods = metadata.get('raw_periods')
    if raw_periods is None:
        raw_periods = list(periods)
    else:
        raw_periods = [int(period) for period in raw_periods]
    # Also upgrades caches created before the long-period adjustment existed.
    periods = adjust_long_periods(raw_periods, seq_len)
    return periods, confidence, raw_periods
