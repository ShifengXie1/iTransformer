"""Period decomposition and phase-lag layers for variable-token forecasting."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import AttentionLayer
from layers.Transformer_EncDec import EncoderLayer


class DominantPeriodEstimator(nn.Module):
    """Choose one shared period for the current batch (and device replica).

    FFT votes are weighted by peak / total non-DC spectral amplitude. A
    Gaussian kernel gives nearby integer periods a tolerance of period_sigma.
    Constant inputs fall back to the configured period, clipped to FFT bounds.
    """

    def __init__(self, seq_len, mode='fixed', period=24, min_period=4,
                 max_period=None, period_sigma=1.5, eps=1e-8):
        super().__init__()
        self.seq_len = int(seq_len)
        self.mode = mode
        self.period = int(period)
        self.min_period = int(min_period)
        self.max_period = self.seq_len if max_period is None else int(max_period)
        self.period_sigma = float(period_sigma)
        self.eps = eps
        if self.seq_len < 1 or self.period < 1:
            raise ValueError('seq_len and period must be positive')
        if mode not in ('fixed', 'fft'):
            raise ValueError('period_mode must be fixed or fft')
        if not math.isfinite(self.period_sigma) or self.period_sigma <= 0:
            raise ValueError('period_sigma must be finite and positive')
        if mode == 'fft' and not 1 <= self.min_period <= self.max_period <= self.seq_len:
            raise ValueError('FFT periods must satisfy 1 <= min_period <= max_period <= seq_len')

    @torch.no_grad()
    def forward(self, x):
        # x: [B, L, N]. The discrete period is not a trainable quantity.
        if self.mode == 'fixed':
            return self.period
        fallback = min(max(self.period, self.min_period), self.max_period)
        with torch.autocast(device_type=x.device.type, enabled=False):
            amplitude = torch.fft.rfft(x.float(), dim=1).abs()
            if amplitude.shape[1] <= 1:
                return fallback
            amplitude = amplitude[:, 1:, :]  # Exclude DC even for constant inputs.
            peak, index = amplitude.max(dim=1)  # [B, N], k = index + 1
            total = amplitude.sum(dim=1)
            votes = (x.shape[1] / (index.float() + 1)).round()
            weights = peak / total.clamp_min(self.eps)
            valid = ((total > self.eps) & (votes >= self.min_period)
                     & (votes <= self.max_period))
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            if weights.sum().item() <= self.eps:
                return fallback
            # A histogram avoids [candidate_period, B, N] intermediates.
            histogram = amplitude.new_zeros(self.max_period + 1)
            histogram.scatter_add_(0, votes.clamp(0, self.max_period).long().flatten(),
                                   weights.flatten())
            radius = min(self.max_period, int(math.ceil(3 * self.period_sigma)))
            offsets = torch.arange(-radius, radius + 1, device=x.device,
                                   dtype=torch.float32)
            kernel = torch.exp(-0.5 * (offsets / self.period_sigma).square())
            scores = F.conv1d(histogram[None, None], kernel[None, None],
                              padding=radius).flatten()
            return int(scores[self.min_period:self.max_period + 1].argmax().item()) + self.min_period


class PeriodPatchEmbedding(nn.Module):
    """Right-pad and split time into periods; patches are not attention tokens."""

    def forward(self, x, period):
        # [B, L, N] -> [B, N, M, P], M = ceil(L / P).
        batch, length, variables = x.shape
        count = (length + period - 1) // period
        padding = count * period - length
        sequence = x.transpose(1, 2)
        if padding:
            # Replication avoids an artificial jump to zero at the boundary.
            sequence = F.pad(sequence, (0, padding), mode='replicate')
        patches = sequence.reshape(batch, variables, count, period)
        coverage = torch.ones(count, device=x.device, dtype=torch.float32)
        coverage[-1] = (period - padding) / period
        return patches, coverage


class FrequencyComponentDecomposer(nn.Module):
    """Disjoint, exhaustive DC / fundamental / harmonic / residual masks."""

    def __init__(self, harmonic_max_bin=4):
        super().__init__()
        self.harmonic_max_bin = int(harmonic_max_bin)
        if self.harmonic_max_bin < 1:
            raise ValueError('harmonic_max_bin must be at least 1 (1 disables harmonics)')

    def forward(self, patches):
        # patches: [B, N, M, P]; spec: [B, N, M, F].
        period = patches.shape[-1]
        with torch.autocast(device_type=patches.device.type, enabled=False):
            spec = torch.fft.rfft(patches.float(), dim=-1)
            bins = torch.arange(spec.shape[-1], device=patches.device)
            masks = torch.stack((bins == 0, bins == 1,
                                 (bins >= 2) & (bins <= self.harmonic_max_bin),
                                 bins > self.harmonic_max_bin))  # [C=4, F]
            components = torch.fft.irfft(
                spec[:, None] * masks[None, :, None, None, :],
                n=period, dim=-1,
            )  # [B, C, N, M, P]; sum over C reconstructs patches.
        return components.to(patches.dtype), spec, masks


class ComponentPatchEncoder(nn.Module):
    """Encode one component's patches, then pool to one token per variable.

    Fixed mode uses Linear(P, D) directly. For a changing FFT period, interpolate
    the same learned weight along its phase axis to width P. This is still a
    linear P -> D map; no Parameters are created/replaced during forward, so the
    optimizer and checkpoints remain valid across batches with different P.
    """

    def __init__(self, reference_period, d_model, dropout=0.1, temporal_mixer='conv'):
        super().__init__()
        if temporal_mixer not in ('conv', 'linear', 'none'):
            raise ValueError('component_temporal_mixer must be conv, linear or none')
        self.projection = nn.Linear(reference_period, d_model)
        self.position_projection = nn.Linear(1, d_model, bias=False)
        self.temporal_mixer = temporal_mixer
        self.pre_mixer = nn.Linear(d_model, d_model) if temporal_mixer != 'none' else None
        self.conv = (nn.Conv1d(d_model, d_model, kernel_size=3, padding=1,
                              groups=d_model) if temporal_mixer == 'conv' else None)
        self.pool_score = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model) if temporal_mixer != 'none' else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches, coverage):
        # [B, N, M, P] -> embeddings [B, N, M, D].
        batch, variables, count, period = patches.shape
        patches = patches.to(self.projection.weight.dtype)
        if period == self.projection.in_features:
            embeddings = self.projection(patches)
        else:
            with torch.autocast(device_type=patches.device.type, enabled=False):
                weight = F.interpolate(self.projection.weight.float().unsqueeze(0), size=period,
                                       mode='linear', align_corners=False).squeeze(0)
                weight = weight * (self.projection.in_features / period)
            weight = weight.to(self.projection.weight.dtype)
            embeddings = F.linear(patches, weight, self.projection.bias)
        # Relative patch age preserves order under attention pooling, for any M.
        positions = torch.arange(1 - count, 1, device=patches.device,
                                 dtype=self.position_projection.weight.dtype)
        positions = (positions / max(count - 1, 1)).unsqueeze(-1)
        embeddings = embeddings + self.position_projection(positions)[None, None]
        embeddings = self.dropout(F.gelu(embeddings))
        if self.temporal_mixer != 'none':
            mixed = self.dropout(F.gelu(self.pre_mixer(embeddings)))
            if self.conv is not None:
                mixed = self.conv(mixed.reshape(batch * variables, count, -1)
                                  .transpose(1, 2)).transpose(1, 2)
                mixed = mixed.reshape_as(embeddings)
            embeddings = self.norm(embeddings + self.dropout(mixed))
        # Downweight the partly padded final patch in the pooling distribution.
        logits = self.pool_score(embeddings).squeeze(-1).float()
        logits = logits + coverage.clamp_min(1e-8).log()[None, None, :]
        weights = logits.softmax(dim=-1).to(embeddings.dtype)
        return (embeddings * weights.unsqueeze(-1)).sum(dim=2)  # [B, N, D]


class ComponentLagEstimator(nn.Module):
    """Amplitude-weighted phase lag, separately for each frequency component.

    torch.fft uses exp(-j*w*t). If x_i(t) = x_j(t-delay), phi_i-phi_j
    is -w*delay. Thus positive tau[i,j] (j leads i) uses phi_j-phi_i,
    the phase of conj(F_i)*F_j. Lags are wrapped, frequency-local delays;
    phase alone cannot identify a delay beyond half that frequency's period.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, spec, masks, period, coverage):
        # spec: [B, N, M, F]; lag and strength: [B, C=4, N, N].
        batch, variables, _, frequencies = spec.shape
        with torch.autocast(device_type=spec.device.type, enabled=False):
            # Ignore a padded patch for phase estimation if full patches exist.
            # If P > L, use the only available (partly padded) patch.
            patch_weights = (coverage >= 1).float()
            patch_weights = torch.where(patch_weights.sum() > 0,
                                        patch_weights, coverage)
            patch_weights = patch_weights / patch_weights.sum().clamp_min(self.eps)
            weighted_spec = spec * patch_weights.sqrt()[None, None, :, None]
            energy = weighted_spec.abs().square().sum(dim=2)  # [B, N, F]
            band_energy = torch.einsum('bnf,cf->bcn', energy, masks.float())
            shape = (batch, variables, variables)
            numerators = [energy.new_zeros(shape) for _ in range(4)]
            magnitudes = [energy.new_zeros(shape) for _ in range(4)]
            # Loop over frequency only; all variable pairs are vectorized.
            # This avoids materializing [B, N, N, M, F].
            harmonic_max = int(masks[2].sum().item()) + 1
            for k in range(frequencies):
                component = 0 if k == 0 else (1 if k == 1 else (2 if k <= harmonic_max else 3))
                z = weighted_spec[..., k]  # [B, N, M]
                cross = torch.einsum('bim,bjm->bij', z.conj(), z)
                magnitude = cross.abs()
                reliable = magnitude > self.eps
                magnitude = torch.where(reliable, magnitude, torch.zeros_like(magnitude))
                magnitudes[component] = magnitudes[component] + magnitude
                if k == 0:
                    continue  # DC has no phase delay.
                # Do not differentiate angle at zero or vanishing amplitude.
                safe_cross = torch.where(reliable, cross, torch.ones_like(cross))
                phase = torch.angle(safe_cross)  # atan2 wraps to [-pi, pi].
                tau = phase * (period / (2 * math.pi * k))
                numerators[component] = numerators[component] + magnitude * tau
            magnitude = torch.stack(magnitudes, dim=1)
            lag = torch.stack(numerators, dim=1) / magnitude.clamp_min(self.eps)
            # Cauchy-Schwarz normalizes cross magnitude to [0, 1].
            amplitude = band_energy.clamp_min(self.eps).sqrt()
            denominator = amplitude.unsqueeze(-1) * amplitude.unsqueeze(-2)
            strength = (magnitude / denominator.clamp_min(self.eps)).clamp(0, 1)
            diagonal = torch.eye(variables, dtype=torch.bool, device=spec.device)
            lag = lag.masked_fill(diagonal[None, None], 0)
        return lag, strength


class _LagBiasedAttention(nn.Module):
    """AttentionLayer-compatible attention with a precomputed additive bias."""

    def __init__(self, dropout, output_attention):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # Q/K/V: [B, N, heads, head_dim]; delta: [B, N, N].
        with torch.autocast(device_type=queries.device.type, enabled=False):
            scores = torch.einsum('bihe,bjhe->bhij', queries.float(), keys.float())
            scores = scores / math.sqrt(queries.shape[-1])
            if delta is not None:
                scores = scores + delta.float().unsqueeze(1)
            attention = scores.softmax(dim=-1)
        attention = self.dropout(attention).to(values.dtype)
        output = torch.einsum('bhij,bjhd->bihd', attention, values)
        return output.contiguous(), attention if self.output_attention else None


class ComponentLagAttention(nn.Module):
    """Shared Q/K/V and FFN; each component has its own relations and scales."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation='gelu',
                 max_lag=24, lag_alpha=0.1, lag_beta=0.1, output_attention=False):
        super().__init__()
        self.max_lag = float(max_lag)
        if not math.isfinite(self.max_lag) or self.max_lag <= 0:
            raise ValueError('max_lag must be finite and positive')
        if not math.isfinite(lag_alpha) or not math.isfinite(lag_beta):
            raise ValueError('lag_alpha and lag_beta must be finite')
        self.alpha = nn.Parameter(torch.full((4,), float(lag_alpha)))
        self.beta = nn.Parameter(torch.full((4,), float(lag_beta)))
        self.lag_mlp = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        self.layer = EncoderLayer(
            AttentionLayer(_LagBiasedAttention(dropout, output_attention),
                           d_model, n_heads),
            d_model, d_ff, dropout=dropout, activation=activation,
        )

    def forward(self, tokens, lag, strength, period):
        # tokens: [B, C, N, D]; relations: [B, C, N, N].
        outputs, attentions = [], []
        for c in range(4):
            tau = lag[:, c]
            phase = tau * (2 * math.pi / period)
            features = torch.stack((tau / self.max_lag, phase.sin(), phase.cos()), dim=-1)
            features = features.to(self.lag_mlp[0].weight.dtype)
            lag_bias = self.lag_mlp(features).squeeze(-1)
            # Suppress undefined phase bias for missing/zero-energy bands.
            lag_bias = lag_bias * (strength[:, c] > 0).to(lag_bias.dtype)
            bias = self.alpha[c].float() * strength[:, c] + self.beta[c].float() * lag_bias.float()
            output, attention = self.layer(tokens[:, c], delta=bias)
            outputs.append(output)
            attentions.append(attention)
        return torch.stack(outputs, dim=1), attentions  # [B, C, N, D]


class AdaptiveComponentFusion(nn.Module):
    """Learn variable-specific softmax weights, or use uniform ablation weights."""

    def __init__(self, d_model, adaptive=True):
        super().__init__()
        self.adaptive = bool(adaptive)
        self.gate = nn.Linear(d_model, 1) if self.adaptive else None

    def forward(self, tokens):
        # [B, C, N, D] -> component weights [B, N, C] -> fusion [B, N, D].
        if self.adaptive:
            logits = self.gate(tokens).squeeze(-1).permute(0, 2, 1)
            weights = logits.float().softmax(dim=-1).to(tokens.dtype)
        else:
            weights = tokens.new_full((tokens.shape[0], tokens.shape[2], tokens.shape[1]),
                                      1.0 / tokens.shape[1])
        fused = (tokens * weights.permute(0, 2, 1).unsqueeze(-1)).sum(dim=1)
        return fused, weights
