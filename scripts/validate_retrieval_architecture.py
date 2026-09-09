"""Retrain retrieval architecture ablations from scratch on ETT-hour data.

No existing checkpoints are accepted or read. Each run starts from the same
backbone initialization and training data order for its horizon/seed. Validation
selects this run's best weights; test is evaluated once with those weights.
Standalone iTransformer training is not included.

Default: 5 retrieval variants x 4 horizons x 1 seed = 20 training runs.
Compare adapted, previous full, local candidates only, position gate only,
and both changes. Previous six variant names retain their original settings.
  python scripts/validate_retrieval_architecture.py
  python scripts/validate_retrieval_architecture.py --horizons 720

Validation/test cover every window without shuffling or dropping the last batch.
This differs from run.py's shuffled validation subset. Compare variants within
this script; do not expect bit-for-bit agreement with previous server runs.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_provider.data_loader import Dataset_ETT_hour
from model.itransformer_retrieval import Model, initialize_retrieval_memory, retrieval_setting_suffix
from utils.retrieval_diagnostics import RetrievalDiagnostics, print_retrieval_summary


VARIANTS = {
    'adapted': 'Previous embedding retrieval with Future adaptation and the old gate',
    'ctx1': 'Contextual variable retrieval with the old gate',
    'full': 'Global candidate filtering + local refinement + consensus gate',
    'all_candidates': 'Full model without global shortlist truncation; keep global score weighting',
    'no_disagreement': 'Full model without the monotone disagreement penalty; keep variance penalty',
    'all_candidates_no_disagreement': 'Remove shortlist truncation and disagreement penalty together',
    'local_candidates': 'Variable-specific causal Top-K with global weighting; previous consensus gate',
    'horizon_only': 'Previous full retrieval with position-aware consensus gate',
    'local_horizon': 'Local candidates + global weighting + position-aware consensus gate',
}

DEFAULT_VARIANTS = ['adapted', 'full', 'local_candidates', 'horizon_only', 'local_horizon']


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-path', default=str(ROOT / 'dataset/ETT-small'))
    p.add_argument('--data-path', default='ETTh1.csv')
    p.add_argument('--features', choices=['M', 'MS', 'S'], default='M')
    p.add_argument('--target', default='OT')
    p.add_argument('--horizons', type=int, nargs='+', choices=[96, 192, 336, 720], default=[96, 192, 336, 720])
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=DEFAULT_VARIANTS)
    p.add_argument('--seeds', type=int, nargs='+', default=[2023])
    p.add_argument('--seq-len', type=int, default=96)
    p.add_argument('--label-len', type=int, default=48)
    p.add_argument('--d-model', type=int, default=None, help='Default: 256 for H96/192; 512 for H336/720')
    p.add_argument('--d-ff', type=int, default=None, help='Default: same width as d_model')
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--e-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--use-norm', type=int, choices=[0, 1], default=1)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--memory-size', type=int, default=1024)
    p.add_argument('--top-k', type=int, default=8)
    p.add_argument('--global-top-k', type=int, default=64)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--base-loss-weight', type=float, default=.2)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='auto', help='auto, cpu, cuda:0, ...')
    p.add_argument('--amp', action='store_true', help='Use CUDA AMP for every variant')
    p.add_argument('--output-dir', default=None, help='New directory; defaults to retrieval_validation/<timestamp>')
    p.add_argument('--max-train-batches', type=int, default=0, help='Smoke check only; 0 means full training data')
    p.add_argument('--max-eval-batches', type=int, default=0, help='Smoke check only; 0 means full validation/test data')
    return p


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(args, horizon, variant):
    width = args.d_model if args.d_model is not None else (256 if horizon <= 192 else 512)
    if width < 1 or args.n_heads < 1 or width % args.n_heads:
        raise ValueError('d_model must be positive and divisible by n_heads')
    new_structure = variant not in ('adapted', 'ctx1')
    return SimpleNamespace(
        seq_len=args.seq_len, pred_len=horizon, d_model=width,
        d_ff=args.d_ff if args.d_ff is not None else width,
        n_heads=args.n_heads, e_layers=args.e_layers, dropout=args.dropout,
        embed='timeF', freq='h', factor=1, activation='gelu', class_strategy='projection',
        output_attention=False, use_norm=bool(args.use_norm), use_retrieval=True,
        retrieval_use_future=True, retrieval_use_gate=True, retrieval_weighted=True,
        retrieval_reliability_gate=True, retrieval_contextual=variant != 'adapted',
        retrieval_global_filter=new_structure, retrieval_consensus_gate=new_structure,
        retrieval_local_candidates=variant in ('local_candidates', 'local_horizon'),
        retrieval_horizon_gate=variant in ('horizon_only', 'local_horizon'),
        retrieval_disagreement_penalty=variant not in ('no_disagreement', 'all_candidates_no_disagreement'),
        retrieval_global_top_k=args.global_top_k, retrieval_top_k=args.top_k,
        retrieval_memory_size=args.memory_size, retrieval_temperature=args.temperature,
        retrieval_stride=1, retrieval_chunk_size=args.chunk_size, retrieval_variable_chunk_size=32,
        retrieval_base_loss_weight=args.base_loss_weight)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


@torch.no_grad()
def evaluate(model, loader, horizon, feature_start, device, amp, limit):
    model.eval()
    diagnostics = RetrievalDiagnostics(feature_start=feature_start)
    for batch, (x, y, xm, _) in enumerate(loader, 1):
        with torch.cuda.amp.autocast(enabled=amp):
            parts = model(x.float().to(device), xm.float().to(device), None, None, return_components=True)
        diagnostics.update(parts, y[:, -horizon:, feature_start:].float())
        if limit and batch >= limit:
            break
    return diagnostics.summary()


def train_variant(args, cfg, variant, seed, datasets, device, directory):
    seed_everything(seed)
    # Model constructs the backbone before any variant-specific modules, so
    # the same seed/config gives exactly the same initial backbone weights.
    model = Model(cfg).to(device)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == 'cuda')
    train_loader = DataLoader(datasets['train'], shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(seed), **loader_options)
    if not len(train_loader):
        raise ValueError('batch-size exceeds the number of training windows')
    train_loader = initialize_retrieval_memory(model, datasets['train'], train_loader)
    if variant in ('all_candidates', 'all_candidates_no_disagreement'):
        # Keep global_filter=True: switching it off would also change the
        # representation and remove global weighting, confounding the ablation.
        model.retrieval_global_top_k = model.memory_starts.numel()
        cfg.retrieval_global_top_k = model.retrieval_global_top_k
    val_loader = DataLoader(datasets['val'], shuffle=False, **loader_options)
    test_loader = DataLoader(datasets['test'], shuffle=False, **loader_options)
    directory.mkdir()
    write_json(directory / 'config.json', dict(model=vars(cfg), seed=seed, variant=variant,
                                              setting_suffix=retrieval_setting_suffix(cfg), training=vars(args)))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    amp = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    feature_start = -1 if args.features == 'MS' else 0
    best, best_state, best_epoch, stale, history = float('inf'), None, 0, 0, []
    # Architecture initialization consumes different numbers of random draws.
    # Reset dropout RNG separately; the loader uses its own seeded generator.
    seed_everything(seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.monotonic()
        loss_sum, sample_count = 0., 0
        for batch, (x, y, xm, _, ends) in enumerate(train_loader, 1):
            x, xm = x.float().to(device), xm.float().to(device)
            target = y[:, -cfg.pred_len:, feature_start:].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                parts = model(x, xm, None, None, query_end=ends.to(device), return_components=True)
                prediction = parts['prediction'][:, :, feature_start:]
                loss = (prediction.float()-target).square().mean() + model.base_auxiliary_loss(parts, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss in {} epoch {} batch {}'.format(variant, epoch, batch))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * x.size(0)
            sample_count += x.size(0)
            if batch % 100 == 0:
                print('H={} {} seed={} epoch={} batch={}/{} loss={:.6f}'.format(
                    cfg.pred_len, variant, seed, epoch, batch, len(train_loader), loss.item()), flush=True)
            if args.max_train_batches and batch >= args.max_train_batches:
                break
        validation = evaluate(model, val_loader, cfg.pred_len, feature_start, device, amp, args.max_eval_batches)
        score = validation['fused_mse']
        row = dict(epoch=epoch, train_loss=loss_sum/sample_count, validation=validation,
                   seconds=time.monotonic()-started)
        history.append(row)
        print('H={} {} seed={} epoch={} train={:.6f} seconds={:.1f}'.format(
            cfg.pred_len, variant, seed, epoch, row['train_loss'], row['seconds']), flush=True)
        print_retrieval_summary('validation', validation)
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, directory / 'checkpoint.pth')
        else:
            stale += 1
        write_json(directory / 'history.json', history)
        if stale >= args.patience:
            break
        # Same type1 schedule as utils/tools.py: epochs 1 and 2 use the initial LR.
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate * 0.5 ** (epoch - 1)
    if best_state is None:
        raise RuntimeError('Training did not produce a finite validation checkpoint')
    # Only weights produced by THIS run are used. No checkpoint file is read.
    model.load_state_dict(best_state, strict=True)
    test = evaluate(model, test_loader, cfg.pred_len, feature_start, device, amp, args.max_eval_batches)
    print_retrieval_summary('test', test)
    report = dict(variant=variant, description=VARIANTS[variant], horizon=cfg.pred_len, seed=seed,
                  best_epoch=best_epoch, validation=history[best_epoch-1]['validation'], test=test,
                  config=vars(cfg), device=str(device), torch_version=torch.__version__,
                  partial_data=bool(args.max_train_batches or args.max_eval_batches),
                  metric_note='base is this retrieval model\'s jointly trained branch, not a standalone baseline.')
    write_json(directory / 'metrics.json', report)
    return dict(horizon=cfg.pred_len, seed=seed, variant=variant, best_epoch=best_epoch,
                partial_data=report['partial_data'],
                validation_mse=best, **{k: test[k] for k in (
                    'base_mse', 'retrieval_mse', 'fused_mse', 'fused_mae', 'mean_gate',
                    'fused_mse_q1', 'fused_mse_q2', 'fused_mse_q3', 'fused_mse_q4',
                    'base_mse_q1', 'base_mse_q2', 'base_mse_q3', 'base_mse_q4',
                    'retrieval_mse_q1', 'retrieval_mse_q2', 'retrieval_mse_q3', 'retrieval_mse_q4',
                    'mean_gate_q1', 'mean_gate_q2', 'mean_gate_q3', 'mean_gate_q4')})


def main():
    args = parser().parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.seq_len, args.threads, args.e_layers) < 1:
        raise ValueError('epochs, patience, batch-size, seq-len, threads and e-layers must be positive')
    if not 0 <= args.label_len <= args.seq_len or min(args.workers, args.max_train_batches, args.max_eval_batches) < 0:
        raise ValueError('label-len must be within [0, seq-len]; workers/batch limits must be nonnegative')
    if args.learning_rate <= 0 or not np.isfinite(args.learning_rate):
        raise ValueError('learning-rate must be finite and positive')
    for name in ('horizons', 'variants', 'seeds'):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    torch.set_num_threads(args.threads)
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp requires a CUDA device')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = (Path(args.output_dir) if args.output_dir else
              ROOT / 'retrieval_validation' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', dict(arguments=vars(args), variants=VARIANTS,
                                             protocol='Fresh training; same backbone seed, batch order, dropout seed and common hyperparameters. No parameter search.'))
    total = len(args.horizons)*len(args.variants)*len(args.seeds)
    print('Training {} runs from scratch on {}. Output: {}'.format(total, device, output), flush=True)
    rows = []
    for horizon in args.horizons:
        datasets = {split: Dataset_ETT_hour(args.root_path, flag=split,
                                            size=[args.seq_len, args.label_len, horizon], features=args.features,
                                            data_path=args.data_path, target=args.target, timeenc=1, freq='h')
                    for split in ('train', 'val', 'test')}
        if any(len(d) < 1 for d in datasets.values()):
            raise ValueError('Dataset has no windows for horizon {}'.format(horizon))
        for seed in args.seeds:
            for variant in args.variants:
                cfg = make_config(args, horizon, variant)
                directory = output / 'H{}_seed{}_{}'.format(horizon, seed, variant)
                rows.append(train_variant(args, cfg, variant, seed, datasets, device, directory))
                with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    print('\nSummary:', output / 'summary.csv', flush=True)


if __name__ == '__main__':
    main()
