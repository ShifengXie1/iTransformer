"""Read diagnostic logs, verify A/C equivalence, and summarize paired results.

Uses only the standard library; can also be run locally after copying logs.
"""

import argparse
import ast
from pathlib import Path
import re
import statistics


def read_runs(log_dir, batch_tag):
    runs = {}
    for path in sorted(Path(log_dir).glob('*.log')):
        text = path.read_text(encoding='utf-8')
        namespace = next((line for line in text.splitlines()
                          if line.startswith('Namespace(')), None)
        if namespace is None:
            continue
        if 'reverse_diag_rng_' + batch_tag not in namespace:
            continue
        params = {kw.arg: ast.literal_eval(kw.value)
                  for kw in ast.parse(namespace, mode='eval').body.keywords}
        if params.get('des') != 'reverse_diag_rng_' + batch_tag:
            continue
        group = params['model_id'].rsplit('_', 1)[-1]
        if group not in ('A', 'B', 'C'):
            continue
        key = (params['seed'], group)
        if key in runs:
            raise ValueError(f'Duplicate run for {key}; use a unique batch tag')
        initial = re.search(r'Repro initial_backbone_sha256: (\w+)', text)
        traces = {(int(epoch), name): value for epoch, name, value in re.findall(
            r'Repro epoch=(\d+) (\w+_sha256): (\w+)', text
        )}
        epochs = [tuple(map(float, row)) for row in re.findall(
            r'Epoch: (\d+), Steps: \d+ \| Train Loss: ([\d.eE+-]+) '
            r'Vali Loss: ([\d.eE+-]+) Test Loss: ([\d.eE+-]+)', text
        )]
        final = re.findall(r'^mse:([\d.eE+-]+), mae:([\d.eE+-]+)', text, re.M)
        if not initial or not traces or not epochs or not final:
            raise ValueError(f'Incomplete run or missing diagnostics: {path.name}')
        runs[key] = dict(path=path, initial=initial[1], traces=traces,
                         epochs=epochs, final=tuple(map(float, final[-1])))
    return runs


def check_ac(runs, seed):
    if any((seed, group) not in runs for group in ('A', 'C')):
        raise ValueError(f'Missing A/C logs for seed {seed}')
    a, c = (runs[(seed, group)] for group in ('A', 'C'))
    if a['initial'] != c['initial']:
        raise ValueError(f'seed {seed}: A/C initial backbone differs')
    for run in (a, c):
        for row in run['epochs']:
            for name in ('first_batch_sha256', 'first_prediction_sha256',
                         'first_step_backbone_sha256'):
                if (int(row[0]), name) not in run['traces']:
                    raise ValueError(f'seed {seed}: missing {name} in {run["path"].name}')
    if a['traces'] != c['traces']:
        differing = [key for key in sorted(set(a['traces']) | set(c['traces']))
                     if a['traces'].get(key) != c['traces'].get(key)]
        raise ValueError(f'seed {seed}: A/C reproducibility mismatch: {differing}')
    if len(a['epochs']) != len(c['epochs']):
        raise ValueError(f'seed {seed}: A/C stopped at different epochs')
    pairs = list(zip(a['final'], c['final']))
    pairs += [(x, y) for ra, rc in zip(a['epochs'], c['epochs']) for x, y in zip(ra, rc)]
    if any(abs(x - y) > 1e-6 for x, y in pairs):
        raise ValueError(f'seed {seed}: A/C loss or final metric differs by more than 1e-6')


def make_report(runs, seeds, ac_only=False):
    lines = []
    deltas = []
    for seed in seeds:
        check_ac(runs, seed)
        lines.append(f'Seed {seed}: A/C PASS (initial weights, batch/prediction/step hashes, losses, final metrics)')
        if ac_only:
            continue
        if (seed, 'B') not in runs:
            raise ValueError(f'Missing B log for seed {seed}')
        a, b = runs[(seed, 'A')], runs[(seed, 'B')]
        if a['initial'] != b['initial']:
            raise ValueError(f'seed {seed}: B initial backbone differs from A')
        if a['traces'][(1, 'first_prediction_sha256')] != b['traces'].get((1, 'first_prediction_sha256')):
            raise ValueError(f'seed {seed}: B first prediction differs before the first update')
        common_epochs = min(len(a['epochs']), len(b['epochs']))
        for epoch in range(1, common_epochs + 1):
            key = (epoch, 'first_batch_sha256')
            if a['traces'][key] != b['traces'].get(key):
                raise ValueError(f'seed {seed}: B first batch differs at epoch {epoch}')
        for group in ('A', 'B', 'C'):
            run = runs[(seed, group)]
            best_epoch = int(min(run['epochs'], key=lambda row: row[2])[0])
            lines.append(f'  {group}: MSE={run["final"][0]:.7f} MAE={run["final"][1]:.7f} best_epoch={best_epoch} log={run["path"].name}')
        delta = b['final'][0] - a['final'][0]
        deltas.append(delta)
        lines.append(f'  B-A MSE: {delta:+.7f} ({100 * delta / a["final"][0]:+.3f}%; negative is better)')
    if deltas:
        lines.append('Across seeds (sample standard deviation):')
        for group in ('A', 'B', 'C'):
            for index, metric in enumerate(('MSE', 'MAE')):
                values = [runs[(seed, group)]['final'][index] for seed in seeds]
                spread = f'{statistics.stdev(values):.7f}' if len(values) > 1 else 'N/A'
                lines.append(f'  {group} {metric}: mean={statistics.mean(values):.7f} std={spread}')
        lines.append(f'Paired B-A MSE mean: {statistics.mean(deltas):+.7f}; B improves in {sum(d < 0 for d in deltas)}/{len(deltas)} seeds.')
        lines.append('This is a paired diagnostic, not a statistical significance claim. Select checkpoints by validation loss.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', default=str(Path(__file__).resolve().parents[1] / 'logs'))
    parser.add_argument('--batch-tag', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--check-ac', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.batch_tag):
        parser.error('batch-tag must contain only letters, digits, underscores and hyphens')
    try:
        runs = read_runs(args.log_dir, args.batch_tag)
        seeds = [args.seed] if args.seed is not None else sorted({seed for seed, _ in runs})
        if not seeds:
            raise ValueError('No matching diagnostic logs found')
        report = make_report(runs, seeds, ac_only=args.check_ac)
    except (ValueError, SyntaxError, KeyError) as error:
        parser.exit(1, f'Diagnostic check FAILED: {error}\n')
    print(report)
    if not args.check_ac:
        path = Path(args.log_dir) / f'reverse_diag_{args.batch_tag}_summary.txt'
        path.write_text(report + '\n', encoding='utf-8')
        print('Summary saved:', path)


if __name__ == '__main__':
    main()
