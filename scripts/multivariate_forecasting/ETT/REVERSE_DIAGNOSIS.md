# Reproducible reverse-loss diagnosis

Run from the repository root after activating the server's existing Python
environment (the same dependencies as the forecasting project):

```bash
bash scripts/multivariate_forecasting/ETT/iTransformer_reverse_diagnose_ETTh1.sh
```

The script first runs synthetic CPU/CUDA preflight checks, then runs seeds
2023, 2024, and 2025. Each seed uses history 96 and forecast horizon 336:

1. A: original iTransformer.
2. C: iTransformer_reverse with reverse loss disabled.
3. Check A/C initial backbone, first batch/prediction/update hashes each epoch,
   epoch losses, and final metrics. Abort on a mismatch.
4. B: iTransformer_reverse with weight 0.05 and reconstruction length 96.

The final summary also verifies B's initial backbone, first pre-update
prediction, and first batches against A. All three use the same training
settings; the reverse loss definition has not changed. Strict deterministic
operations are required. Unsupported deterministic operations stop the run
instead of silently weakening the comparison.

For an initial single-seed check (three training runs):

```bash
SEEDS="2023" bash scripts/multivariate_forecasting/ETT/iTransformer_reverse_diagnose_ETTh1.sh
```

Use `CUDA_VISIBLE_DEVICES=1` to select a physical GPU, or set `PYTHON_BIN` to
the desired Python executable. The tests and training use that same executable.

## Files to upload

- `run.py`
- `data_provider/data_factory.py`
- `experiments/exp_long_term_forecasting.py`
- `model/iTransformer_reverse.py`
- `utils/reproducibility.py`
- `utils/summarize_reverse_diagnosis.py`
- `tests/test_reverse_reproducibility.py`
- `scripts/multivariate_forecasting/ETT/iTransformer_reverse_diagnose_ETTh1.sh`

The server must also retain the previously added `utils/run_logging.py` and
the registration of `iTransformer_reverse` in `experiments/exp_basic.py`.

## Results

Training logs and `reverse_diag_<batch-tag>_summary.txt` are saved under `logs/`.
The default run produces nine training logs and a summary with per-seed MSE/MAE,
sample standard deviations, and paired B-A MSE differences. Negative B-A means
improvement. Checkpoints/results contain the diagnostic batch tag and seed so
these runs do not overwrite older experiments.

Copy the full set of logs and the summary back for analysis. To rebuild a
summary after copying logs, use the batch tag printed by the shell script:

```bash
python utils/summarize_reverse_diagnosis.py --batch-tag <batch-tag>
```

Validation now keeps every window and aggregates errors by element count.
Compare these new A/B/C runs with one another; validation scores from the old
randomly truncated validation loader are not the same evaluation protocol.
Disabled reverse reconstruction is logged as `disabled`, not as zero MSE.

Exact replay is intended within the same software/hardware environment. The
existing checkpoints do not store optimizer/data-loader/private-RNG states
for resuming training mid-run. GPU errors or preflight failures should be
investigated before starting the full experiment batch.
