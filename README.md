# photic

`photic` is a small package for joint neural-process style modeling of irregular multi-band light curves.

The current project supports:
- joint reconstruction + binary TDE classification training on the Mallorn dataset
- diagnostics and reconstruction plots for saved checkpoints
- alert-stream polling through ALeRCE
- neural-process forecasting from partial alert photometry
- a NiceGUI viewer for forecast JSON outputs

## Install

Base install:

```bash
pip install -e .
```

With broker support:

```bash
pip install -e ".[brokers]"
```

With GUI support:

```bash
pip install -e ".[gui]"
```

With both:

```bash
pip install -e ".[brokers,gui]"
```

## CLI Overview

Show top-level help:

```bash
photic --help
```

Current commands:
- `train-mallorn`: train a joint NP model on the Mallorn training set. Supports both `convgnp` and `attnnp`, checkpoint selection, staged fine-tuning, and the optional early-time `interestingness` head.
- `evaluate-mallorn-checkpoint`: load a saved checkpoint and score it on the Mallorn validation set using either random or prefix-style contexts, with early-time context/day bins.
- `crossval-mallorn`: train one model per Mallorn `split`, save out-of-fold predictions, and derive a global OOF threshold for leaderboard-style submission tuning.
- `predict-mallorn-test`: run one checkpoint on the Mallorn test set and write a submission CSV in `object_id,prediction` format.
- `predict-mallorn-test-ensemble`: average predictions from multiple checkpoints on the Mallorn test set and write a binary or probability submission file.
- `forecast-alert-stream`: poll ALeRCE for LSST alerts, fetch photometry, run NP forecasts, and write one JSON forecast per alert object.
- `forecast-gui`: launch the NiceGUI viewer for forecast JSON files.

### Train on Mallorn

Strong ConvGNP baseline:

```bash
photic train-mallorn \
  --data-dir /Users/nmearl/research/mallorn \
  --out-dir /Users/nmearl/code/photic/output/photic_joint_retrain \
  --device mps \
  --num-workers 0 \
  --model-type convgnp \
  --lambda-morph 0.15 \
  --checkpoint-metric best_f1
```

Attentive NP with the auxiliary interestingness head:

```bash
photic train-mallorn \
  --data-dir /Users/nmearl/research/mallorn \
  --out-dir /Users/nmearl/code/photic/output/photic_attnnp_interesting \
  --device mps \
  --num-workers 0 \
  --model-type attnnp \
  --lambda-interesting 1.0 \
  --lambda-cls 1.0 \
  --lambda-morph 0.15 \
  --checkpoint-metric best_f1
```

Useful flags:
- `--model-type {convgnp,attnnp}`
- `--context-strategy {random,prefix}`
- `--checkpoint-metric {best_f1,ap,composite}`
- `--lambda-recon FLOAT`
- `--lambda-cls FLOAT`
- `--lambda-interesting FLOAT`
- `--lambda-morph FLOAT`
- `--focal-gamma FLOAT`

Optional stage-2 fine-tuning exists, but single-stage joint training remains the default:
- `--stage2-epochs`
- `--stage2-from`
- `--stage2-lambda-recon`
- `--stage2-focal-gamma`

### Evaluate a Checkpoint

Evaluate a checkpoint on early-time prefix contexts:

```bash
photic evaluate-mallorn-checkpoint \
  --data-dir /Users/nmearl/research/mallorn \
  --checkpoint /Users/nmearl/code/photic/output/photic_attnnp_interesting/best_f1_checkpoint.pt \
  --device mps \
  --num-workers 0 \
  --eval-context-strategy prefix \
  --out-json /Users/nmearl/code/photic/output/photic_attnnp_interesting/early_eval.json
```

This prints:
- overall `best_f1`, `ap`, reconstruction metrics
- context-count bins like `ctx<=3/5/10/20/40/80`
- day-since-anchor bins like `days<=7/30/60/90/120/180`
- auxiliary `interesting` head metrics when enabled

### Cross-Validation and Submission

Train Mallorn split-based cross-validation folds:

```bash
photic crossval-mallorn \
  --data-dir /Users/nmearl/research/mallorn \
  --out-dir /Users/nmearl/code/photic/output/photic_joint_cv \
  --model-type convgnp \
  --context-strategy random \
  --device mps \
  --num-workers 0
```

Single-checkpoint submission:

```bash
photic predict-mallorn-test \
  --data-dir /Users/nmearl/research/mallorn \
  --checkpoint /Users/nmearl/code/photic/output/photic_joint_retrain/best_f1_checkpoint.pt \
  --out-csv /Users/nmearl/code/photic/output/photic_joint_retrain/submission.csv \
  --device mps
```

Ensemble submission from multiple checkpoints:

```bash
photic predict-mallorn-test-ensemble \
  --data-dir /Users/nmearl/research/mallorn \
  --checkpoint /Users/nmearl/code/photic/output/photic_joint_cv/split_01/best_f1_checkpoint.pt \
  --checkpoint /Users/nmearl/code/photic/output/photic_joint_cv/split_02/best_f1_checkpoint.pt \
  --out-csv /Users/nmearl/code/photic/output/photic_joint_cv/submission_ensemble.csv \
  --threshold-json /Users/nmearl/code/photic/output/photic_joint_cv/oof_summary.json \
  --device mps
```

Notes:
- submission files default to binary `0/1` predictions
- `predict-mallorn-test` uses the checkpoint’s saved validation `best_threshold` by default
- `predict-mallorn-test-ensemble` can use the OOF threshold from `oof_summary.json`

### Diagnostics and Plots

Generate diagnostics for a checkpoint:

```bash
python scripts/mallorn_diagnostics.py \
  --data-dir /Users/nmearl/research/mallorn \
  --checkpoint /Users/nmearl/code/photic/output/photic_joint_retrain/best_f1_checkpoint.pt \
  --out-dir /Users/nmearl/code/photic/output/photic_joint_retrain/diagnostics \
  --device mps \
  --num-workers 0 \
  --context-strategy random
```

For early-context plots, use:

```bash
python scripts/mallorn_diagnostics.py \
  --data-dir /Users/nmearl/research/mallorn \
  --checkpoint /Users/nmearl/code/photic/output/photic_attnnp_interesting/best_f1_checkpoint.pt \
  --out-dir /Users/nmearl/code/photic/output/photic_attnnp_interesting/diagnostics \
  --device mps \
  --num-workers 0 \
  --context-strategy prefix
```

Outputs include:
- `diagnostics_summary.json`
- `val_object_metrics.csv`
- `tde_reconstructions.png`
- `tde_prefix_reconstructions.png` when `--context-strategy prefix`

## Alert Forecasting

### Poll ALeRCE and Write Forecast JSON Files

```bash
photic forecast-alert-stream \
  --broker alerce \
  --survey lsst \
  --checkpoint /Users/nmearl/code/photic/output/photic_joint_retrain/best_f1_checkpoint.pt \
  --out-dir /Users/nmearl/research/alert_forecasts \
  --device mps \
  --since-mjd 59000 \
  --min-context-points 5 \
  --min-context-bands 2 \
  --max-polls 1
```

This writes one JSON file per alert object under:

```text
<out-dir>/forecasts/
```

Each file contains:
- broker update metadata
- observed alert photometry
- per-band NP forecast curves
- classification score (`prob_tde`)

### Launch the Forecast GUI

```bash
photic forecast-gui \
  --forecast-dir /Users/nmearl/research/alert_forecasts/forecasts \
  --host 127.0.0.1 \
  --port 8080
```

Then open:

```text
http://127.0.0.1:8080
```

The GUI shows:
- observed photometry by band
- forecast mean and uncertainty by band
- object metadata
- broker update information
- classification confidence

## Notes

- The current alert forecaster expects difference-flux-style inputs. For LSST ALeRCE payloads, this maps to `psfFlux` / `psfFluxErr`.
- Many new alerts only have a few photometric points. Use `--min-context-points` and `--min-context-bands` to avoid generating low-value forecasts.
- Existing forecast JSON files generated before recent CLI patches may not include the observed photometry block needed for full GUI overlay.
