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

Show command help:

```bash
photic --help
```

### Train on Mallorn

Basic training:

```bash
photic train-mallorn \
  --data-dir /path/to/mallorn \
  --out-dir /path/to/output \
  --device mps
```

A more realistic example:

```bash
photic train-mallorn \
  --data-dir /Users/nmearl/research/mallorn \
  --out-dir /Users/nmearl/code/photic/output/photic_joint_morph \
  --device mps \
  --num-workers 4 \
  --lambda-cls 1.5 \
  --lambda-morph 0.10 \
  --checkpoint-metric best_f1
```

Useful flags:
- `--checkpoint-metric {best_f1,ap,composite}`
- `--num-workers N`
- `--lambda-recon FLOAT`
- `--lambda-cls FLOAT`
- `--lambda-morph FLOAT`
- `--focal-gamma FLOAT`

Optional stage-2 fine-tuning exists, but single-stage joint training is the default and currently the safer baseline:
- `--stage2-epochs`
- `--stage2-from`
- `--stage2-lambda-recon`
- `--stage2-focal-gamma`

### Run Diagnostics on a Checkpoint

```bash
python scripts/mallorn_diagnostics.py \
  --data-dir /Users/nmearl/research/mallorn \
  --checkpoint /Users/nmearl/code/photic/output/photic_joint_morph/best_f1_checkpoint.pt \
  --out-dir /Users/nmearl/code/photic/output/photic_joint_morph/diagnostics \
  --device mps
```

This writes:
- `diagnostics_summary.json`
- `val_object_metrics.csv`
- `tde_reconstructions.png`

## Alert Forecasting

### Poll ALeRCE and Write Forecast JSON Files

```bash
photic forecast-alert-stream \
  --broker alerce \
  --survey lsst \
  --checkpoint /Users/nmearl/code/photic/output/photic_joint_morph/best_f1_checkpoint.pt \
  --out-dir /Users/nmearl/research/alert_forecasts \
  --device mps \
  --since-mjd 59000 \
  --min-context-points 5 \
  --min-context-bands 2 \
  --max-polls 1
```

Important flags:
- `--since-mjd FLOAT`
- `--max-polls INT`
- `--poll-interval FLOAT`
- `--page-size INT`
- `--max-objects INT`
- `--min-context-points INT`
- `--min-context-bands INT`
- `--forecast-days FLOAT`
- `--grid-points-per-band INT`

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
