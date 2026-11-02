# xBD EDA and preprocessing run

This report records a real-data EDA and data-pipeline check run against the
local xBD dataset. It is not a new training or evaluation run.

## Dataset and split

- Dataset: local `data/xbd/`; no dataset files were changed.
- Source: `train/`, 2,799 paired tiles across 10 disaster events.
- Existing event-disjoint manifest: 2,565 training tiles across 8 events and
  234 validation tiles across 2 events. The event sets do not overlap.
- EDA subset: 769 training tiles (30%, selected deterministically with seed 42),
  matching the xBD v3 training run. Validation was not subsetted.
- `hold/` was not used for EDA or training. The data checker read its inventory
  and compared six holdout targets for diagnostic purposes; it did not write
  to `hold/`.

## Training-subset label statistics

Object counts come from the selected tiles' post-disaster JSON labels. Pixel
counts come from the same rasterized damage masks used by training, counting
only building classes 1–4.

| Damage class | Labelled objects | Building pixels | Training weight |
|---|---:|---:|---:|
| No damage | 17,372 | 21,805,566 | 0.491263 |
| Minor damage | 3,694 | 4,305,106 | 1.105620 |
| Major damage | 4,206 | 6,096,540 | 0.929086 |
| Destroyed | 2,534 | 2,422,044 | 1.474031 |

Weights match `training_metrics.json`: normalized inverse-square-root pixel
frequency, capped at 10. Training uses shuffled tiles only; it does not combine
these weights with a weighted oversampling sampler.

## Image-pair sample

All 32 sampled image pairs were 1024×1024. RGB statistics were estimated after
resizing each image to 128×128; the report JSON stores the
channel values in OpenCV BGR order. Mean absolute pre/post difference was
`[0.061196, 0.070537, 0.082647]` for B, G, R respectively (pixels scaled to
0–1). This is a sample-level image summary, not a spectral analysis.

## Preprocessing and label checks

- The data checker found 2,799 training pairs and 933 holdout pairs.
- It reused the current event-split manifest; split regeneration is now opt-in
  via `--rebuild-splits`.
- JSON polygon masks matched the shipped `targets/` for 12 sampled
  event-disjoint train/validation tiles at 99.9994% localization agreement and
  99.9447% damage agreement.
- Per-class IoU on that sample: background 0.9994, no-damage 0.9795, and
  minor/major/destroyed 1.0000 each.
- Five qualitative overlays were generated for review.

The checker sampled six `hold/` targets separately: localization agreement was
99.9977% and damage agreement was 99.9961%. This is a diagnostic consistency
check only; `hold/` shares event names with `train/` and is not the primary
event-disjoint generalization metric.

## Local run artifacts

Generated artifacts are ignored local files, not committed to Git:

- `../ai-engine/runs/mentor-pipeline-20261102-132316/eda.json`
- `../ai-engine/runs/mentor-pipeline-20261102-132316/data-check.json`
- `../ai-engine/runs/mentor-pipeline-20261102-132316/overlays/`

Reproduce from the `ai-engine/` directory, using a fresh output directory:

```powershell
$XbdRoot = "..\data\xbd"
$RunOutput = "runs\xbd-pipeline-$(Get-Date -Format yyyyMMdd-HHmmss)"
python scripts\eda_xbd.py --xbd-root $XbdRoot --subset 0.3 `
  --save-json "$RunOutput\eda.json"
python -m ai_engine.data.check --xbd-root $XbdRoot --crosscheck 12 --samples 5 `
  --samples-dir "$RunOutput\overlays" --json-out "$RunOutput\data-check.json"
```
