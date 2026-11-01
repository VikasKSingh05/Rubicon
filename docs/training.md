# Training, Evaluation & Checkpoints

## 1. Hardware Profiles

Selected automatically by `ai_engine.model.hardware.detect_hardware()` and
overridable with `--config`:

| Profile | When selected | Encoder | Crop | Batch | Accum | AMP |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `local_6gb` | CUDA GPU with < 14 GB VRAM (RTX 3050) | `resnet34` | 512 | 2 | 4 | fp16 |
| `kaggle_16gb` | CUDA GPU with >= 14 GB VRAM | `resnet34` | 512 | 8 | 1 | fp16 |
| `cpu_fallback` | No CUDA | `resnet18` | 256 | 2 | 2 | off |

`--config local_6gb` / `kaggle` on a machine without CUDA falls back to the CPU
profile with a warning instead of crashing.

On CUDA OOM the training loop clears the cache, reduces the crop
(512 -> 384 -> 256), rebuilds the loaders, and retries the **current batch**
without restarting the epoch. At 256 it raises instead of looping forever.

Kaggle runs (full split, 16 GB) are documented in `docs/kaggle.md`; the
notebook is `ai-engine/notebooks/kaggle_xbd_training.ipynb`.

## 2. Data

- `XBD_ROOT` (or `--xbd-root`) points at `./data/xbd`.
- The supported root layout is `xbd/train/{images,labels,targets}` and, if
  provided, `xbd/hold/{images,labels,targets}`. Matching pre/post images and
  label JSON files are required. The trainer reads `train/`; it does not train
  on `hold/`.
- Splits are event-disjoint; see `docs/data.md` for the exact events and counts.
- Normalisation statistics are fitted on training tiles only and stored in the
  checkpoint.
- Masks are cached under `data/xbd_cache/` (override with `--cache-dir`).
- `hold/` contains tile-level holdout data from events also found in `train/`.
  The primary generalization metric is therefore the event-disjoint validation
  subset carved from `train/`; `hold/` metrics are supplemental and must be
  reported separately.

## 3. Training

### Windows PowerShell (local dataset)

From the `Rubicon\ai-engine` directory, set the path to the folder containing
your `train` and `hold` directories:

```powershell
Set-Location C:\Users\LENOVO\Desktop\Rubicon\ai-engine
$XbdRoot = "D:\datasets\xbd"

python -m pip install -r requirements.txt
python -m ai_engine.data.check --xbd-root $XbdRoot --samples 5 --crosscheck 5
```

Confirm the data check reports labeled pairs under `train/`, a non-empty
event-disjoint train/validation split, and no unexpected mask errors. Then run
the full dataset (no `--synthetic`, no reduced `--subset`, and no `--max-val`):

```powershell
python train.py --xbd-root $XbdRoot --config auto --epochs 30 --subset 1.0 `
  --output-dir data/models/xbd --log-dir runs/xbd-mentor-demo
```

The trainer uses the detected hardware profile; CUDA is strongly recommended
for a full run. ImageNet encoder weights are enabled by default and may need to
download on first run. If that is not possible, add `--no-pretrained` and
record that choice in your report. A missing or invalid dataset path now stops
the run instead of silently training on synthetic data. A smoke run using
`--subset` or `--max-val` is useful for checking the pipeline, but is not a
full-data result.

The run writes `data/models/xbd/training_metrics.json`, which records the
dataset path and mode, model/parameter counts, hardware profile,
hyperparameters, event and tile counts, split warnings, best epoch, and
per-epoch training/validation history. `checkpoint_best.pt` is the checkpoint
selected by event-disjoint validation; `checkpoint_last.pt` can resume
training. Both are kept separate from the HSI/LiDAR model artifacts.

### Existing CLI examples

```bash
# Local GPU (auto-detected)
python train.py --epochs 30 --config auto

# Kaggle 16 GB
python train.py --epochs 30 --config kaggle

# CPU smoke test on real data (small subset, no pretrained download)
python train.py --config cpu --epochs 1 --subset 0.002 --max-val 2 \
  --no-pretrained --output-dir data/models/smoke

# Resume
python train.py --resume data/models/checkpoint_last.pt --epochs 30
```

Outputs in `--output-dir` (default `data/models`):

| File | Contents | Resumable |
| :--- | :--- | :--- |
| `checkpoint_last.pt` | weights + optimizer/scheduler/scaler + metrics | yes |
| `checkpoint_best.pt` | best-xView2 weights + config + norm stats | no |
| `siamese_unet.pt` | same as best; serving artifact | no |
| `model.pt` | same as best; legacy serving path | no |

## 4. Evaluation and mentor demo

Run evaluation against the same xBD root. The default `val` split is
event-disjoint and is the primary result. If you have `hold/`, evaluate it
separately as a supplemental tile-level check:

```powershell
python eval.py --xbd-root $XbdRoot --checkpoint data/models/xbd/checkpoint_best.pt `
  --split val --output-dir data/models/xbd/eval-val

python eval.py --xbd-root $XbdRoot --checkpoint data/models/xbd/checkpoint_best.pt `
  --split hold --output-dir data/models/xbd/eval-hold
```

Each evaluation directory includes `metrics.json`, `confusion_matrix.png`,
and prediction overlays. The JSON names the evaluated split, records the
checkpoint hash and xView2 metrics, and gives tile/event counts. Do not combine
the `val` and `hold` scores into one number or call the `hold/` score
event-disjoint.

For the mentor, show:

- `training_metrics.json`: architecture, parameters, hyperparameters, data
  split, hardware, training curves, and selected best epoch.
- `eval-val/metrics.json`: the primary validation xView2 score, localization
  F1, harmonic damage F1, per-class F1, and confusion matrix.
- A few `eval-val/eval_overlay_*.png` files: pre/post imagery, ground truth,
  and model predictions.
- Optionally, the separate `eval-hold/metrics.json` and overlays, explicitly
  labeled as supplemental tile-level hold results.

### Serve the trained model in Rubicon

Training places the serving checkpoint at `ai-engine/data/models/xbd/siamese_unet.pt`.
The Compose satellite service is configured to load that directory. From the
repository root, rebuild/restart it after training:

```powershell
docker compose up -d --build ai-engine-satellite
Invoke-RestMethod http://localhost:8002/health
```

The health response should report `"model_loaded": true`. The backend's
satellite upload flow forwards pre/post image pairs to this service; the
legacy HSI/LiDAR engine remains separate. For a native local service instead
of Docker, run from `ai-engine`:

```powershell
$env:RUBICON_MODEL_DIR = (Resolve-Path .\data\models\xbd).Path
python -m uvicorn ai_engine.serve:app --host 127.0.0.1 --port 8002
```

## 5. Checkpoint contract (v2)

Load with `ai_engine.model.checkpoint.load_checkpoint()`, which uses
`weights_only=True`, accepts legacy `model_state_dict` checkpoints, and fills
defaults for older files. `build_model_from_checkpoint()` constructs the
matching architecture.

## 6. Verified smoke run (real xBD, CPU)

Command:

```bash
$env:XBD_ROOT="./data/xbd"
python train.py --config cpu --epochs 1 --subset 0.002 --max-val 2 \
  --no-pretrained --output-dir data/models/smoke
```

Observed output (real run, not fabricated):

```
[INFO] Reduced subset training (0%): 5 train / 1 val tiles.
[INFO] Normalization mean=[0.196501, 0.27153, 0.163887] std=[0.102019, 0.100762, 0.077947]
Epoch [01/01] Train Loss: 1.9209 (Loc: 1.913, Dmg: 1.926) | Val Loss: 1.6607 |
Loc F1: 0.5574 | Dmg Harmonic F1: 0.0000 | xView2 Score: 0.1672 | Time: 7.6s
```

Evaluation on 2 validation tiles with the same checkpoint: xView2 0.1719,
localization F1 0.5731, damage F1 0.0000.

**These numbers only prove the plumbing works.** Five training tiles for one
epoch cannot produce a usable model, and the damage F1 of 0 is expected at that
scale. Real metrics require the full event-disjoint split on a CUDA GPU or
Kaggle; see `docs/audit.md` for the environment status.
