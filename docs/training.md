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

## 2. Data

- `XBD_ROOT` (or `--xbd-root`) points at `./data/xbd`.
- Splits are event-disjoint; see `docs/data.md` for the exact events and counts.
- Normalisation statistics are fitted on training tiles only and stored in the
  checkpoint.
- Masks are cached under `data/xbd_cache/` (override with `--cache-dir`).

## 3. Training

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

## 4. Checkpoint contract (v2)

```python
{
  "format_version": 2,
  "model_version": "siamese_unet_resnet34_v1",
  "encoder_name": "resnet34",
  "num_classes": 5,
  "damage_classes": ["background", "no-damage", "minor-damage",
                     "major-damage", "destroyed"],
  "config": {"encoder": ..., "crop_size": 512, "hardware_profile": ...,
             "train_events": [...], "val_events": [...], "source_split": "train"},
  "norm_mean": [r, g, b],
  "norm_std":  [r, g, b],
  "state_dict": {...},
  "state_dict_hash": "a081915f1fa73d59",
  # checkpoint_last.pt only:
  "epoch": 0,
  "optimizer_state_dict": {...},
  "scheduler_state_dict": {...},
  "scaler_state_dict": {...},
  "metrics": {"xview2_score": ..., ...},
}
```

Load with `ai_engine.model.checkpoint.load_checkpoint()`, which uses
`weights_only=True`, accepts legacy `model_state_dict` checkpoints, and fills
defaults for older files. `build_model_from_checkpoint()` constructs the
matching architecture.

## 5. Evaluation

```bash
python eval.py --checkpoint data/models/checkpoint_best.pt \
  --output-dir docs/results
```

`docs/results/metrics.json` records the checkpoint hash, model version, encoder,
class names, data mode (`real_xbd` / `synthetic`), source split, train and
holdout events, tile count, split warnings, and the full xView2 metric block.
`docs/results/` is gitignored.

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
