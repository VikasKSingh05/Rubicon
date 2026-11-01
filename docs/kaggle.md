# Kaggle Training Workflow (16 GB profile)

The local machine is a 6 GB RTX 3050. The full event-disjoint split
(2,565 train tiles / 234 validation tiles) is meant to be trained on Kaggle with
the `kaggle_16gb` profile, then the resulting checkpoint is brought back for
serving. The notebook is
`ai-engine/notebooks/kaggle_xbd_training.ipynb`.

## 1. Upload the dataset

Create a Kaggle Dataset from the xBD archives. The notebook auto-discovers a
directory that contains `train/images` and `train/labels`, so either layout
works:

```
xbd/
  train/{images,labels,targets}/
  hold/{images,labels,targets}/
```

or a `train/`-only upload. If discovery fails, set `XBD_DATASET_DIR` in the
notebook to the explicit path.

`hold/` is **not** used as a metric: it shares event names with `train/` and is
only a tile-level qualitative holdout (see `docs/data.md`).

## 2. Notebook settings

| Setting | Value | Why |
| :--- | :--- | :--- |
| Accelerator | `GPU T4 x2` or `GPU P100` | >= 15 GB VRAM selects `kaggle_16gb` |
| Internet | On | Clones the repo and downloads ImageNet encoder weights |
| Persistence | Output only | Checkpoints are written under `/kaggle/working` |

## 3. What the notebook does

1. Prints the real GPU and VRAM (refuses to silently continue on CPU).
2. Locates `XBD_ROOT` and counts pre-disaster pairs.
3. Clones `https://github.com/VikasKSingh05/Rubicon.git` (branch `Vanshika`)
   and installs `segmentation-models-pytorch` + `tensorboard`.
4. Runs `ai_engine.data.check.run_data_check(...)` on the real data — the same
   verifier used locally — and asserts the validation split is non-empty.
5. Trains with `python train.py --config kaggle --epochs 30`.
6. Evaluates with `python eval.py --checkpoint .../checkpoint_best.pt`.
7. Prints the provenance block from `metrics.json` (checkpoint hash, data mode,
   train/holdout events, tile count, xView2, localization F1, damage F1).

## 4. Expected runtime (order of magnitude, not a guarantee)

At `512x512`, batch 8, AMP fp16, ResNet34 encoder, ~2,565 training tiles:

- ~2,565 / 8 ≈ 320 steps per epoch.
- A T4/P100 epoch is roughly in the tens of minutes; 30 epochs is a multi-hour
  session. Kaggle sessions are capped (about 9 hours with a verified account),
  so use `checkpoint_last.pt` + `--resume` across sessions if needed.
- No runtime number is quoted as fact here because it has **not** been measured
  on Kaggle yet. Measure it in the first session and update this section.

## 5. Collecting artifacts

From the notebook **Output** tab download:

- `models/checkpoint_best.pt` (weights + config + norm stats)
- `models/siamese_unet.pt` (serving artifact)
- `results/metrics.json`
- `samples/*.png` (qualitative overlays)

Place the serving checkpoint in `data/models/` locally (gitignored) and point
`RUBICON_MODEL_DIR` at that directory:

```powershell
$env:RUBICON_MODEL_DIR="data/models"
python -m uvicorn ai_engine.serve:app --host 0.0.0.0 --port 8000
```

Serving normalises inputs with the checkpoint's own `norm_mean`/`norm_std`, so
no manual preprocessing step is required.

## 6. Honest reporting rules

- Report metrics **per held-out event** (`mexico-earthquake`,
  `palu-tsunami`) as well as the aggregate. Two events give a coarse estimate
  and per-event variance is large.
- Always state the checkpoint hash and `data_mode` (`real_xbd`) with any number.
- Do not quote CPU smoke-test numbers (xView2 ≈ 0.17 on 5 tiles) as model
  quality; they only prove the pipeline runs.
- The local GPU has not yet produced metrics because the installed PyTorch
  wheel was CPU-only (`docs/audit.md`). Kaggle is the supported path to real
  numbers until a CUDA wheel is installed locally.

## 7. Local equivalent of the same run

If a CUDA-enabled PyTorch is installed locally:

```powershell
$env:XBD_ROOT="./data/xbd"
python train.py --epochs 30 --config local_6gb
python eval.py --checkpoint data/models/checkpoint_best.pt
```

On CUDA OOM the trainer reduces the crop (512 -> 384 -> 256), rebuilds the
loaders, and retries the current batch. At 256 it fails loudly instead of
looping.
