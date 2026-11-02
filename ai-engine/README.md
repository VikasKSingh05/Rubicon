# Rubicon AI Engine — xBD Dual-Task Siamese U-Net

This part of Rubicon contains the xBD Siamese U-Net for building localization
and damage assessment from paired pre- and post-disaster optical satellite
imagery. The same FastAPI service also provides high-resolution drone triage.

## 🏗️ Model Architecture

The architecture implements a **Dual-Task Siamese U-Net** based on the research specification:
1. **Weight-Shared Siamese Encoders**: A shared convolutional backbone (ResNet-18 / ResNet-34) processes the pre-disaster and post-disaster images in parallel to extract multiscale feature representations:
   - $F_{\text{pre}} = \text{Encoder}(I_{\text{pre}})$
   - $F_{\text{post}} = \text{Encoder}(I_{\text{post}})$
2. **Feature Fusion Layer**: Fuses pre and post feature maps via concatenation and absolute difference:
   - $F_{\text{fused}} = [F_{\text{pre}} \,\|\, F_{\text{post}} \,\|\, |F_{\text{post}} - F_{\text{pre}}|]$
3. **Shared U-Net Decoder and Two Task Heads**:
   - **Localization Head**: Predicts building footprints (Binary Cross-Entropy + Dice Loss).
   - **Damage Head**: Predicts four damage classes only on ground-truth building pixels; background is ignored by this loss and handled by localization:
     - `No damage`
     - `Minor damage`
     - `Major damage`
     - `Destroyed`
4. **Polygon Aggregator**: Masks classification output with predicted building footprints and aggregates per-building polygon damage confidence distributions.

The root `train.py` uses normalized inverse-square-root weights computed from
the selected real training tiles' damage-pixel frequencies (capped at 10). It
shuffles tiles without weighted oversampling. Validation always uses the full
event-disjoint split unless an explicit `--max-val` smoke-test cap is requested;
each epoch logs localization F1, per-class damage precision/recall/F1, and
predicted- and ground-truth-building damage histograms. A class above 90% of
predicted building pixels after epoch 2 emits a warning but does not stop
training. The local 6 GB profile uses batch size 2 with gradient accumulation
2 (effective batch size 4).

The actual six-epoch real-xBD v3 run and its full validation metrics are
documented in the [mentor training report](../docs/xbd-v3-training-report.md).

---

## 📁 Directory Structure

```
ai-engine/
├── ai_engine/
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset for xBD pre/post pairs
│   │   ├── parser.py         # xBD JSON parser, GeoJSON/WKT polygon rasterization
│   │   ├── splits.py         # Disaster-event disjoint train/val/test splits
│   │   └── synthetic.py      # Synthetic xBD dataset generator for local dev & testing
│   ├── model/
│   │   ├── siamese_unet.py   # Dual-Task Siamese U-Net model definition
│   │   ├── losses.py         # Joint localization (BCE+Dice) & damage (Focal) loss
│   │   ├── metrics.py        # xView2 F1 macro competition metrics
│   │   ├── aggregator.py     # Polygon footprint damage aggregation
│   │   ├── sat_serving.py    # Satellite inference runner
│   │   └── hardware.py       # Hardware auto-detection (MPS/CUDA/CPU/RAM)
│   ├── training/
│   │   └── train.py          # Compatibility entry point for the root v3 trainer
│   └── serve.py              # FastAPI app: POST /infer/satellite and /infer/drone
├── scripts/
│   ├── download_xbd.py       # xBD dataset access guide & synthetic generator
│   ├── eda_xbd.py            # Comprehensive xBD Exploratory Data Analysis
├── notebooks/
│   └── eda.ipynb             # Interactive Jupyter EDA notebook with visualizations
└── tests/
    └── test_xbd_data.py      # Unit & integration tests for xBD parser, dataset & splits
```

---

## ⚡ Quick Start

### Train on a local real xBD dataset (Windows PowerShell)

The dataset root must contain `train\images`, `train\labels`, and optionally
`hold\images` and `hold\labels`. From this `ai-engine` directory:

```powershell
$XbdRoot = "D:\datasets\xbd"
$RunOutput = "runs\xbd-demo-$(Get-Date -Format yyyyMMdd-HHmmss)"
$ModelOutput = "data\models\xbd-demo-$(Get-Date -Format yyyyMMdd-HHmmss)"
$LogOutput = "runs\xbd-training-$(Get-Date -Format yyyyMMdd-HHmmss)"
python -m pip install -r requirements.txt
python -m ai_engine.data.check --xbd-root $XbdRoot --samples 5 --crosscheck 5 `
  --samples-dir "$RunOutput\overlays" --json-out "$RunOutput\data-check.json"
python scripts\eda_xbd.py --xbd-root $XbdRoot --subset 0.3 `
  --save-json "$RunOutput\eda.json"
python train.py --xbd-root $XbdRoot --config auto --epochs 30 --subset 1.0 `
  --output-dir $ModelOutput --log-dir $LogOutput
python eval.py --xbd-root $XbdRoot --checkpoint "$ModelOutput\checkpoint_best.pt" `
  --split val --output-dir "$ModelOutput\eval-val"
```

The primary reported score is from the event-disjoint validation subset of
`train/`. `hold/` contains tile-level holdout data and is supplemental; see the
[training guide](../docs/training.md) for running that evaluation, interpreting
results, and loading the trained model in Rubicon. Missing data paths stop the
run instead of silently substituting synthetic data.

### 1. Generate Synthetic Data
```powershell
$SyntheticOutput = "runs\synthetic-xbd-$(Get-Date -Format yyyyMMdd-HHmmss)"
python scripts\download_xbd.py --synthetic --num-pairs 10 --dest $SyntheticOutput
```

### 2. Run Exploratory Data Analysis (EDA)
```powershell
# Terminal summary
$XbdRoot = "D:\datasets\xbd"
$RunOutput = "runs\xbd-eda-$(Get-Date -Format yyyyMMdd-HHmmss)"
python scripts\eda_xbd.py --xbd-root $XbdRoot --subset 0.3 `
  --save-json "$RunOutput\eda.json"

# Or launch Jupyter notebook
jupyter lab notebooks/eda.ipynb
```

### 3. Run Tests
```bash
pytest tests/test_xbd_data.py
```

### 4. Start Inference Server
```bash
python -m uvicorn ai_engine.serve:app --host 127.0.0.1 --port 8002
```
- Health Check: `GET http://localhost:8002/health` (`model_loaded` should be `true`)
- Satellite API: `POST http://localhost:8002/infer/satellite` (multipart `pre` and `post`)

The xBD and drone inference endpoints run from `ai_engine.serve` on port 8002.