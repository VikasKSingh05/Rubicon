# Rubicon AI Engine — xBD Dual-Task Siamese U-Net

This part of Rubicon contains the xBD Siamese U-Net for automated post-disaster
building damage assessment from pre- and post-disaster optical imagery. It is
the satellite model; the repository's HSI/LiDAR Mamba-Transformer pipeline is
a separate model and uses separate artifacts.

## 🏗️ Model Architecture

The architecture implements a **Dual-Task Siamese U-Net** based on the research specification:
1. **Weight-Shared Siamese Encoders**: A shared convolutional backbone (ResNet-18 / ResNet-34) processes the pre-disaster and post-disaster images in parallel to extract multiscale feature representations:
   - $F_{\text{pre}} = \text{Encoder}(I_{\text{pre}})$
   - $F_{\text{post}} = \text{Encoder}(I_{\text{post}})$
2. **Feature Fusion Layer**: Fuses pre and post feature maps via concatenation and absolute difference:
   - $F_{\text{fused}} = [F_{\text{pre}} \,\|\, F_{\text{post}} \,\|\, |F_{\text{post}} - F_{\text{pre}}|]$
3. **Dual-Task Decoders**:
   - **Localization Decoder**: Predicts building footprints (Binary Cross-Entropy + Dice Loss).
   - **Classification Decoder**: Predicts five pixel classes (weighted Cross-Entropy + Focal Loss):
     - `0: Background`
     - `1: No damage`
     - `2: Minor damage`
     - `3: Major damage`
     - `4: Destroyed`
4. **Polygon Aggregator**: Masks classification output with predicted building footprints and aggregates per-building polygon damage confidence distributions.

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
│   │   └── train.py          # Training loop with AMP, checkpointing & logging
│   └── serve.py              # FastAPI satellite app, including POST /infer/satellite
├── src/                      # Separate legacy Houston HSI/LiDAR experiment (/predict)
├── scripts/
│   ├── download_xbd.py       # xBD dataset access guide & synthetic generator
│   ├── eda_xbd.py            # Comprehensive xBD Exploratory Data Analysis
│   └── tune_hyperparams.py   # Hyperparameter tuning search for Siamese U-Net
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
python -m pip install -r requirements.txt
python -m ai_engine.data.check --xbd-root $XbdRoot --samples 5 --crosscheck 5
python train.py --xbd-root $XbdRoot --config auto --epochs 30 --subset 1.0 `
  --output-dir data/models/xbd --log-dir runs/xbd-mentor-demo
python eval.py --xbd-root $XbdRoot --checkpoint data/models/xbd/checkpoint_best.pt `
  --split val --output-dir data/models/xbd/eval-val
```

The primary reported score is from the event-disjoint validation subset of
`train/`. `hold/` contains tile-level holdout data and is supplemental; see the
[training guide](../docs/training.md) for running that evaluation, interpreting
results, and loading the trained model in Rubicon. Missing data paths stop the
run instead of silently substituting synthetic data.

### 1. Generate Synthetic Data
```bash
python scripts/download_xbd.py --synthetic --num-pairs 10
```

### 2. Run Exploratory Data Analysis (EDA)
```bash
# Terminal summary
python scripts/eda_xbd.py --synthetic

# Or launch Jupyter notebook
jupyter lab notebooks/eda.ipynb
```

### 3. Hyperparameter Tuning
```bash
# Fast dry-run
python scripts/tune_hyperparams.py --synthetic --trials 5 --dry-run
```

### 4. Run Tests
```bash
pytest tests/test_xbd_data.py
```

### 5. Start Inference Server
```bash
python -m uvicorn ai_engine.serve:app --host 127.0.0.1 --port 8002
```
- Health Check: `GET http://localhost:8002/health` (`model_loaded` should be `true`)
- Satellite API: `POST http://localhost:8002/infer/satellite` (multipart `pre` and `post`)

The separate legacy HSI/LiDAR app is served from `src.inference` on port 8000.
It is not the xBD Siamese U-Net and has independent model artifacts.