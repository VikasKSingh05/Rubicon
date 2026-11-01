# Rubicon ai-engine — xBD Dual-Task Siamese U-Net

The Rubicon AI Engine is a deep learning service for automated post-disaster building damage assessment using pre- and post-disaster high-resolution optical satellite imagery from the **xBD (xView2) dataset**.

## 🏗️ Model Architecture

The architecture implements a **Dual-Task Siamese U-Net** based on the research specification:
1. **Weight-Shared Siamese Encoders**: A shared convolutional backbone (ResNet-18 / ResNet-34) processes the pre-disaster and post-disaster images in parallel to extract multiscale feature representations:
   - $F_{\text{pre}} = \text{Encoder}(I_{\text{pre}})$
   - $F_{\text{post}} = \text{Encoder}(I_{\text{post}})$
2. **Feature Fusion Layer**: Fuses pre and post feature maps via concatenation and absolute difference:
   - $F_{\text{fused}} = [F_{\text{pre}} \,\|\, F_{\text{post}} \,\|\, |F_{\text{post}} - F_{\text{pre}}|]$
3. **Dual-Task Decoders**:
   - **Localization Decoder**: Segments all building footprints from pre-disaster features (Binary Cross-Entropy + Dice Loss).
   - **Classification Decoder**: Classifies 4-tier damage severity on post-disaster features (Weighted Cross-Entropy / Focal Loss):
     - `0: No Damage`
     - `1: Minor Damage`
     - `2: Major Damage`
     - `3: Destroyed`
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
│   └── serve.py              # FastAPI server implementing the `/predict` contract
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
python -m uvicorn ai_engine.serve:app --host 0.0.0.0 --port 8000
```
- Health Check: `GET http://localhost:8000/health`
- Predict API: `POST http://localhost:8000/predict` (multipart `pre_image` and `post_image`)