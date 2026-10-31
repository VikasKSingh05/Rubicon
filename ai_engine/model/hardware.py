"""Hardware auto-detection and environment profiling for Rubicon training and inference.

Detects GPU/VRAM/RAM/disk and configures:
- Local 6 GB profile (RTX 3050): ResNet34, 512 crops, batch 2, accum 4, AMP fp16, channels_last
- Kaggle profile (>=15 GB VRAM): ResNet34, 512 crops, batch 8, accum 1
- CPU profile (No GPU): ResNet18, 256 crops, batch 2, subset mode with warning
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, Any

try:
    import psutil
except ImportError:
    psutil = None
import torch


@dataclass
class HardwareProfile:
    name: str
    device: str
    encoder: str
    crop_size: int
    batch_size: int
    grad_accum_steps: int
    use_amp: bool
    channels_last: bool
    num_workers: int
    pin_memory: bool
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    notes: str = ""


def detect_hardware() -> HardwareProfile:
    """Auto-detect system hardware and return optimal training/inference profile."""
    # RAM and Disk
    if psutil is not None:
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    else:
        ram_gb = 16.0
    disk_free_gb = round(shutil.disk_usage(os.getcwd()).free / (1024 ** 3), 1)

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    print("=" * 65)
    print("RUBICON HARDWARE AUTO-DETECTION")
    print("=" * 65)
    print(f"  CPU Logical Cores: {os.cpu_count()}")
    print(f"  System RAM:        {ram_gb} GB")
    print(f"  Disk Space Free:   {disk_free_gb} GB")

    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(vram_bytes / (1024 ** 3), 1)
        print(f"  GPU Detected:      {gpu_name}")
        print(f"  VRAM Available:    {vram_gb} GB")
        print(f"  CUDA Version:      {torch.version.cuda}")

        if vram_gb >= 14.0:
            # Kaggle / Cloud High-VRAM Profile
            print("  Selected Profile:  KAGGLE HIGH-VRAM (>=15 GB)")
            print("  Configuration:     ResNet34 | 512x512 | Batch 8 | Accum 1 | AMP fp16")
            print("=" * 65)
            return HardwareProfile(
                name="kaggle_16gb",
                device="cuda",
                encoder="resnet34",
                crop_size=512,
                batch_size=8,
                grad_accum_steps=1,
                use_amp=True,
                channels_last=True,
                num_workers=4 if os.name != "nt" else 2,
                pin_memory=True,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                disk_free_gb=disk_free_gb,
                notes="Kaggle P100/T4 profile",
            )
        else:
            # Local 6 GB Profile (RTX 3050)
            print("  Selected Profile:  LOCAL 6GB VRAM (NVIDIA RTX 3050)")
            print("  Configuration:     ResNet34 | 512x512 | Batch 2 | Accum 4 (eff 8) | AMP fp16 | OOM-Recovery")
            print("=" * 65)
            return HardwareProfile(
                name="local_6gb",
                device="cuda",
                encoder="resnet34",
                crop_size=512,
                batch_size=2,
                grad_accum_steps=4,  # effective batch = 8
                use_amp=True,
                channels_last=True,
                num_workers=2 if os.name == "nt" else 4,
                pin_memory=True,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                disk_free_gb=disk_free_gb,
                notes="Local RTX 3050 6GB profile with dynamic OOM halving",
            )
    else:
        # CPU Fallback Profile
        print("  [WARN] No GPU/CUDA detected. Running in reduced CPU mode.")
        print("  Selected Profile:  CPU SUBSET")
        print("  Configuration:     ResNet18 | 256x256 | Batch 2 | Subset only | No AMP")
        print("=" * 65)
        return HardwareProfile(
            name="cpu_fallback",
            device="cpu",
            encoder="resnet18",
            crop_size=256,
            batch_size=2,
            grad_accum_steps=2,
            use_amp=False,
            channels_last=False,
            num_workers=0,  # avoid multiprocessing overhead on Windows CPU
            pin_memory=False,
            vram_gb=0.0,
            ram_gb=ram_gb,
            disk_free_gb=disk_free_gb,
            notes="CPU fallback mode for testing and lightweight inference",
        )
