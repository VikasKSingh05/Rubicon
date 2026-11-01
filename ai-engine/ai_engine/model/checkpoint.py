"""Single checkpoint contract shared by training, evaluation, and serving.

A Rubicon checkpoint is a plain dict that can be loaded with
``torch.load(..., weights_only=True)``.  It contains:

* ``format_version``      - schema version (2 for the current format)
* ``state_dict``          - model weights
* ``config``              - encoder, crop size, model version, hardware profile
* ``norm_mean``/``norm_std`` - RGB normalisation used during training
* ``damage_classes``      - ordered class names for the 5 damage channels
* ``model_version``       - human-readable version string
* ``state_dict_hash``     - short sha256 over the weights for provenance
* optional training state - ``epoch``, ``optimizer_state_dict``,
  ``scheduler_state_dict``, ``scaler_state_dict``, ``metrics``

Legacy checkpoints that stored weights under ``model_state_dict`` are still
accepted by :func:`load_checkpoint`, so the earlier training runs keep working.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

CHECKPOINT_FORMAT_VERSION = 2
DAMAGE_CLASS_NAMES = [
    "background",
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_WEIGHT_KEYS = ("state_dict", "model_state_dict")


def _to_builtin(value: Any) -> Any:
    """Recursively convert numpy/torch scalars so weights_only loading works."""
    if isinstance(value, Mapping):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # numpy scalar / array and anything else exotic
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, RuntimeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _to_builtin(tolist())
    return str(value)


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Return a short, deterministic sha256 over every tensor in *state_dict*."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        digest.update(key.encode("utf-8"))
        if torch.is_tensor(tensor):
            cpu = tensor.detach().to("cpu").contiguous()
            digest.update(str(cpu.dtype).encode("utf-8"))
            digest.update(str(tuple(cpu.shape)).encode("utf-8"))
            digest.update(cpu.numpy().tobytes())
        else:
            digest.update(repr(tensor).encode("utf-8"))
    return digest.hexdigest()[:16]


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    model_version: str = "siamese_unet-v1",
    encoder_name: str = "resnet34",
    num_classes: int = 5,
    crop_size: int = 512,
    norm_mean: Sequence[float] | None = None,
    norm_std: Sequence[float] | None = None,
    damage_classes: Sequence[str] | None = None,
    epoch: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    hardware_profile: str | None = None,
    extra_config: Mapping[str, Any] | None = None,
) -> dict:
    """Write a checkpoint and return the payload that was saved."""
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    config = {
        "encoder": encoder_name,
        "encoder_name": encoder_name,
        "num_classes": num_classes,
        "crop_size": crop_size,
        "model_version": model_version,
        "hardware_profile": hardware_profile,
    }
    if epoch is not None:
        config["epochs_trained"] = epoch + 1
    if extra_config:
        config.update({str(k): _to_builtin(v) for k, v in extra_config.items()})

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_version": model_version,
        "encoder_name": encoder_name,
        "num_classes": num_classes,
        "damage_classes": list(damage_classes or DAMAGE_CLASS_NAMES),
        "config": config,
        "norm_mean": list(norm_mean or IMAGENET_MEAN),
        "norm_std": list(norm_std or IMAGENET_STD),
        "state_dict": state,
        "state_dict_hash": state_dict_hash(state),
    }
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if metrics is not None:
        payload["metrics"] = _to_builtin(metrics)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    *,
    expected_num_classes: int | None = None,
) -> dict:
    """Load a checkpoint safely, upgrading legacy keys to the current contract."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(str(path), map_location=map_location, weights_only=True)

    state = None
    for key in _WEIGHT_KEYS:
        if key in checkpoint:
            state = checkpoint[key]
            break
    if state is None:
        raise KeyError(
            f"{path} does not contain model weights "
            f"(expected one of {_WEIGHT_KEYS})"
        )

    config = dict(checkpoint.get("config") or {})
    checkpoint["state_dict"] = state
    checkpoint.setdefault("format_version", 1)
    checkpoint.setdefault(
        "model_version",
        config.get("model_version", "siamese_unet-v1"),
    )
    checkpoint.setdefault(
        "encoder_name",
        checkpoint.get("encoder_name") or config.get("encoder") or "resnet34",
    )
    checkpoint.setdefault(
        "num_classes",
        checkpoint.get("num_classes") or config.get("num_classes") or 5,
    )
    checkpoint.setdefault("config", config)
    checkpoint.setdefault("norm_mean", list(IMAGENET_MEAN))
    checkpoint.setdefault("norm_std", list(IMAGENET_STD))
    checkpoint.setdefault("damage_classes", list(DAMAGE_CLASS_NAMES))

    if expected_num_classes is not None and checkpoint["num_classes"] != expected_num_classes:
        raise ValueError(
            f"Checkpoint {path} has {checkpoint['num_classes']} classes, "
            f"expected {expected_num_classes}"
        )
    return checkpoint


def norm_stats_from_checkpoint(checkpoint: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """Return ``(mean, std)`` RGB triples stored in *checkpoint*."""
    return (
        [float(v) for v in checkpoint.get("norm_mean", IMAGENET_MEAN)],
        [float(v) for v in checkpoint.get("norm_std", IMAGENET_STD)],
    )


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Instantiate the Siamese U-Net described by *checkpoint* and load weights."""
    from .siamese_unet import SiameseUNet

    model = SiameseUNet(
        encoder_name=checkpoint.get("encoder_name", "resnet34"),
        encoder_weights=None,
        num_classes=int(checkpoint.get("num_classes", 5)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model.to(device)
