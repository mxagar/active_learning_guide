from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Sequence
import random
from tqdm import tqdm

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    # Core
    num_classes: int
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True

    # Optimization
    optimizer: str = "adamw"  # "adam" | "sgd" | "adamw"
    momentum: float = 0.9  # for SGD
    scheduler: Optional[str] = None  # None | "step" | "cosine"
    step_size: int = 10
    gamma: float = 0.1
    cosine_tmax: Optional[int] = None

    # Logging / checkpointing
    out_dir: str = "runs/flowers_cnn"
    run_name: str = "exp"
    metric_for_best: str = "val_acc"  # "val_acc" | "val_loss" | "val_f1"
    maximize_metric: bool = True  # True for accuracy, False for loss

    # Reproducibility
    seed: int = 42

    # Optional: gradient clipping
    grad_clip_norm: Optional[float] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    """
    A small CNN for 64x64 RGB images.

    In forward:
    - If feature_vector=False (default): returns logits (B, num_classes)
    - If feature_vector=True: returns feature vector (B, 256) from penultimate layer
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 32,  # number of channels @ 1st conv layer, 2x & 4x later
        classifier_hidden: int = 256,  # hidden dim of classifier head (feature vector size)
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 8 * 8, classifier_hidden),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        feat = self.projection(x)  # (B, classifier_hidden)
        logits = self.classifier(feat)  # (B, num_classes)

        if return_embeddings:
            return logits, feat

        return logits

    @property
    def num_classes(self) -> int:
        return int(self.classifier[-1].out_features)

    @property
    def num_features(self) -> int:
        return int(self.classifier[-1].in_features)


def save_model(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    cfg: Optional[TrainConfig] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    if cfg is not None:
        ckpt["config"] = asdict(cfg)

    torch.save(ckpt, path)


def load_model(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str] = None,
) -> dict[str, Any]:
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    return ckpt


def _is_better(new: float, best: float, maximize: bool) -> bool:
    return new > best if maximize else new < best


@torch.no_grad()
def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == y).sum().item()
    return correct / max(1, y.numel())


def _build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = cfg.optimizer.lower()
    if opt == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    if cfg.scheduler is None:
        return None
    s = cfg.scheduler.lower()
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if s == "cosine":
        tmax = cfg.cosine_tmax if cfg.cosine_tmax is not None else cfg.epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


@torch.no_grad()
def _update_confusion_matrix(cm: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    cm: (K, K) where rows=true, cols=pred
    y_true, y_pred: (B,)
    """
    k = cm.size(0)
    idx = y_true * k + y_pred
    cm.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=cm.dtype))
    return cm


@torch.no_grad()
def _macro_f1_from_cm(cm: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Macro-F1 over classes, ignoring classes with 0 support (optional: keep them as 0).
    """
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # Average over classes present in ground truth
    support = cm.sum(dim=1)
    mask = support > 0
    if mask.any():
        return f1[mask].mean().item()
    return f1.mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    grad_clip_norm: Optional[float] = None,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)

    for x, y in tqdm(loader, desc="Training...", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        preds = logits.argmax(dim=1)

        bs = y.size(0)
        running_loss += loss.item() * bs
        running_correct += (preds == y).sum().item()
        total += bs

        _update_confusion_matrix(cm, y, preds)

    train_f1 = _macro_f1_from_cm(cm)

    return {
        "train_loss": running_loss / max(1, total),
        "train_acc": running_correct / max(1, total),
        "train_f1": train_f1,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    prefix: str = "val",
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)

    for x, y in tqdm(loader, desc="Evaluating...", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=1)

        bs = y.size(0)
        running_loss += loss.item() * bs
        running_correct += (preds == y).sum().item()
        total += bs

        _update_confusion_matrix(cm, y, preds)

    f1 = _macro_f1_from_cm(cm)

    return {
        f"{prefix}_loss": running_loss / max(1, total),
        f"{prefix}_acc": running_correct / max(1, total),
        f"{prefix}_f1": f1,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    # Same as validate, but uses "test" prefix by convention
    return validate(model, loader, device, num_classes=num_classes, prefix="test")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
) -> list[dict[str, float]]:
    """
    Full training loop. Saves:
    - last checkpoint each epoch
    - best checkpoint (based on cfg.metric_for_best)

    Returns:
        history: list of dicts with metrics per epoch
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = run_dir / "last.pt"
    best_ckpt_path = run_dir / "best.pt"

    # Init best metric
    best_metric = -float("inf") if cfg.maximize_metric else float("inf")

    history: list[dict[str, float]] = []

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Training epochs"):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_classes=cfg.num_classes,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=cfg.num_classes,
            prefix="val",
        )

        # LR logging
        lr = optimizer.param_groups[0]["lr"]
        metrics = {"epoch": epoch, "lr": lr, **train_metrics, **val_metrics}
        history.append(metrics)

        # Save "last"
        save_model(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            cfg=cfg,
        )

        # Save "best"
        current = metrics.get(cfg.metric_for_best)
        if current is None:
            raise KeyError(
                f"metric_for_best='{cfg.metric_for_best}' not found in metrics keys: {list(metrics.keys())}"
            )

        if _is_better(float(current), float(best_metric), maximize=cfg.maximize_metric):
            best_metric = float(current)
            save_model(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                cfg=cfg,
            )

        if scheduler is not None:
            scheduler.step()

        # Console log (now includes F1)
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={metrics['train_loss']:.4f} train_acc={metrics['train_acc']:.3f} train_f1={metrics['train_f1']:.3f} | "
            f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']:.3f} val_f1={metrics['val_f1']:.3f} | "
            f"lr={lr:.2e}"
        )

    return history


def plot_history(
    history: list[dict[str, float]],
    keys: Optional[list[str]] = None,
    title: str = "Training history",
) -> None:
    """
    Plot metrics over epochs.

    keys: which metrics to plot. If None, tries common ones.
    """
    if not history:
        raise ValueError("Empty history")

    epochs = [h["epoch"] for h in history]

    if keys is None:
        candidates = [
            "train_loss", "val_loss",
            "train_acc", "val_acc",
            "train_f1", "val_f1",
        ]
        keys = [k for k in candidates if k in history[0]]

    loss_keys = [k for k in keys if k.endswith("_loss")]
    other_keys = [k for k in keys if not k.endswith("_loss")]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Plot losses (left axis)
    for k in loss_keys:
        ys = [h.get(k, float("nan")) for h in history]
        ax1.plot(epochs, ys, label=k, linestyle="-")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Plot metrics (right axis)
    for k in other_keys:
        ys = [h.get(k, float("nan")) for h in history]
        ax2.plot(epochs, ys, label=k, linestyle="--")

    ax2.set_ylabel("Score")

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(title)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    batch: torch.Tensor | Sequence,
    device: Optional[str | torch.device] = None,
    class_names: Optional[Sequence[str]] = None,
) -> list[int]:
    """
    Predict class labels for a batch tensor.

    Args:
        model: torch model returning logits (B, num_classes)
        batch: input tensor (B, C, H, W) or a sequence like [batch_x, batch_y]
        device: if provided, moves batch to this device; otherwise uses model's device
        class_names: optional mapping from label index to class name;
            if provided, returns list of class names instead of indices

    Returns:
        list[int]: predicted class indices
    """
    model.eval()

    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
        batch_x = batch[0]
    else:
        batch_x = batch

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    x = batch_x.to(device, non_blocking=True)
    logits = model(x)
    preds = logits.argmax(dim=1).detach().cpu().tolist()

    if class_names is not None:
        return [class_names[p] for p in preds]
    return preds


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image: str | Path | Image.Image,
    transform,  # torchvision transform (must output normalized tensor CxHxW)
    device: Optional[str | torch.device] = None,
    class_names: Optional[Sequence[str]] = None,
) -> int | str:
    """
    Predict label for a single image.

    Args:
        model: torch model returning logits (B, num_classes)
        image: file path or PIL.Image
        transform: same eval transform used for val/test (Resize->ToTensor->Normalize)
        device: device to run on (optional)
        class_names: optional mapping label -> name

    Returns:
        int label index, or str class name if class_names provided
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            im = im.convert("RGB")
    else:
        im = image.convert("RGB")

    x = transform(im)          # (C, H, W)
    x = x.unsqueeze(0)         # (1, C, H, W)

    pred_idx = predict(model, x, device=device)[0]

    if class_names is not None:
        return class_names[pred_idx]

    return pred_idx
