import math
from pathlib import Path
import random
from collections import defaultdict
from typing import Optional, Sequence

from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def resize_min_side(img: Image.Image, min_side: int = 64) -> Image.Image:
    """Resize so that min(width, height) == min_side, keeping aspect ratio."""
    w, h = img.size
    if min(w, h) == min_side:
        return img

    scale = min_side / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # High-quality down/up-sampling
    return img.resize((new_w, new_h), resample=Image.LANCZOS)


def build_resized_image_folder(
    src_root: str | Path,
    dst_root: str | Path,
    min_side: int = 64,
    overwrite: bool = False,
) -> None:
    """Build a resized image folder by resizing all images, keeping directory structure.

    Example usage:
    
        # Assuming our structure is like: dataset_root/train/class_x/*.jpg
        dataset_root = Path("/path/to/flowers")  # <- change this
        build_resized_image_folder(
            src_root=dataset_root / "train",
            dst_root=dataset_root / "train_64",
            min_side=64,
            overwrite=False,
        )

    Args:
        src_root (str | Path): folder to read images from; will be searched recursively.
        dst_root (str | Path): folder to write resized images to; directory structure will be preserved.
        min_side (int, optional): minimum size of the smaller image side after resizing.
            Defaults to 64.
        overwrite (bool, optional): whether to overwrite existing images in the destination folder.
            Defaults to False.

    Raises:
        FileNotFoundError: _description_
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if not src_root.exists():
        raise FileNotFoundError(f"Source folder not found: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_written = 0

    for src_path in tqdm(list(src_root.rglob("*")), desc="Processing images"):
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in IMG_EXTS:
            continue

        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not overwrite:
            n_total += 1
            continue

        try:
            with Image.open(src_path) as im:
                im = im.convert("RGB")  # ensure consistent 3-channel
                im_resized = resize_min_side(im, min_side=min_side)
                # Keep format by saving to same extension; JPEG uses quality
                save_kwargs = {}
                if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
                    save_kwargs = {"quality": 95, "optimize": True}
                im_resized.save(dst_path, **save_kwargs)

            n_written += 1
        except Exception as e:
            print(f"[WARN] Failed on {src_path}: {e}")

        n_total += 1

    print(f"Done. Processed {n_total} images. Wrote {n_written} resized images to: {dst_root}")


def build_paths_and_labels(root: str | Path):
    """
    Create a list of image paths and corresponding labels based on subfolder names.
    
    Args:
        root (str | Path): folder containing class subfolders with images.
        
    Returns:
        paths (list of Path): list of image file paths.
        labels (list of int): list of integer labels corresponding to class subfolders.
        classe_names (list of str): list of class names corresponding to subfolder names.
        class_to_id (dict): mapping from class name to label index.
    
    Raises:
        RuntimeError: if no class subfolders or no images are found under the root folder.
    """
    root = Path(root)
    classe_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not classe_names:
        raise RuntimeError(f"No class subfolders found under: {root}")

    class_to_id = {cls_: i for i, cls_ in enumerate(classe_names)}

    paths = []
    labels = []
    for cls_ in classe_names:
        cls_dir = root / cls_
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(p)
                labels.append(class_to_id[cls_])

    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    
    return paths, labels, classe_names, class_to_id


def train_test_val_pool_split(
    labels: list[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    train_ratio: float = None,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Split indices into train/val/test/pool sets, stratified by class labels.

    Split order per class (IMPORTANT for stability across train_ratio changes):
        1) test
        2) val
        3) train
        4) pool (the rest)

    Ratios are applied w.r.t. the FULL dataset size per class.

    If train_ratio is None:
        train_ratio = 1.0 - val_ratio - test_ratio  (so pool is empty)

    Args:
        labels (list[str]): list of class labels for each data point.
        val_ratio (float, optional): proportion of data to use for validation. Defaults to 0.1.
        test_ratio (float, optional): proportion of data to use for testing. Defaults to 0.1.
        train_ratio (float, optional): proportion of data to use for training. If None, it is set to 1 - val_ratio - test_ratio. Defaults to None.
        seed (int, optional): random seed for reproducibility. Defaults to 42.

    Returns:
        train_idx (list[int]): list of indices for training set.
        val_idx (list[int]): list of indices for validation set.
        test_idx (list[int]): list of indices for test set.
        pool_idx (list[int]): list of indices for pool set.

    Raises:
        ValueError: if any of the ratios are negative or if their sum exceeds 1.0.
    """
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be >= 0.")
    if train_ratio is None:
        train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio < 0:
        raise ValueError("train_ratio must be >= 0.")
    if train_ratio + val_ratio + test_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be <= 1.0")

    rng = random.Random(seed)

    by_class: dict[str, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)

    train_idx, pool_idx, val_idx, test_idx = [], [], [], []

    for y, idxs in by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)

        n = len(idxs)

        # Allocate TEST first, then VAL: these stay stable if seed/ratios stay the same
        n_test = int(round(n * test_ratio))
        n_val  = int(round(n * val_ratio))

        # Clamp in case rounding makes n_test + n_val > n
        if n_test + n_val > n:
            overflow = (n_test + n_val) - n
            # reduce val first, then test
            take = min(overflow, n_val)
            n_val -= take
            overflow -= take
            if overflow > 0:
                n_test -= min(overflow, n_test)

        test = idxs[:n_test]
        val  = idxs[n_test:n_test + n_val]

        remaining = idxs[n_test + n_val:]  # candidates for train/pool only

        # Allocate TRAIN from the remaining, using train_ratio w.r.t. full n, but capped by remaining
        n_train_target = int(round(n * train_ratio))
        n_train = min(n_train_target, len(remaining))

        train = remaining[:n_train]
        pool  = remaining[n_train:]

        test_idx.extend(test)
        val_idx.extend(val)
        train_idx.extend(train)
        pool_idx.extend(pool)

    # Shuffle within each split (optional, but usually nice)
    rng.shuffle(train_idx)
    rng.shuffle(pool_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx, pool_idx


class CustomDataset(Dataset):
    """
    Dataset based on:
    - full paths list
    - full labels list
    - subset indices (train / val / test / pool)

    This is ideal for active learning because:
    - We never duplicate data
    - We can dynamically grow/shrink subsets
    """

    def __init__(
        self,
        paths: list[Path|str],
        labels: list[str],
        indices: list[int],
        transform: T.Compose = None,
    ) -> None:
        if not len(paths) == len(labels):
            raise ValueError("paths and labels must have the same length")
        if len(indices) > len(paths):
            raise ValueError("indices length cannot exceed paths/labels length")

        self.paths = paths
        self.labels = labels
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path = self.paths[real_idx]
        label = self.labels[real_idx]

        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def unnormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, H, W) normalized with ImageNet stats
    returns: (B, C, H, W) in [0, 1] approx
    """
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1, 3, 1, 1)
    return x * std + mean


@torch.no_grad()
def visualize_batch(
    batch: Sequence | torch.Tensor,
    class_names: Optional[list[str]] = None,
    max_images: int = 16,
    nrow: int = 4,
    figsize: tuple[int,int] = (10, 10),
    title: Optional[str] = None,
    font_size: int = 10,
) -> None:
    """
    Plot a batch of images in a grid, with optional labels below.

    Args:
        batch: either (images, labels) or images
            images: Tensor (B, C, H, W) normalized
            labels: Tensor (B,) optional
        class_names: list[str] optional, maps label -> name
        max_images: int, maximum number of images to display (will take first N)
        nrow: int, number of images per row in the grid
        figsize: tuple, size of the matplotlib figure
        title: optional title for the plot
        font_size: int, font size for labels
    """
    if torch.is_tensor(batch):
        images = batch
        labels = None
    elif isinstance(batch, Sequence) and len(batch) >= 1 and torch.is_tensor(batch[0]):
        images = batch[0]
        labels = batch[1] if len(batch) > 1 and torch.is_tensor(batch[1]) else None
    else:
        raise TypeError(
            "batch must be either a Tensor (B,C,H,W) or a sequence like (images, labels)."
        )

    images = images[:max_images].detach().cpu()
    if labels is not None:
        labels = labels[:max_images].detach().cpu()

    # Unnormalize + clamp
    images_vis = unnormalize_imagenet(images).clamp(0.0, 1.0)

    B, C, H, W = images_vis.shape
    ncol = nrow
    nrows = math.ceil(B / ncol)

    fig, axes = plt.subplots(nrows, ncol, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    if title:
        fig.suptitle(title)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= B:
            continue

        img = images_vis[i].permute(1, 2, 0).numpy()
        ax.imshow(img)

        if labels is not None:
            lab_idx = int(labels[i])
            if class_names is not None:
                lab = class_names[lab_idx] if 0 <= lab_idx < len(class_names) else str(lab_idx)
            else:
                lab = str(lab_idx)

            ax.text(
                0.5,
                -0.10,
                lab,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=font_size,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.90)
    plt.show()
