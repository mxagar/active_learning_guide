from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
from collections import defaultdict
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
        classes (list of str): list of class names corresponding to subfolder names.
        class_to_idx (dict): mapping from class name to label index.
    
    Raises:
        RuntimeError: if no class subfolders or no images are found under the root folder.
    """
    root = Path(root)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"No class subfolders found under: {root}")

    class_to_idx = {cls_: i for i, cls_ in enumerate(classes)}

    paths = []
    labels = []
    for cls_ in classes:
        cls_dir = root / cls_
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(p)
                labels.append(class_to_idx[cls_])

    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    
    return paths, labels, classes, class_to_idx


def train_test_val_split(
    labels: list[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    train_ratio: float = None,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Split indices into train/val/test sets, stratified by class labels.

    IMPORTANT: All ratios are applied to the FULL dataset size per class.
        n_train = round(n_class * train_ratio)
        n_val   = round(n_class * val_ratio)
        n_test  = round(n_class * test_ratio)

    pool_idx is always the rest:
        pool = remaining indices after taking train/val/test

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

    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)

    train_idx, val_idx, test_idx, pool_idx = [], [], [], []

    for y, idxs in by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)

        n = len(idxs)
        n_train = int(round(n * train_ratio))
        n_val   = int(round(n * val_ratio))
        n_test  = int(round(n * test_ratio))

        # Just in case rounding pushes us over n, clip deterministically.
        # Excess gets removed from train, then val, then test (pool stays the remainder).
        total = n_train + n_val + n_test
        if total > n:
            overflow = total - n
            # reduce train first
            take = min(overflow, n_train)
            n_train -= take
            overflow -= take
            # then val
            if overflow > 0:
                take = min(overflow, n_val)
                n_val -= take
                overflow -= take
            # then test
            if overflow > 0:
                take = min(overflow, n_test)
                n_test -= take
                overflow -= take

        # Assign slices (order doesn't matter; keep it consistent)
        train = idxs[:n_train]
        val   = idxs[n_train:n_train + n_val]
        test  = idxs[n_train + n_val:n_train + n_val + n_test]
        pool  = idxs[n_train + n_val + n_test:]

        train_idx.extend(train)
        val_idx.extend(val)
        test_idx.extend(test)
        pool_idx.extend(pool)

    # Shuffle within each split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    rng.shuffle(pool_idx)

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

    def __init__(self, paths, labels, indices, transform=None):
        """
        Args:
            paths: List[Path] or List[str]
            labels: List[int]
            indices: List[int] (subset indices)
            transform: torchvision transforms
        """
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
