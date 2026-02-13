from typing import Optional, Literal

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from skactiveml.base import SkactivemlClassifier
from skactiveml.pool import RandomSampling, UncertaintySampling, Badge


MISSING_LABEL_INT = -1
SearchStrategy = Literal[
    "random",
    "least_confident",
    "margin_sampling",
    "entropy",
    "badge",
]


class TorchClassifierWrapper(SkactivemlClassifier):
    """
    Adapter that makes your PyTorch model compatible with scikit-activeml querying.

    Interpretation: X is treated as indices into pool_ds (not real features).

    Provides:
    - predict_proba(X): returns class probabilities for those indices
    - compute_features(X): returns embedding vectors for those indices
    """
    def __init__(
        self,
        model: torch.nn.Module,
        pool_ds,
        batch_size: int = 16,
        missing_label=MISSING_LABEL_INT,
        classes=None,
        device=None,
        num_workers: int = 0,
        pin_memory: bool = False,
        random_state=None,
        verbose: bool = False,
    ):
        self.model = model
        self.pool_ds = pool_ds
        self.batch_size = batch_size
        self.device = torch.device(device if device is not None else next(model.parameters()).device)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose
        self.num_classes = self.model.num_classes
        self.num_features = self.model.num_features
        if classes is not None:
            self.classes_ = classes
        else:
            self.classes_ = np.arange(self.num_classes)
        super().__init__(classes=self.classes_, missing_label=missing_label, random_state=random_state)

    def _to_indices(self, X: np.ndarray | list[int]) -> list[int]:
        idx = np.asarray(X).reshape(-1).astype(int)
        # basic sanity check (optional)
        if len(idx) > 0 and (idx.min() < 0 or idx.max() >= len(self.pool_ds)):
            raise IndexError(f"Some indices in X are out of bounds for pool_ds. X range: [{idx.min()}, {idx.max()}], pool_ds length: {len(self.pool_ds)}")
        return idx.tolist()

    def _prepare_loader(self, X: np.ndarray | list[int]) -> DataLoader:
        idx = self._to_indices(X)        
        subset = Subset(self.pool_ds, idx)
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        return loader

    @torch.no_grad()
    def predict_proba(
        self,
        X: np.ndarray | list[int],
        return_embeddings: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Args:
            X: array-like of indices into pool_ds, shape (n,) or (n,1)
        Returns:
            probs: (n, num_classes) numpy float32 if return_embeddings=False
            (probs, feats): tuple of (n, num_classes) and (n, feature_dim) numpy float32
                if return_embeddings=True
        """
        if len(X) == 0:
            probas = np.empty((0, self.num_classes), dtype=np.float32)
            if return_embeddings:
                emb = np.empty((0, self.num_features), dtype=np.float32)
                return probas, emb
            return probas

        loader = self._prepare_loader(X)
        self.model.eval()
        probs_all, feats_all = [], []

        for batch in tqdm(loader, disable=not self.verbose, desc="predict_proba"):
            x = batch[0].to(self.device, non_blocking=True)
            output = self.model(x, return_embeddings=return_embeddings)
            logits, feats = None, None
            if return_embeddings:
                logits, feats = output  # (B, num_classes), (B, num_features)
                feats_all.append(feats.detach().cpu().numpy().astype(np.float32))
            else:
                logits = output  # (B, num_classes)

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
            probs_all.append(probs)

        probas = np.concatenate(probs_all, axis=0)

        if return_embeddings:
            embs = np.concatenate(feats_all, axis=0) if len(feats_all) else np.empty((0, self.num_features), np.float32)
            return probas, embs

        return probas

    @torch.no_grad()
    def compute_embeddings(self, X: np.ndarray | list[int]) -> np.ndarray:
        """
        Compute feature vectors using model(x, return_embeddings=True).

        Args:
            X: array-like of indices into pool_ds, shape (n,) or (n,1)
        Returns:
            feats: (n, D) numpy float32 (D = classifier_hidden, e.g. 256)
        """
        return self.predict_proba(X, return_embeddings=True)[1]

    # Keep sklearn-ish signature compatibility
    # SkactivemlClassifier requires fit to accept sample_weight
    def fit(self, X, y=None, sample_weight=None):
        return self


def compute_next_candidates(
    model,
    pool_ds,
    query_size: int,
    method: SearchStrategy = "entropy",
    seed: int = 42,
    batch_size: int = 128,
    classes: Optional[np.ndarray] = None,
    missing_label: int = -1,
    device=None,
    num_workers: int = 0,
    pin_memory: bool = True,
    verbose: bool = False,
) -> list[int]:
    """
    Use scikit-activeml to select the next samples to label from pool_ds.

    Returns:
        candidates_idx: list of indices *into pool_ds* (pool-local indices).
    """
    if method not in {"random", "least_confident", "margin_sampling", "entropy", "badge", "coreset"}:
        raise ValueError(f"Unsupported method: {method}")

    n_pool = len(pool_ds)
    if n_pool == 0:
        return []

    # X: dummy "feature matrix" = pool indices (n_pool, 1)
    X = np.arange(n_pool, dtype=int).reshape(-1, 1)

    # y: all missing labels (assume pool is unlabeled)
    y = np.full(n_pool, missing_label, dtype=int)

    # clf: wrapper provides predict_proba(X)
    clf = TorchClassifierWrapper(
        model=model,
        pool_ds=pool_ds,
        batch_size=batch_size,
        classes=classes,
        missing_label=missing_label,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
        random_state=seed,
        verbose=verbose,
    )

    query_strategy = None
    candidates_idx = []
    k = min(query_size, n_pool)

    if method == "random":
        query_strategy = RandomSampling(
            missing_label=missing_label,
            random_state=seed
        )
        candidates_idx_array = query_strategy.query(X=X, y=y, batch_size=k, candidates=None)
        candidates_idx = candidates_idx_array.tolist()

    if method in {"least_confident", "margin_sampling", "entropy"}:
        query_strategy = UncertaintySampling(
            method=method,
            missing_label=missing_label,
            random_state=seed,
        )
        # Important: pass candidates=None to query from all unlabeled samples
        candidates_idx_array = query_strategy.query(
            X=X,
            y=y,
            clf=clf,
            batch_size=k,
            candidates=None,  # Explicitly set to query all samples marked as missing_label
            fit_clf=False,  # Do NOT train any sklearn classifier
        )
        candidates_idx = candidates_idx_array.tolist()

    elif method == "badge":
        query_strategy = Badge(
            clf_embedding_flag_name="return_embeddings",  # Matches your predict_proba arg
            missing_label=missing_label,
            random_state=seed,
        )
        candidates_idx = query_strategy.query(X=X, y=y, clf=clf, batch_size=k, candidates=None, fit_clf=False)
        candidates_idx = [int(idx) for idx in candidates_idx]

    # WARNING: candidates_idx are pool-local indices (0 to len(pool_ds)-1),
    # NOT global indices into the original dataset!
    return candidates_idx


def transfer_candidates_idx(
    train_idx: list[int],
    pool_idx: list[int],
    candidates_idx: np.ndarray | list[int],
) -> tuple[list[int], list[int], list[int]]:
    """
    Transfer selected pool-local indices from pool_idx to train_idx.

    Args:
        train_idx: GLOBAL indices currently labeled
        pool_idx: GLOBAL indices currently in pool (unlabeled)
        candidates_idx: indices into pool_ds (i.e., LOCAL positions in pool_idx)

    Returns:
        Global indices for updated train_idx, pool_idx,
        and the candidates that were transferred.
    """
    if len(pool_idx) == 0:
        return train_idx, pool_idx, []

    cand = np.asarray(candidates_idx, dtype=int).reshape(-1)
    if cand.size == 0:
        return train_idx, pool_idx, []

    # Convert pool-local indices -> global indices
    candidates_global_idx = [pool_idx[i] for i in cand.tolist()]

    # Add to training set (global)
    train_new_idx = list(train_idx) + candidates_global_idx

    # Remove from pool (global)
    remove_set = set(candidates_global_idx)
    pool_new_idx = [g for g in pool_idx if g not in remove_set]

    return train_new_idx, pool_new_idx, candidates_global_idx
