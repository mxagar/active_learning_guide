import time
from typing import Optional, Literal
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
from torch.utils.data import DataLoader, Subset

from sklearn.manifold import TSNE
from umap import UMAP

from skactiveml.base import SkactivemlClassifier
from skactiveml.pool import RandomSampling, UncertaintySampling, Badge

from data_utils import CustomDataset 
from model_utils import (
    SimpleCNN,
    TrainConfig,
    train,
    evaluate,
    load_model,
    set_seed,
)


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


def plot_embeddings_2d(
    model,
    paths: list,
    labels: list,
    classe_names: list,
    train_idx: list[int],
    pool_idx: list[int],
    selected_idx: list[int],
    eval_transform,
    method: str = "umap",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Visualize embeddings in 2D using UMAP or t-SNE."""
    
    # Combine all indices we want to visualize
    all_idx = list(train_idx) + list(pool_idx) + list(selected_idx)
    
    # Remove duplicates while preserving order
    seen = set()
    all_idx_unique = []
    for idx in all_idx:
        if idx not in seen:
            seen.add(idx)
            all_idx_unique.append(idx)
    all_idx = all_idx_unique
    
    # Create dataset and loader
    all_ds = CustomDataset(paths, labels, all_idx, transform=eval_transform)
    all_loader = DataLoader(
        all_ds, batch_size=batch_size, shuffle=False, 
        num_workers=0, pin_memory=False
    )
    
    # Extract embeddings
    model = model.to(device)
    model.eval()
    
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(all_loader, desc="Extracting embeddings"):
            batch_x = batch_x.to(device)
            _, feats = model(batch_x, return_embeddings=True)
            embeddings_list.append(feats.cpu().numpy())
            labels_list.extend(batch_y.numpy())
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels_array = np.array(labels_list)
    
    # Reduce to 2D
    if method.lower() == "umap":
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, n_jobs=1)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create mapping from global idx to position
    idx_to_pos = {global_idx: i for i, global_idx in enumerate(all_idx)}
    
    # Split into groups
    train_positions = [idx_to_pos[idx] for idx in train_idx if idx in idx_to_pos]
    pool_positions = [idx_to_pos[idx] for idx in pool_idx if idx in idx_to_pos]
    selected_positions = [idx_to_pos[idx] for idx in selected_idx if idx in idx_to_pos]
    
    train_coords = embeddings_2d[train_positions]
    train_labels = labels_array[train_positions]
    
    pool_coords = embeddings_2d[pool_positions]
    pool_labels = labels_array[pool_positions]
    
    selected_coords = embeddings_2d[selected_positions]
    selected_labels = labels_array[selected_positions]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    num_classes = len(classe_names)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    # 1. Pool (light, small)
    for class_idx in range(num_classes):
        mask = pool_labels == class_idx
        if mask.any():
            ax.scatter(
                pool_coords[mask, 0], pool_coords[mask, 1],
                c=[colors[class_idx]], s=20, alpha=0.3,
                marker='o', edgecolors='none', zorder=1
            )
    
    # 2. Training set (bold, medium)
    for class_idx in range(num_classes):
        mask = train_labels == class_idx
        if mask.any():
            ax.scatter(
                train_coords[mask, 0], train_coords[mask, 1],
                c=[colors[class_idx]], s=100, alpha=0.7,
                marker='o', edgecolors='black', linewidths=1, zorder=2
            )
    
    # 3. Selected samples (stars with red edges, largest)
    if len(selected_coords) > 0:
        for class_idx in range(num_classes):
            mask = selected_labels == class_idx
            if mask.any():
                ax.scatter(
                    selected_coords[mask, 0], selected_coords[mask, 1],
                    c=[colors[class_idx]], s=500, alpha=1.0,
                    marker='*', edgecolors='red', linewidths=4, zorder=3
                )
    
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend - Complete version with marker types AND class colors
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=5, label='Pool', alpha=0.3),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markeredgewidth=1,
               markersize=10, label='Training', alpha=0.7),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
               markeredgecolor='red', markeredgewidth=2,
               markersize=18, label=f'Selected ({len(selected_idx)})', alpha=1.0),
    ]
    
    # Add spacer and class colors
    legend_elements.append(Line2D([0], [0], color='none', label='', markersize=0))  # Spacer
    legend_elements.append(Line2D([0], [0], color='none', label='Classes:', markersize=0))
    
    for class_idx, class_name in enumerate(classe_names):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors[class_idx], markersize=8,
                   label=f'  {class_name}', alpha=0.8)
        )
    
    ax.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def evaluate_active_learning(
    # Data components
    paths: list,
    labels: list,
    classe_names: list,
    initial_train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    initial_pool_idx: list[int],
    train_transform,
    eval_transform,
    
    # Active learning parameters
    extension_ratio: float = 0.05,
    method: SearchStrategy = "badge",
    max_iterations: Optional[int] = None,
    
    # Training parameters
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    
    # Model saving
    artifacts_dir: str = "runs/active_learning",
    run_name_slug: str = "exp",
    
    # Output
    output_csv: str = "active_learning_experiments.csv",
    
    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Evaluate active learning by iteratively adding samples to the training set.
    
    Args:
        paths: List of all image paths
        labels: List of all labels (same length as paths)
        classe_names: List of class names
        initial_train_idx: Initial training set indices (global)
        val_idx: Validation set indices (global, remains fixed)
        test_idx: Test set indices (global, remains fixed)
        initial_pool_idx: Initial pool indices (global, decreases each iteration)
        train_transform: Transform for training data
        eval_transform: Transform for evaluation data
        extension_ratio: Fraction of pool to query each iteration (converted to query_size)
        method: Active learning strategy
        max_iterations: Maximum number of iterations (None = until pool is empty)
        epochs: Number of training epochs per iteration
        lr: Learning rate
        batch_size: Batch size for training
        artifacts_dir: Directory to save model checkpoints
        run_name_slug: Base name for run (will be modified with iteration info)
        output_csv: Path to output CSV file
        device: Device to train on
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Dictionary with final statistics and history as pandas DataFrame
    """
    # Initialize output directory and CSV path
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Helper function to create dataloaders
    def _make_loader(ds, batch_size=16, shuffle=False):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    
    # Initialize indices (make copies to avoid modifying original)
    train_idx = list(initial_train_idx)
    pool_idx = list(initial_pool_idx)
    
    # Calculate total samples
    total_samples = len(paths)
    
    # History tracking with DataFrame
    results_list = []
    iteration = 0
    
    if verbose:
        print("Starting active learning evaluation")
        print(f"Method: {method}")
        print(f"Extension ratio: {extension_ratio}")
        print(f"Initial train size: {len(train_idx)}")
        print(f"Val size: {len(val_idx)}")
        print(f"Test size: {len(test_idx)}")
        print(f"Initial pool size: {len(pool_idx)}")
        print(f"Output CSV: {output_csv}")
        print(f"Embeddings plots in directory: {artifacts_dir}")
        print("=" * 80)
    
    while len(pool_idx) > 0:
        if max_iterations is not None and iteration >= max_iterations:
            if verbose:
                print(f"Reached maximum iterations: {max_iterations}")
            break
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Iteration {iteration}")
            print(f"Train size: {len(train_idx)}, Pool size: {len(pool_idx)}")
        
        # Calculate current train ratio (rounded to 2 decimals)
        current_train_ratio = round(len(train_idx) / total_samples, 2)

        # Create datasets
        train_ds = CustomDataset(paths, labels, train_idx, transform=train_transform)
        val_ds = CustomDataset(paths, labels, val_idx, transform=eval_transform)
        test_ds = CustomDataset(paths, labels, test_idx, transform=eval_transform)
        pool_ds = CustomDataset(paths, labels, pool_idx, transform=eval_transform)
        
        # Create dataloaders
        train_loader = _make_loader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = _make_loader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = _make_loader(test_ds, batch_size=batch_size, shuffle=False)
        
        # Initialize fresh model
        model = SimpleCNN(num_classes=len(classe_names))
        
        # Training configuration
        # method, iteration, train ratio in name for better tracking (2 decimal precision)
        run_name = f"{run_name_slug}_{method}_iteration_{iteration:03d}_train_ratio_{current_train_ratio:.2f}"
        cfg = TrainConfig(
            num_classes=len(classe_names),
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            out_dir=artifacts_dir.as_posix(),
            run_name=run_name,
            metric_for_best="val_f1",
            maximize_metric=True,
            device=device,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
            #seed=seed + iteration,  # Different seed per iteration
        )
        
        # Train the model
        if verbose:
            print(f"Training model for iteration {iteration}...")
        start_time = time.time()
        train_history = train(model, train_loader, val_loader, cfg)
        train_time = time.time() - start_time
        
        # Load best checkpoint and evaluate on test set
        best_ckpt_path = Path(cfg.out_dir) / cfg.run_name / "best.pt"
        ckpt = load_model(best_ckpt_path, model, map_location=cfg.device)
        model = model.to(torch.device(cfg.device))
        
        test_metrics = evaluate(model, test_loader, torch.device(cfg.device), num_classes=len(classe_names))
        
        # Get best epoch metrics from history
        best_epoch_idx = -1
        best_metric_value = ckpt.get("best_metric", None)
        
        # Find the best epoch in history
        for i, epoch_metrics in enumerate(train_history):
            if epoch_metrics.get(cfg.metric_for_best) == best_metric_value:
                best_epoch_idx = i
                break
        
        if best_epoch_idx == -1:
            best_epoch_idx = -1  # Use last epoch
        
        best_epoch_metrics = train_history[best_epoch_idx]
        
        # Prepare row data
        row_data = {
            "iteration": iteration,
            "method": method,
            "train_ratio": current_train_ratio,
            "extension_ratio": 0.0 if iteration == 0 else round(extension_ratio, 2),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "pool_size": len(pool_idx),
            "query_size": 0,  # Will be updated after querying (0 for first iteration)
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "train_time_sec": train_time,
            "train_loss": best_epoch_metrics["train_loss"],
            "train_acc": best_epoch_metrics["train_acc"],
            "train_f1": best_epoch_metrics["train_f1"],
            "val_loss": best_epoch_metrics["val_loss"],
            "val_acc": best_epoch_metrics["val_acc"],
            "val_f1": best_epoch_metrics["val_f1"],
            "test_loss": test_metrics["test_loss"],
            "test_acc": test_metrics["test_acc"],
            "test_f1": test_metrics["test_f1"],
            "best_metric": best_metric_value,
            "run_name": run_name,
        }
        
        if verbose:
            print(f"Results - Val F1: {best_epoch_metrics['val_f1']:.4f}, Test F1: {test_metrics['test_f1']:.4f}")
            print(f"Training time: {train_time:.2f}s")
        
        # Active learning: query next candidates
        # Query happens AFTER each iteration (including iteration 0)
        if len(pool_idx) > 0:
            query_size = max(1, int(total_samples * extension_ratio))
            
            if verbose:
                print(f"Querying {query_size} samples using {method}...")
            
            candidates_local_idx = compute_next_candidates(
                model=model,
                pool_ds=pool_ds,
                query_size=query_size,
                method=method,
                seed=seed,
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
                pin_memory=pin_memory,
                verbose=verbose,
            )
            
            # Save old train idx for visualization (before transfer)
            old_train_idx = list(train_idx)
            
            # Transfer candidates
            train_idx, pool_idx, candidates_global_idx = transfer_candidates_idx(
                train_idx=train_idx,
                pool_idx=pool_idx,
                candidates_idx=candidates_local_idx,
            )
            
            row_data["query_size"] = len(candidates_global_idx)
            
            if verbose:
                print(f"Added {len(candidates_global_idx)} samples to training set")
                plot_save_path = artifacts_dir / f"{run_name}_embedding_plot.png"
                plot_embeddings_2d(
                    model=model,
                    paths=paths,
                    labels=labels,
                    classe_names=classe_names,
                    train_idx=old_train_idx,
                    pool_idx=pool_idx,  # New pool (after removal)
                    selected_idx=candidates_global_idx,  # Newly selected
                    eval_transform=eval_transform,
                    method="umap",  # or "tsne"
                    batch_size=batch_size,
                    device=device,
                    save_path=str(plot_save_path),
                    title=f"Iteration {iteration} - {method.upper()} Selection"
                )
        elif iteration == 0:
            # First iteration: no querying yet, just evaluate baseline
            if verbose:
                print("Baseline iteration - no querying performed")
        
        # Add to results list
        results_list.append(row_data)
        
        # Convert to DataFrame and save to CSV
        new_row_df = pd.DataFrame([row_data])
        
        if csv_path.exists():
            # Load existing CSV and append new row
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, new_row_df], ignore_index=True)
        else:
            # First iteration - create new DataFrame
            df = new_row_df
        
        df.to_csv(csv_path, index=False)
        
        if verbose:
            print(f"CSV updated: {csv_path}")
        
        # Cleanup model to free memory and avoid re-using weights
        del model
        
        # Increase iteration counter
        iteration += 1
    
    # Final DataFrame
    #results_df = pd.DataFrame(results_list)
    results_df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"\n{'='*80}")
        print("Active learning evaluation completed!")
        print(f"Total iterations: {iteration}")
        print(f"Final train size: {len(train_idx)}")
        print(f"Final pool size: {len(pool_idx)}")
        print(f"Results saved to: {csv_path}")
        print("\nFinal results summary:")
        print(results_df[['iteration', 'train_size', 'train_ratio', 'val_f1', 'test_f1']].to_string(index=False))
    
    return {
        "history_df": results_df,
        "final_train_idx": train_idx,
        "final_pool_idx": pool_idx,
        "num_iterations": iteration,
        "csv_path": str(csv_path),
    }
