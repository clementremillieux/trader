#!/usr/bin/env python3
"""Training entrypoint for the Temporal Fusion Transformer model."""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm  # type: ignore[import]

try:
    AUTOCast = torch.amp.autocast  # type: ignore[attr-defined]
    GradScalerCls = torch.amp.GradScaler  # type: ignore[attr-defined]
    # Check if device_type is supported
    import inspect

    sig = inspect.signature(GradScalerCls.__init__)
    AMP_HAS_DEVICE_TYPE = "device_type" in sig.parameters
except AttributeError:  # pragma: no cover - fallback for older PyTorch
    from torch.cuda.amp import autocast as AUTOCast  # type: ignore
    from torch.cuda.amp import GradScaler as GradScalerCls  # type: ignore

    AMP_HAS_DEVICE_TYPE = False

    AMP_HAS_DEVICE_TYPE = False


def amp_autocast(device_type: str, enabled: bool):
    # Toujours passer device_type, car certaines versions de torch.amp.autocast l'exigent
    return AUTOCast(device_type=device_type, enabled=enabled)  # type: ignore[call-arg]


def create_grad_scaler(enabled: bool):
    if enabled:
        if AMP_HAS_DEVICE_TYPE:
            return GradScalerCls(device_type="cuda", enabled=enabled)  # type: ignore[call-arg]
        else:
            return GradScalerCls(enabled=enabled)
    return None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model.temporal_fusion_transformer import (  # noqa: E402
    TFTConfig,
    TemporalFusionTransformer,
)


LOGGER = logging.getLogger("train_tft")

MetricsDict = Dict[str, Any]


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        gamma_per_class: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        label_smoothing: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma doit être >= 0")
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        if gamma_per_class is not None and gamma_per_class.dim() != 1:
            raise ValueError("gamma_per_class doit être un tenseur 1D")
        self.register_buffer(
            "gamma_per_class",
            gamma_per_class if gamma_per_class is not None else None,
        )
        if alpha is not None and alpha.dim() != 1:
            raise ValueError("alpha doit être un tenseur 1D")
        self.register_buffer("alpha", alpha if alpha is not None else None)
        if label_smoothing is not None and label_smoothing.dim() != 1:
            raise ValueError("label_smoothing doit être un tenseur 1D")
        self.register_buffer(
            "label_smoothing_tensor",
            label_smoothing if label_smoothing is not None else None,
        )
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction doit être 'none', 'mean' ou 'sum'")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets = targets.long()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.gamma_per_class is not None:
            gammas = self.gamma_per_class.to(logits.device).gather(0, targets)
        else:
            gammas = torch.full_like(target_probs, self.gamma, dtype=logits.dtype)
        focal_factor = (1 - target_probs).pow(gammas)

        if self.weight is not None:
            weights = self.weight.to(logits.device)
            focal_factor = focal_factor * weights.gather(0, targets)
        if self.alpha is not None:
            alphas = self.alpha.to(logits.device)
            focal_factor = focal_factor * alphas.gather(0, targets)

        if self.label_smoothing_tensor is not None:
            smoothing = self.label_smoothing_tensor.to(logits.device)
            n_classes = logits.size(1)
            eps = smoothing.gather(0, targets).unsqueeze(1)
            true_dist = torch.full_like(log_probs, 0.0)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0)
            true_dist = true_dist * (1 - eps) + (eps / (n_classes - 1))
            loss = -(true_dist * focal_factor.unsqueeze(1) * log_probs).sum(dim=1)
        else:
            loss = -focal_factor * target_log_probs

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    class_weights: np.ndarray
    tau_vocab_size: int
    feature_dim: int
    seq_len: int
    samples_per_file: Dict[str, int]
    reg_mean: float
    reg_std: float
    vol_mean: float
    vol_std: float
    vol_log: bool


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def list_dataset_files(
    path: Path | str, prefix: str, max_files: int | None
) -> List[Path]:
    base_path = Path(path)
    files = sorted(base_path.glob(f"{prefix}*.pickle"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le préfixe {prefix!r}")
    return files


def sanitize_array(
    array: np.ndarray,
    *,
    nan_value: float = 0.0,
    posinf_value: float | None = None,
    neginf_value: float | None = None,
) -> np.ndarray:
    """Replace NaN/Inf values with finite fallbacks."""

    if posinf_value is None:
        posinf_value = nan_value
    if neginf_value is None:
        neginf_value = nan_value
    return np.nan_to_num(
        array,
        nan=nan_value,
        posinf=posinf_value,
        neginf=neginf_value,
    )


def compute_statistics(
    file_paths: Sequence[Path],
    sample_fraction: float,
    seed: int,
    limit_samples_per_file: int | None,
) -> NormalizationStats:
    rng = np.random.default_rng(seed)
    sum_vec: np.ndarray | None = None
    sum_sq: np.ndarray | None = None
    total_count = 0
    class_counts = np.zeros(3, dtype=np.float64)
    sum_reg = 0.0
    sum_reg_sq = 0.0
    sum_vol = 0.0
    sum_vol_sq = 0.0
    total_sequences = 0
    tau_max = 0
    feature_dim = None
    seq_len = None
    samples_per_file: Dict[str, int] = {}

    vol_log = True

    for path in file_paths:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        X = payload["X"]  # (N, T, F)
        y_cls = payload["y_cls"]
        tau = payload.get("tau")
        if tau is None:
            raise KeyError("Le fichier ne contient pas la clé 'tau'.")
        y_reg = payload["y_reg"]
        y_vol = payload["y_vol"]
        n_seq, t, f = X.shape
        seq_len = seq_len or t
        feature_dim = feature_dim or f

        take = min(limit_samples_per_file, n_seq) if limit_samples_per_file else n_seq
        samples_per_file[str(path)] = take
        total_sequences += take

        indices = np.arange(take)
        if sample_fraction < 1.0:
            sample_size = max(1, int(len(indices) * sample_fraction))
            indices = rng.choice(indices, size=sample_size, replace=False)

        X_sel = sanitize_array(X[indices].astype(np.float64))
        y_sel = y_cls[indices]
        reg_sel = sanitize_array(y_reg[indices].astype(np.float64))
        vol_sel = np.clip(y_vol[indices].astype(np.float64), a_min=0.0, a_max=None)
        if vol_log:
            vol_sel = np.log1p(vol_sel)
        vol_sel = sanitize_array(vol_sel)
        tau_sel = sanitize_array(tau[indices].astype(np.float64))
        tau_max = max(tau_max, int(np.max(tau_sel)))

        flat = X_sel.reshape(-1, f)
        if sum_vec is None:
            sum_vec = flat.sum(axis=0)
            sum_sq = np.square(flat).sum(axis=0)
        else:
            sum_vec += flat.sum(axis=0)
            sum_sq += np.square(flat).sum(axis=0)
        total_count += flat.shape[0]

        class_counts += np.bincount((y_cls[:take] + 1).astype(np.int64), minlength=3)
        sum_reg += reg_sel.sum()
        sum_reg_sq += np.square(reg_sel).sum()
        sum_vol += vol_sel.sum()
        sum_vol_sq += np.square(vol_sel).sum()

        del payload, X, y_cls, tau, X_sel, y_sel, tau_sel, reg_sel, vol_sel, flat

    if sum_vec is None or sum_sq is None or feature_dim is None or seq_len is None:
        raise RuntimeError("Calcul des statistiques impossible: données vides")

    mean = sum_vec / total_count
    var = sum_sq / total_count - mean**2
    std = np.sqrt(np.maximum(var, 1e-6))

    class_weights = total_sequences / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    reg_mean = sum_reg / total_sequences
    reg_var = sum_reg_sq / total_sequences - reg_mean**2
    reg_std = float(max(math.sqrt(max(reg_var, 1e-6)), 1e-3))

    vol_mean = sum_vol / total_sequences
    vol_var = sum_vol_sq / total_sequences - vol_mean**2
    vol_std = float(max(math.sqrt(max(vol_var, 1e-6)), 1e-3))

    return NormalizationStats(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        class_weights=class_weights.astype(np.float32),
        tau_vocab_size=max(tau_max + 2, 16),
        feature_dim=feature_dim,
        seq_len=seq_len,
        samples_per_file=samples_per_file,
        reg_mean=float(reg_mean),
        reg_std=reg_std,
        vol_mean=float(vol_mean),
        vol_std=vol_std,
        vol_log=vol_log,
    )


class MultiFileSequenceDataset(IterableDataset):
    """Stream large pickled sequences without loading everything in memory."""

    def __init__(
        self,
        file_paths: Sequence[Path],
        stats: NormalizationStats,
        shuffle_files: bool,
        shuffle_samples: bool,
        seed: int,
        limit_per_file: int | None,
        balance_classes: bool,
    ) -> None:
        super().__init__()
        self.file_paths = [Path(p) for p in file_paths]
        self.shuffle_files = shuffle_files
        self.shuffle_samples = shuffle_samples
        self.limit_per_file = limit_per_file
        self.stats = stats
        self.seed = seed
        self.balance_classes = balance_classes
        self.mean = stats.mean.reshape(1, 1, -1)
        self.std = stats.std.reshape(1, 1, -1)
        self.reg_mean = stats.reg_mean
        self.reg_std = stats.reg_std
        self.vol_mean = stats.vol_mean
        self.vol_std = stats.vol_std
        self.vol_log = stats.vol_log

    def __getitem__(
        self, index: int
    ) -> Any:  # pragma: no cover - IterableDataset contract
        raise NotImplementedError(
            "MultiFileSequenceDataset ne supporte pas l'indexation"
        )

    def _select_files_for_worker(self) -> List[Path]:
        worker_info = get_worker_info()
        if worker_info is None:
            return list(self.file_paths)
        return list(self.file_paths[worker_info.id :: worker_info.num_workers])

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        files = self._select_files_for_worker()
        worker_info = get_worker_info()
        worker_seed = self.seed if worker_info is None else self.seed + worker_info.id
        rng = random.Random(worker_seed)

        if self.shuffle_files:
            rng.shuffle(files)

        for path in files:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)

            X = payload["X"].astype(np.float32)
            y_cls = payload["y_cls"].astype(np.int64)
            y_reg = payload["y_reg"].astype(np.float32)
            y_vol = payload["y_vol"].astype(np.float32)
            tau = payload["tau"].astype(np.float32)

            if self.limit_per_file:
                limit = min(self.limit_per_file, X.shape[0])
            else:
                limit = X.shape[0]
            X = X[:limit]
            y_cls = y_cls[:limit]
            y_reg = y_reg[:limit]
            y_vol = y_vol[:limit]
            tau = tau[:limit]

            indices = list(range(limit))
            if self.shuffle_samples:
                rng.shuffle(indices)

            if self.balance_classes:
                class_buckets: Dict[int, List[int]] = {0: [], 1: [], 2: []}
                for idx in indices:
                    cls_idx = int(y_cls[idx] + 1)
                    class_buckets.setdefault(cls_idx, []).append(idx)
                non_empty = [bucket for bucket in class_buckets.values() if bucket]
                if non_empty:
                    max_len = max(len(bucket) for bucket in non_empty)
                    balanced_indices: List[int] = []
                    for i in range(max_len):
                        for bucket in non_empty:
                            balanced_indices.append(bucket[i % len(bucket)])
                    if self.shuffle_samples:
                        rng.shuffle(balanced_indices)
                    indices = balanced_indices

            np.subtract(X, self.mean, out=X)
            np.divide(X, self.std, out=X)
            X = sanitize_array(X)

            y_reg = (y_reg - self.reg_mean) / self.reg_std
            y_reg = sanitize_array(y_reg)
            y_vol = np.clip(y_vol, a_min=0.0, a_max=None)
            if self.vol_log:
                y_vol = np.log1p(y_vol)
            y_vol = (y_vol - self.vol_mean) / self.vol_std
            y_vol = sanitize_array(y_vol)
            tau = sanitize_array(tau, nan_value=0.0, posinf_value=0.0, neginf_value=0.0)
            tau = np.clip(tau, a_min=0.0, a_max=None)

            for idx in indices:
                yield {
                    "inputs": torch.from_numpy(X[idx]),
                    "y_cls": torch.tensor(int(y_cls[idx] + 1), dtype=torch.long),
                    "y_reg": torch.tensor(float(y_reg[idx]), dtype=torch.float32),
                    "y_vol": torch.tensor(float(y_vol[idx]), dtype=torch.float32),
                    "tau": torch.tensor(int(round(float(tau[idx]))), dtype=torch.long),
                }

            del payload, X, y_cls, y_reg, y_vol, tau


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    inputs = torch.stack([item["inputs"] for item in batch], dim=0)
    y_cls = torch.stack([item["y_cls"] for item in batch], dim=0)
    y_reg = torch.stack([item["y_reg"] for item in batch], dim=0)
    y_vol = torch.stack([item["y_vol"] for item in batch], dim=0)
    tau = torch.stack([item["tau"] for item in batch], dim=0)
    return {
        "inputs": inputs,
        "y_cls": y_cls,
        "y_reg": y_reg,
        "y_vol": y_vol,
        "tau": tau,
    }


def prepare_dataloader(
    files: Sequence[Path],
    stats: NormalizationStats,
    batch_size: int,
    shuffle_files: bool,
    shuffle_samples: bool,
    seed: int,
    limit_per_file: int | None,
    num_workers: int,
    balance_classes: bool,
) -> DataLoader:
    dataset = MultiFileSequenceDataset(
        file_paths=files,
        stats=stats,
        shuffle_files=shuffle_files,
        shuffle_samples=shuffle_samples,
        seed=seed,
        limit_per_file=limit_per_file,
        balance_classes=balance_classes,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def estimate_num_samples(files: Sequence[Path], stats: NormalizationStats) -> int:
    total = 0
    for path in files:
        total += stats.samples_per_file.get(str(path), 0)
    return total


def train(
    model: TemporalFusionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip: float,
    lambda_reg: float,
    lambda_vol: float,
    mixed_precision: bool,
    output_dir: Path,
    save_every: int,
    focal_gamma: float = 2.0,
    focal_gamma_per_class: torch.Tensor | None = None,
    focal_alpha: torch.Tensor | None = None,
    label_smoothing: torch.Tensor | None = None,
    curriculum_epochs: int = 0,
    early_stopping_patience: int = 0,
    log_interval: int = 0,
    val_log_interval: int = 0,
) -> MetricsDict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    autocast_device = "cuda" if device.type == "cuda" else device.type
    use_cuda_amp = mixed_precision and device.type == "cuda"
    scaler = create_grad_scaler(use_cuda_amp)
    class_weights = class_weights.to(torch.float32)
    focal_gamma_tensor = (
        focal_gamma_per_class.to(torch.float32)
        if focal_gamma_per_class is not None
        else None
    )
    focal_alpha_tensor = (
        focal_alpha.to(torch.float32) if focal_alpha is not None else None
    )
    label_smoothing_tensor = (
        label_smoothing.to(torch.float32) if label_smoothing is not None else None
    )
    cls_loss_fn = FocalLoss(
        weight=class_weights,
        gamma=focal_gamma,
        gamma_per_class=focal_gamma_tensor,
        alpha=focal_alpha_tensor,
        label_smoothing=label_smoothing_tensor,
    )
    reg_loss_fn = torch.nn.SmoothL1Loss()
    vol_loss_fn = torch.nn.SmoothL1Loss()

    binary_loss_fn: torch.nn.Module | None = None
    if curriculum_epochs > 0:
        directional_weight = class_weights[[0, 2]].mean()
        binary_weights = torch.stack([class_weights[1], directional_weight])
        binary_weights = binary_weights / binary_weights.sum() * 2.0
        binary_loss_fn = torch.nn.NLLLoss(weight=binary_weights.to(device))

    best_val_loss = float("inf")
    best_metrics: MetricsDict = {}
    best_balanced_acc = -float("inf")
    patience_counter = 0

    def _safe_len(obj: Any) -> int | None:
        try:
            return len(obj)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            return None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_reg = 0.0
        running_vol = 0.0
        running_acc = 0.0
        total_samples = 0

        total_train_batches = _safe_len(train_loader)
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch} [train]",
            leave=False,
            total=total_train_batches,
        )
        for step, batch in enumerate(progress, start=1):
            inputs = batch["inputs"].to(device)
            tau = batch["tau"].to(device)
            y_cls = batch["y_cls"].to(device)
            y_reg = batch["y_reg"].to(device)
            y_vol = batch["y_vol"].to(device)
            batch_size = inputs.size(0)

            optimizer.zero_grad(set_to_none=True)

            with amp_autocast(autocast_device, use_cuda_amp):
                outputs = model(inputs, tau)
                logits = outputs["logits"]
                reg_pred = outputs["ret"]
                vol_pred = outputs["vol"]

                if curriculum_epochs > 0 and epoch <= curriculum_epochs:
                    if binary_loss_fn is None:
                        raise RuntimeError("binary_loss_fn non initialisé")
                    log_probs = F.log_softmax(logits, dim=1)
                    binary_target = (y_cls != 1).long()
                    dir_log_prob = torch.logaddexp(log_probs[:, 0], log_probs[:, 2])
                    binary_log_probs = torch.stack(
                        [log_probs[:, 1], dir_log_prob], dim=1
                    )
                    cls_loss = binary_loss_fn(binary_log_probs, binary_target)
                else:
                    cls_loss = cls_loss_fn(logits, y_cls)
                reg_loss = reg_loss_fn(reg_pred, y_reg)
                vol_loss = vol_loss_fn(vol_pred, y_vol)
                loss = cls_loss + lambda_reg * reg_loss + lambda_vol * vol_loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "NaN ou Inf détecté dans la loss. Réduisez le learning_rate, "
                    "désactivez la précision mixte ou inspectez vos données pour des "
                    "valeurs extrêmes / NaN."
                )

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                running_loss += loss.item() * batch_size
                running_cls += cls_loss.item() * batch_size
                running_reg += reg_loss.item() * batch_size
                running_vol += vol_loss.item() * batch_size
                preds = logits.argmax(dim=1)
                running_acc += (preds == y_cls).sum().item()
                total_samples += batch_size
                # Accumule pour matrice confusion train
                if "all_train_labels" not in locals():
                    all_train_labels = []
                    all_train_preds = []
                all_train_labels.extend(y_cls.cpu().numpy().tolist())
                all_train_preds.extend(preds.cpu().numpy().tolist())

            progress.set_postfix(
                loss=running_loss / total_samples,
                acc=running_acc / total_samples,
            )

            if log_interval > 0 and step % log_interval == 0:
                LOGGER.info(
                    "Epoch %d [train] step %d%s | loss=%.4f cls=%.4f reg=%.4f vol=%.4f acc=%.4f",
                    epoch,
                    step,
                    f"/{total_train_batches}" if total_train_batches else "",
                    running_loss / total_samples,
                    running_cls / total_samples,
                    running_reg / total_samples,
                    running_vol / total_samples,
                    running_acc / total_samples,
                )
                # Affichage matrice confusion train (cumulée)
                if step > 0 and len(all_train_labels) > 0:
                    import numpy as np

                    train_confusion = np.zeros((3, 3), dtype=int)
                    for t, p in zip(all_train_labels, all_train_preds):
                        train_confusion[t, p] += 1
                    buy_tp = int(train_confusion[2, 2])
                    buy_fp_from_sell = int(train_confusion[0, 2])
                    sell_tp = int(train_confusion[0, 0])
                    sell_fp_from_buy = int(train_confusion[2, 0])

                    def ratio(tp, fp):
                        denom = tp + fp
                        return float(tp / denom) if denom > 0 else None

                    buy_ratio = ratio(buy_tp, buy_fp_from_sell)
                    sell_ratio = ratio(sell_tp, sell_fp_from_buy)
                    print("\nMatrice de confusion (train, partiel):")
                    print(train_confusion)
                    print(
                        f"Achats vrais: {buy_tp} | Achats faux (depuis vente): {buy_fp_from_sell} | Ratio: {buy_ratio if buy_ratio is not None else 'nan'}"
                    )
                    print(
                        f"Ventes vraies: {sell_tp} | Ventes fausses (depuis achat): {sell_fp_from_buy} | Ratio: {sell_ratio if sell_ratio is not None else 'nan'}\n"
                    )

        # Calcul matrice confusion train sur toute l'époque
        if len(all_train_labels) > 0:
            import numpy as np

            train_confusion = np.zeros((3, 3), dtype=int)
            for t, p in zip(all_train_labels, all_train_preds):
                train_confusion[t, p] += 1
            buy_tp = int(train_confusion[2, 2])
            buy_fp_from_sell = int(train_confusion[0, 2])
            sell_tp = int(train_confusion[0, 0])
            sell_fp_from_buy = int(train_confusion[2, 0])

            def ratio(tp, fp):
                denom = tp + fp
                return float(tp / denom) if denom > 0 else None

            buy_ratio = ratio(buy_tp, buy_fp_from_sell)
            sell_ratio = ratio(sell_tp, sell_fp_from_buy)
            print("\nMatrice de confusion (train, fin d'époque):")
            print(train_confusion)
            print(
                f"Achats vrais: {buy_tp} | Achats faux (depuis vente): {buy_fp_from_sell} | Ratio: {buy_ratio if buy_ratio is not None else 'nan'}"
            )
            print(
                f"Ventes vraies: {sell_tp} | Ventes fausses (depuis achat): {sell_fp_from_buy} | Ratio: {sell_ratio if sell_ratio is not None else 'nan'}\n"
            )

        train_metrics: MetricsDict = {
            "train_loss": running_loss / total_samples,
            "train_cls_loss": running_cls / total_samples,
            "train_reg_loss": running_reg / total_samples,
            "train_vol_loss": running_vol / total_samples,
            "train_acc": running_acc / total_samples,
        }
        LOGGER.info(
            "Epoch %d | %s", epoch, json.dumps(train_metrics, ensure_ascii=False)
        )
        # Affichage humain lisible de la matrice de confusion train à la fin d'époque
        # (à implémenter dans la boucle d'entraînement ci-dessus)

        # Validation
        model.eval()
        val_loss = 0.0
        val_cls = 0.0
        val_reg = 0.0
        val_vol = 0.0
        val_acc = 0.0
        val_samples = 0
        val_confusion = torch.zeros((3, 3), dtype=torch.long)

        total_val_batches = _safe_len(val_loader)
        with torch.no_grad():
            for val_step, batch in enumerate(
                tqdm(
                    val_loader,
                    desc=f"Epoch {epoch} [val]",
                    leave=False,
                    total=total_val_batches,
                ),
                start=1,
            ):
                inputs = batch["inputs"].to(device)
                tau = batch["tau"].to(device)
                y_cls = batch["y_cls"].to(device)
                y_reg = batch["y_reg"].to(device)
                y_vol = batch["y_vol"].to(device)
                batch_size = inputs.size(0)

                outputs = model(inputs, tau)
                logits = outputs["logits"]
                reg_pred = outputs["ret"]
                vol_pred = outputs["vol"]

                if curriculum_epochs > 0 and epoch <= curriculum_epochs:
                    if binary_loss_fn is None:
                        raise RuntimeError("binary_loss_fn non initialisé")
                    log_probs = F.log_softmax(logits, dim=1)
                    binary_target = (y_cls != 1).long()
                    dir_log_prob = torch.logaddexp(log_probs[:, 0], log_probs[:, 2])
                    binary_log_probs = torch.stack(
                        [log_probs[:, 1], dir_log_prob], dim=1
                    )
                    cls_loss = binary_loss_fn(binary_log_probs, binary_target)
                else:
                    cls_loss = cls_loss_fn(logits, y_cls)
                reg_loss = reg_loss_fn(reg_pred, y_reg)
                vol_loss = vol_loss_fn(vol_pred, y_vol)
                loss = cls_loss + lambda_reg * reg_loss + lambda_vol * vol_loss

                val_loss += loss.item() * batch_size
                val_cls += cls_loss.item() * batch_size
                val_reg += reg_loss.item() * batch_size
                val_vol += vol_loss.item() * batch_size
                preds = logits.argmax(dim=1)
                val_acc += (preds == y_cls).sum().item()
                val_samples += batch_size

                pairs = (y_cls.cpu() * 3 + preds.cpu()).long()
                counts = torch.bincount(pairs, minlength=9)
                val_confusion += counts.view(3, 3)

                if val_log_interval > 0 and val_step % val_log_interval == 0:
                    LOGGER.info(
                        "Epoch %d [val] step %d%s | loss=%.4f cls=%.4f reg=%.4f vol=%.4f acc=%.4f",
                        epoch,
                        val_step,
                        f"/{total_val_batches}" if total_val_batches else "",
                        val_loss / val_samples,
                        val_cls / val_samples,
                        val_reg / val_samples,
                        val_vol / val_samples,
                        val_acc / val_samples,
                    )
                    # Affichage humain lisible de la matrice de confusion et ratios
                    confusion = val_confusion.numpy()
                    buy_tp = int(confusion[2, 2])
                    buy_fp_from_sell = int(confusion[0, 2])
                    sell_tp = int(confusion[0, 0])
                    sell_fp_from_buy = int(confusion[2, 0])
                    print("\nMatrice de confusion (val):")
                    print(confusion)
                    print(
                        f"Achats vrais: {buy_tp} | Achats faux (depuis vente): {buy_fp_from_sell}"
                    )
                    print(
                        f"Ventes vraies: {sell_tp} | Ventes fausses (depuis achat): {sell_fp_from_buy}\n"
                    )

        epoch_metrics: MetricsDict

        if val_samples > 0:
            confusion = val_confusion.numpy()
            per_class = confusion.sum(axis=1)
            recalls = np.divide(
                np.diag(confusion),
                per_class,
                out=np.zeros_like(per_class, dtype=np.float64),
                where=per_class > 0,
            )
            balanced_acc = float(np.mean(recalls))
            epoch_metrics = {
                **train_metrics,
                "val_loss": val_loss / val_samples,
                "val_cls_loss": val_cls / val_samples,
                "val_reg_loss": val_reg / val_samples,
                "val_vol_loss": val_vol / val_samples,
                "val_acc": val_acc / val_samples,
                "val_balanced_acc": balanced_acc,
                "val_confusion": confusion.tolist(),
            }

            buy_tp = int(confusion[2, 2])
            buy_fp_from_sell = int(confusion[0, 2])
            sell_tp = int(confusion[0, 0])
            sell_fp_from_buy = int(confusion[2, 0])

            def ratio(tp: int, fp: int) -> float | None:
                denom = tp + fp
                return float(tp / denom) if denom > 0 else None

            buy_ratio = ratio(buy_tp, buy_fp_from_sell)
            sell_ratio = ratio(sell_tp, sell_fp_from_buy)

            epoch_metrics.update(
                {
                    "val_buy_true": buy_tp,
                    "val_buy_false_from_sell": buy_fp_from_sell,
                    "val_buy_true_ratio": buy_ratio,
                    "val_sell_true": sell_tp,
                    "val_sell_false_from_buy": sell_fp_from_buy,
                    "val_sell_true_ratio": sell_ratio,
                }
            )
            LOGGER.info(
                "Epoch %d | %s", epoch, json.dumps(epoch_metrics, ensure_ascii=False)
            )

            # Affichage humain lisible de la matrice de confusion et ratios à la fin d'époque
            print("\nMatrice de confusion (val, fin d'époque):")
            print(confusion)
            print(
                f"Achats vrais: {buy_tp} | Achats faux (depuis vente): {buy_fp_from_sell} | Ratio: {buy_ratio if buy_ratio is not None else 'nan'}"
            )
            print(
                f"Ventes vraies: {sell_tp} | Ventes fausses (depuis achat): {sell_fp_from_buy} | Ratio: {sell_ratio if sell_ratio is not None else 'nan'}\n"
            )
        else:
            LOGGER.warning(
                "Validation vide: aucun échantillon n'a été évalué. "
                "Les métriques val seront nulles."
            )
            epoch_metrics = {
                **train_metrics,
                "val_loss": None,
                "val_cls_loss": None,
                "val_reg_loss": None,
                "val_vol_loss": None,
                "val_acc": None,
                "val_balanced_acc": None,
                "val_confusion": val_confusion.tolist(),
                "val_buy_true": 0,
                "val_buy_false_from_sell": 0,
                "val_buy_true_ratio": None,
                "val_sell_true": 0,
                "val_sell_false_from_buy": 0,
                "val_sell_true_ratio": None,
            }
            LOGGER.info(
                "Epoch %d | %s", epoch, json.dumps(epoch_metrics, ensure_ascii=False)
            )

        val_balanced_acc_metric = epoch_metrics.get("val_balanced_acc")
        improved_balanced = False
        if isinstance(val_balanced_acc_metric, (int, float)):
            if val_balanced_acc_metric > best_balanced_acc + 1e-4:
                best_balanced_acc = float(val_balanced_acc_metric)
                improved_balanced = True
                patience_counter = 0
        elif val_balanced_acc_metric is None:
            improved_balanced = False
        if (
            not improved_balanced
            and early_stopping_patience > 0
            and isinstance(val_balanced_acc_metric, (int, float))
        ):
            patience_counter += 1

        val_loss_metric = epoch_metrics.get("val_loss")
        if (
            isinstance(val_loss_metric, (int, float))
            and float(val_loss_metric) < best_val_loss
        ):
            best_val_loss = float(val_loss_metric)

        if improved_balanced or not best_metrics:
            best_metrics = epoch_metrics
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metrics": epoch_metrics,
                },
                output_dir / "best_val_checkpoint.pt",
            )

        if save_every > 0 and epoch % save_every == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metrics": epoch_metrics,
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            LOGGER.info(
                "Arrêt précoce déclenché après %d epochs sans amélioration "
                "de la balanced accuracy",
                early_stopping_patience,
            )
            break

    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Temporal Fusion Transformer"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--lambda-vol", type=float, default=0.02)
    parser.add_argument("--train-prefix", type=str, default="full_dataset_focus_train_")
    parser.add_argument("--val-prefix", type=str, default="full_dataset_focus_val_")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-val-files", type=int, default=None)
    parser.add_argument("--limit-samples-per-file", type=int, default=None)
    parser.add_argument("--stat-sample-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models/tft"))
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--focal-gamma", type=float, default=2.5)
    parser.add_argument(
        "--focal-gamma-per-class",
        type=float,
        nargs=3,
        metavar=("GAMMA_SELL", "GAMMA_NEUTRAL", "GAMMA_BUY"),
        default=[3.0, 2.0, 3.0],
        help="Gamma focal par classe (ordre: sell, neutral, buy)",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        nargs=3,
        metavar=("ALPHA_SELL", "ALPHA_NEUTRAL", "ALPHA_BUY"),
        default=[1.4, 1.0, 1.6],
        help="Pondérations alpha par classe pour la focal loss",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        nargs=3,
        metavar=("SMOOTH_SELL", "SMOOTH_NEUTRAL", "SMOOTH_BUY"),
        default=[0.05, 0.02, 0.05],
        help="Label smoothing asymétrique (valeurs entre 0 et 0.3 typiquement)",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Rééquilibre les classes en sur-échantillonnant les minoritaires",
    )
    parser.add_argument(
        "--curriculum-epochs",
        type=int,
        default=0,
        help="Nombre d'epochs en curriculum (neutre vs directionnel)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Patience (en epochs) pour l'arrêt précoce basé sur la balanced accuracy",
    )
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-transformer-blocks", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--static-dim", type=int, default=192)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="Nombre de batchs entre deux logs détaillés en entraînement (0 = désactivé)",
    )
    parser.add_argument(
        "--val-log-interval",
        type=int,
        default=100,
        help="Nombre de batchs entre deux logs détaillés en validation (0 = désactivé)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    dataset_root = Path(args.dataset_root).expanduser()
    train_files = list_dataset_files(
        dataset_root, args.train_prefix, args.max_train_files
    )
    val_files = list_dataset_files(dataset_root, args.val_prefix, args.max_val_files)

    LOGGER.info(
        "Calcul des statistiques de normalisation (%d fichiers)", len(train_files)
    )
    stats = compute_statistics(
        file_paths=train_files,
        sample_fraction=args.stat_sample_fraction,
        seed=args.seed,
        limit_samples_per_file=args.limit_samples_per_file,
    )

    LOGGER.info(
        "Features=%d | SeqLen=%d | Tau vocab=%d",
        stats.feature_dim,
        stats.seq_len,
        stats.tau_vocab_size,
    )

    train_loader = prepare_dataloader(
        files=train_files,
        stats=stats,
        batch_size=args.batch_size,
        shuffle_files=True,
        shuffle_samples=True,
        seed=args.seed,
        limit_per_file=args.limit_samples_per_file,
        num_workers=args.num_workers,
        balance_classes=args.balance_classes,
    )
    val_loader = prepare_dataloader(
        files=val_files,
        stats=stats,
        batch_size=args.batch_size,
        shuffle_files=False,
        shuffle_samples=False,
        seed=args.seed,
        limit_per_file=args.limit_samples_per_file,
        num_workers=args.num_workers,
        balance_classes=args.balance_classes,
    )

    est_train_samples = estimate_num_samples(train_files, stats)
    est_val_samples = estimate_num_samples(val_files, stats)
    LOGGER.info(
        "Jeu d'entraînement ≈ %d séquences | Validation ≈ %d séquences",
        est_train_samples,
        est_val_samples,
    )

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    LOGGER.info("Utilisation du device: %s", device)

    cfg = TFTConfig(
        input_dim=stats.feature_dim,
        seq_len=stats.seq_len,
        tau_vocab_size=stats.tau_vocab_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_transformer_blocks,
        dropout=args.dropout,
        conv_kernel_sizes=(3, 5, 7),
        conv_dilations=(1, 2, 4),
        static_dim=args.static_dim,
    )

    model = TemporalFusionTransformer(cfg).to(device)
    class_weights = torch.from_numpy(stats.class_weights)

    focal_gamma_per_class_tensor = (
        torch.tensor(args.focal_gamma_per_class, dtype=torch.float32)
        if args.focal_gamma_per_class is not None
        else None
    )
    focal_alpha_tensor = (
        torch.tensor(args.focal_alpha, dtype=torch.float32)
        if args.focal_alpha is not None
        else None
    )
    label_smoothing_tensor = (
        torch.tensor(args.label_smoothing, dtype=torch.float32)
        if args.label_smoothing is not None
        else None
    )

    if label_smoothing_tensor is not None and torch.any(
        (label_smoothing_tensor < 0) | (label_smoothing_tensor >= 1)
    ):
        raise ValueError("Les valeurs de label smoothing doivent être dans [0, 1).")

    class_count = class_weights.numel()
    for name, tensor in {
        "focal_gamma_per_class": focal_gamma_per_class_tensor,
        "focal_alpha": focal_alpha_tensor,
        "label_smoothing": label_smoothing_tensor,
    }.items():
        if tensor is not None and tensor.numel() != class_count:
            raise ValueError(
                f"{name} doit contenir {class_count} valeurs (une par classe)."
            )

    best_metrics = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate,
        grad_clip=args.grad_clip,
        lambda_reg=args.lambda_reg,
        lambda_vol=args.lambda_vol,
        mixed_precision=args.mixed_precision,
        output_dir=args.output_dir,
        save_every=args.save_every,
        focal_gamma=args.focal_gamma,
        focal_gamma_per_class=focal_gamma_per_class_tensor,
        focal_alpha=focal_alpha_tensor,
        label_smoothing=label_smoothing_tensor,
        curriculum_epochs=args.curriculum_epochs,
        early_stopping_patience=args.early_stopping_patience,
        log_interval=args.log_interval,
        val_log_interval=args.val_log_interval,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "normalization": {
                "mean": stats.mean.tolist(),
                "std": stats.std.tolist(),
                "class_weights": stats.class_weights.tolist(),
                "tau_vocab_size": stats.tau_vocab_size,
                "reg_mean": stats.reg_mean,
                "reg_std": stats.reg_std,
                "vol_mean": stats.vol_mean,
                "vol_std": stats.vol_std,
                "vol_log": stats.vol_log,
            },
            "best_metrics": best_metrics,
            "args": vars(args),
        },
        args.output_dir / "final_model.pt",
    )
    LOGGER.info("Entraînement terminé. Modèle sauvegardé dans %s", args.output_dir)


if __name__ == "__main__":
    main()
