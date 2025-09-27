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
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model.temporal_fusion_transformer import (  # noqa: E402
    TFTConfig,
    TemporalFusionTransformer,
)


LOGGER = logging.getLogger("train_tft")


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


def list_dataset_files(path: str, prefix: str, max_files: int | None) -> List[Path]:
    files = sorted(Path(path).glob(f"{prefix}*.pickle"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le préfixe {prefix!r}")
    return files


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

        X_sel = X[indices]
        y_sel = y_cls[indices]
        reg_sel = y_reg[indices].astype(np.float64)
        vol_sel = np.clip(y_vol[indices].astype(np.float64), a_min=0.0, a_max=None)
        if vol_log:
            vol_sel = np.log1p(vol_sel)
        tau_sel = tau[indices].astype(np.float64)
        tau_max = max(tau_max, int(np.max(tau_sel)))

        flat = X_sel.reshape(-1, f).astype(np.float64)
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
    ) -> None:
        super().__init__()
        self.file_paths = [Path(p) for p in file_paths]
        self.shuffle_files = shuffle_files
        self.shuffle_samples = shuffle_samples
        self.limit_per_file = limit_per_file
        self.stats = stats
        self.seed = seed
        self.mean = stats.mean.reshape(1, 1, -1)
        self.std = stats.std.reshape(1, 1, -1)
        self.reg_mean = stats.reg_mean
        self.reg_std = stats.reg_std
        self.vol_mean = stats.vol_mean
        self.vol_std = stats.vol_std
        self.vol_log = stats.vol_log

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

            np.subtract(X, self.mean, out=X)
            np.divide(X, self.std, out=X)

            y_reg = (y_reg - self.reg_mean) / self.reg_std
            y_vol = np.clip(y_vol, a_min=0.0, a_max=None)
            if self.vol_log:
                y_vol = np.log1p(y_vol)
            y_vol = (y_vol - self.vol_mean) / self.vol_std

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
) -> DataLoader:
    dataset = MultiFileSequenceDataset(
        file_paths=files,
        stats=stats,
        shuffle_files=shuffle_files,
        shuffle_samples=shuffle_samples,
        seed=seed,
        limit_per_file=limit_per_file,
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
) -> Dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    cls_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    reg_loss_fn = torch.nn.SmoothL1Loss()
    vol_loss_fn = torch.nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_reg = 0.0
        running_vol = 0.0
        running_acc = 0.0
        total_samples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for batch in progress:
            inputs = batch["inputs"].to(device)
            tau = batch["tau"].to(device)
            y_cls = batch["y_cls"].to(device)
            y_reg = batch["y_reg"].to(device)
            y_vol = batch["y_vol"].to(device)
            batch_size = inputs.size(0)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=mixed_precision):
                outputs = model(inputs, tau)
                logits = outputs["logits"]
                reg_pred = outputs["ret"]
                vol_pred = outputs["vol"]

                cls_loss = cls_loss_fn(logits, y_cls)
                reg_loss = reg_loss_fn(reg_pred, y_reg)
                vol_loss = vol_loss_fn(vol_pred, y_vol)
                loss = cls_loss + lambda_reg * reg_loss + lambda_vol * vol_loss

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

            progress.set_postfix(
                loss=running_loss / total_samples,
                acc=running_acc / total_samples,
            )

        train_metrics = {
            "train_loss": running_loss / total_samples,
            "train_cls_loss": running_cls / total_samples,
            "train_reg_loss": running_reg / total_samples,
            "train_vol_loss": running_vol / total_samples,
            "train_acc": running_acc / total_samples,
        }
        LOGGER.info(
            "Epoch %d | %s", epoch, json.dumps(train_metrics, ensure_ascii=False)
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_cls = 0.0
        val_reg = 0.0
        val_vol = 0.0
        val_acc = 0.0
        val_samples = 0
        val_confusion = torch.zeros((3, 3), dtype=torch.long)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False):
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

        if val_samples > 0:
            metrics = {
                **train_metrics,
                "val_loss": val_loss / val_samples,
                "val_cls_loss": val_cls / val_samples,
                "val_reg_loss": val_reg / val_samples,
                "val_vol_loss": val_vol / val_samples,
                "val_acc": val_acc / val_samples,
                "val_confusion": val_confusion.tolist(),
            }

            confusion = val_confusion.numpy()
            buy_tp = int(confusion[2, 2])
            buy_fp_from_sell = int(confusion[0, 2])
            sell_tp = int(confusion[0, 0])
            sell_fp_from_buy = int(confusion[2, 0])

            def ratio(tp: int, fp: int) -> float | None:
                denom = tp + fp
                return float(tp / denom) if denom > 0 else None

            buy_ratio = ratio(buy_tp, buy_fp_from_sell)
            sell_ratio = ratio(sell_tp, sell_fp_from_buy)

            metrics.update(
                {
                    "val_buy_true": buy_tp,
                    "val_buy_false_from_sell": buy_fp_from_sell,
                    "val_buy_true_ratio": buy_ratio,
                    "val_sell_true": sell_tp,
                    "val_sell_false_from_buy": sell_fp_from_buy,
                    "val_sell_true_ratio": sell_ratio,
                }
            )
            LOGGER.info("Epoch %d | %s", epoch, json.dumps(metrics, ensure_ascii=False))

            LOGGER.info(
                "Validation confusion (rows=réel [-1,0,1], colonnes=prédit [-1,0,1]): %s",
                confusion.tolist(),
            )
            LOGGER.info(
                "Achats prédits: vrais=%d, faux_depuis_ventes=%d, ratio=%.3f",
                buy_tp,
                buy_fp_from_sell,
                buy_ratio if buy_ratio is not None else float("nan"),
            )
            LOGGER.info(
                "Ventes prédites: vraies=%d, faux_depuis_achats=%d, ratio=%.3f",
                sell_tp,
                sell_fp_from_buy,
                sell_ratio if sell_ratio is not None else float("nan"),
            )
        else:
            LOGGER.warning(
                "Validation vide: aucun échantillon n'a été évalué. "
                "Les métriques val seront nulles."
            )
            metrics = {
                **train_metrics,
                "val_loss": None,
                "val_cls_loss": None,
                "val_reg_loss": None,
                "val_vol_loss": None,
                "val_acc": None,
                "val_confusion": val_confusion.tolist(),
                "val_buy_true": 0,
                "val_buy_false_from_sell": 0,
                "val_buy_true_ratio": None,
                "val_sell_true": 0,
                "val_sell_false_from_buy": 0,
                "val_sell_true_ratio": None,
            }
            LOGGER.info("Epoch %d | %s", epoch, json.dumps(metrics, ensure_ascii=False))

        if metrics["val_loss"] is not None and metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            best_metrics = metrics
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                },
                output_dir / "best_val_checkpoint.pt",
            )

        if save_every > 0 and epoch % save_every == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Temporal Fusion Transformer"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0.3)
    parser.add_argument("--lambda-vol", type=float, default=0.05)
    parser.add_argument("--train-prefix", type=str, default="full_dataset_focus_train_")
    parser.add_argument("--val-prefix", type=str, default="full_dataset_focus_val_")
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
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    train_files = list_dataset_files(args.train_prefix, args.max_train_files)
    val_files = list_dataset_files(args.val_prefix, args.max_val_files)

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
        hidden_dim=512,
        num_heads=8,
        num_transformer_blocks=6,
        dropout=0.2,
        conv_kernel_sizes=(3, 5, 7),
        conv_dilations=(1, 2, 4),
        static_dim=256,
    )

    model = TemporalFusionTransformer(cfg).to(device)
    class_weights = torch.from_numpy(stats.class_weights)

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
