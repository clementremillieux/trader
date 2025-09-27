from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

# Ce script d'évaluation rapide fonctionne uniquement avec les pickles dict
# (pas les objets CryptoDataset scalés). Si vous utilisez les pickles *scaled*,
# chargez-les via le module d'entraînement principal.

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore
from sklearn.feature_selection import (  # type: ignore
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    r2_score,
    mean_absolute_error,
)

# Import optionnel de PyTorch pour la baseline séquentielle
try:  # pragma: no cover - import facultatif
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False


DEFAULT_FEATURE_NAMES = None  # fallback: None → générer f0..fN si meta absent


def _load_dataset(path: str) -> dict:
    """Charge un dataset au format dict, avec support des shards _part*.pickle.

    Si le fichier principal est vide/corrompu, on tente d'agréger tous les
    shards correspondants, triés par index de shard.
    """
    p = Path(path)
    dirp = p.parent
    suffix = p.suffix

    def _concat_shards(shard_paths: list[Path]) -> dict:
        datasets = []
        for sp in shard_paths:
            with sp.open("rb") as fh:
                datasets.append(pickle.load(fh))
        keys = [k for k in datasets[0].keys() if k != "meta"]
        result = {k: datasets[0][k] for k in keys}
        for d in datasets[1:]:
            for k in keys:
                result[k] = np.concatenate([result[k], d[k]], axis=0)
        meta = datasets[-1].get("meta", {}).copy()
        meta["_loaded_from_shards"] = [sp.name for sp in shard_paths]
        result["meta"] = meta
        return result

    name = p.name
    if "_part" in name:
        # Agrège tous les shards de la même base
        base = name.split("_part", 1)[0]
        shard_paths = sorted(dirp.glob(f"{base}_part*{suffix}"))
        if shard_paths:
            return _concat_shards(shard_paths)

    try:
        with p.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        # Fallback: si le fichier principal est vide/corrompu, on tente via shards
        base = p.stem
        shard_paths = sorted(dirp.glob(f"{base}_part*{suffix}"))
        if shard_paths:
            return _concat_shards(shard_paths)
        raise


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    xv = x - x.mean()
    yv = y - y.mean()
    denom = xv.std() * yv.std()
    return float((xv * yv).mean() / denom) if denom > 0 else 0.0


def _load_all(split: str, prefix: str) -> dict:
    """Charge et concatène tous les lots disponibles pour un split donné.

    Gère les shards (_part*.pickle) et les fichiers simples.
    """
    base_glob = f"datasets/{prefix}_{split}_*.pickle"
    paths = [Path(p) for p in glob.glob(base_glob) if "scaled" not in p]
    if not paths:
        raise FileNotFoundError(f"Aucun pickle trouvé pour split={split}")
    # Grouper par base (sans _part...)
    groups: dict[str, list[Path]] = {}
    for p in paths:
        name = p.name
        if "_part" in name:
            base = name.split("_part", 1)[0] + p.suffix
        else:
            base = name
        groups.setdefault(base, []).append(p)

    datasets = []
    for base, files in sorted(groups.items()):
        # Priorité aux shards s'ils existent
        shard_files = [f for f in files if "_part" in f.name]
        main_file = next((f for f in files if "_part" not in f.name), None)
        if shard_files:
            shard_files = sorted(shard_files, key=lambda x: x.name)
            ds = _load_dataset(
                str(shard_files[0])
            )  # _load_dataset agrège tout le groupe
            datasets.append(ds)
        elif main_file and main_file.stat().st_size > 0:
            ds = _load_dataset(str(main_file))
            datasets.append(ds)
        else:
            # ignore fichiers 0B/corrompus
            continue

    if not datasets:
        raise FileNotFoundError(f"Aucun dataset chargeable pour split={split}")

    keys = [k for k in datasets[0].keys() if k != "meta"]
    merged = {k: datasets[0][k] for k in keys}
    for d in datasets[1:]:
        for k in keys:
            merged[k] = np.concatenate([merged[k], d[k]], axis=0)
    # Meta: conserver la dernière et documenter la fusion
    meta = datasets[-1].get("meta", {}).copy()
    meta["_merged_batches"] = sorted(groups.keys())
    merged["meta"] = meta
    return merged


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prefix",
        default="ticker_dataset",
        help="Préfixe des pickles à évaluer (def=ticker_dataset)",
    )
    p.add_argument(
        "--lags",
        default="0,6,12,24,48,96",
        help=(
            "Décalages intra-fenêtre à tester pour les corrélations "
            "(indices depuis la fin de fenêtre)"
        ),
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="Chemin d'export JSON du rapport (facultatif)",
    )
    # Baseline séquentielle optionnelle
    p.add_argument(
        "--seq-baseline",
        default="none",
        choices=["none", "lstm"],
        help="Ajoute une baseline séquentielle (ex: lstm) pour y_bin",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Époques pour la baseline séquentielle",
    )
    p.add_argument("--hidden", type=int, default=64, help="Taille cachée LSTM")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout LSTM")
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size pour l'entraînement LSTM",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate pour l'optimiseur LSTM",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    prefix = args.prefix

    # Charge et concatène tous les lots disponibles
    tr = _load_all("train", prefix)
    va = _load_all("val", prefix)

    Xtr: np.ndarray = tr["X"]
    Xva: np.ndarray = va["X"]
    y_cls_tr: np.ndarray = tr["y_cls"]
    y_cls_va: np.ndarray = va["y_cls"]
    y_reg_tr: np.ndarray = tr["y_reg"]
    y_reg_va: np.ndarray = va["y_reg"]
    y_vol_tr: np.ndarray = tr["y_vol"]
    y_vol_va: np.ndarray = va["y_vol"]
    y_bin_tr = tr.get("y_bin")
    y_bin_va = va.get("y_bin")

    meta = tr.get("meta", {}) if isinstance(tr, dict) else {}
    feature_names = meta.get("feature_names") or DEFAULT_FEATURE_NAMES

    # Dernière timestep comme features simples (lag=0)
    FTR = Xtr[:, -1, :]
    FVA = Xva[:, -1, :]
    nfeat = FTR.shape[1]
    if not feature_names:
        feature_names = [f"f{i}" for i in range(nfeat)]
    elif len(feature_names) != nfeat:
        print(
            f"[WARN] feature_names({len(feature_names)}) ≠ nfeat({nfeat}), "
            "fallback f0..fN"
        )
        feature_names = [f"f{i}" for i in range(nfeat)]

    # Statuts et distribution
    hist_tr = np.bincount(y_cls_tr + 1, minlength=3)
    hist_va = np.bincount(y_cls_va + 1, minlength=3)
    hist_bin_tr = (
        np.bincount(y_bin_tr + 1, minlength=3) if y_bin_tr is not None else None
    )
    hist_bin_va = (
        np.bincount(y_bin_va + 1, minlength=3) if y_bin_va is not None else None
    )
    maj_rate = hist_tr.max() / len(y_cls_tr)
    print(f"Train shape={Xtr.shape}, Val shape={Xva.shape}")
    print(
        f"Class hist train={hist_tr.tolist()} val={hist_va.tolist()} "
        f"(majority={maj_rate:.3f})"
    )

    # Corrélations (Pearson) simples avec chaque cible
    corr_cls = [_corr(FTR[:, i], y_cls_tr) for i in range(nfeat)]
    corr_reg = [_corr(FTR[:, i], y_reg_tr) for i in range(nfeat)]
    corr_vol = [_corr(FTR[:, i], y_vol_tr) for i in range(nfeat)]

    def _topk(corrs: list[float], k: int = 8):
        idx = np.argsort(np.abs(corrs))[::-1][:k]
        res = []
        for i in idx:
            name = feature_names[i] if i < len(feature_names) else f"f{i}"
            res.append((int(i), name, float(corrs[i])))
        return res

    print("\nTop |corr| with y_cls:")
    for i, name, c in _topk(corr_cls):
        print(f"  #{i:02d} {name:<18} corr={c:+.4f}")

    print("\nTop |corr| with y_reg:")
    for i, name, c in _topk(corr_reg):
        print(f"  #{i:02d} {name:<18} corr={c:+.4f}")

    print("\nTop |corr| with y_vol:")
    for i, name, c in _topk(corr_vol):
        print(f"  #{i:02d} {name:<18} corr={c:+.4f}")

    if y_bin_tr is not None and y_bin_va is not None:
        corr_bin = [_corr(FTR[:, i], y_bin_tr) for i in range(nfeat)]
        print("\nTop |corr| with y_bin:")
        for i, name, c in _topk(corr_bin):
            print(f"  #{i:02d} {name:<18} corr={c:+.4f}")

    # Cross-corr intra-fenêtre: tester plusieurs lags (features à l'instant -lag)
    try:
        lag_list = [int(x) for x in str(args.lags).split(",") if x.strip() != ""]
    except ValueError:
        lag_list = [0, 6, 12, 24, 48, 96]
    lag_list = [lag for lag in lag_list if lag >= 0]

    def _report_lag_corr(target_name: str, y_tr: np.ndarray):
        print(f"\nCross-correlations vs {target_name} (by intra-window lag):")
        for lag in lag_list:
            if lag >= Xtr.shape[1]:
                continue
            F_lag = Xtr[:, -(lag + 1), :]
            corrs = [_corr(F_lag[:, i], y_tr) for i in range(nfeat)]
            tops = _topk(corrs)
            tops_fmt = ", ".join([f"#{i}:{name}({c:+.3f})" for i, name, c in tops])
            print(f"  lag={lag:>3d} → {tops_fmt}")

    _report_lag_corr("y_cls", y_cls_tr)
    if y_bin_tr is not None:
        _report_lag_corr("y_bin", y_bin_tr)

    # Spearman (approx) = Pearson sur rangs (gestion simple des ex-æquo)
    def _ranks(a: np.ndarray) -> np.ndarray:
        # rangs 0..N-1; gestion simple des ties par rang moyen
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        return ranks

    sp_cls = [
        _corr(_ranks(FTR[:, i]), _ranks(y_cls_tr.astype(np.float64)))
        for i in range(nfeat)
    ]
    sp_reg = [
        _corr(_ranks(FTR[:, i]), _ranks(y_reg_tr.astype(np.float64)))
        for i in range(nfeat)
    ]
    print("\nTop |Spearman| with y_cls:")
    for i, name, c in _topk(sp_cls):
        print(f"  #{i:02d} {name:<18} corr={c:+.4f}")
    print("\nTop |Spearman| with y_reg:")
    for i, name, c in _topk(sp_reg):
        print(f"  #{i:02d} {name:<18} corr={c:+.4f}")
    if y_bin_tr is not None:
        sp_bin = [
            _corr(_ranks(FTR[:, i]), _ranks(y_bin_tr.astype(np.float64)))
            for i in range(nfeat)
        ]
        print("\nTop |Spearman| with y_bin:")
        for i, name, c in _topk(sp_bin):
            print(f"  #{i:02d} {name:<18} corr={c:+.4f}")

    # Information mutuelle (échantillon pour rapidité)
    rng = np.random.default_rng(123)

    def _subsample(X: np.ndarray, y: np.ndarray, max_n: int = 20000):
        n = len(y)
        if n <= max_n:
            return X, y
        idx = rng.choice(n, size=max_n, replace=False)
        return X[idx], y[idx]

    Xmi, y_cls_mi = _subsample(FTR, y_cls_tr)
    mi_cls = mutual_info_classif(
        Xmi, y_cls_mi, discrete_features=False, random_state=42
    )
    print("\nTop MI with y_cls:")
    for i, name, _ in _topk(mi_cls.tolist()):
        print(f"  #{i:02d} {name:<18} MI={mi_cls[i]:.4f}")

    Xmir, y_reg_mi = _subsample(FTR, y_reg_tr)
    mi_reg = mutual_info_regression(Xmir, y_reg_mi, random_state=42)
    print("\nTop MI with y_reg:")
    for i, name, _ in _topk(mi_reg.tolist()):
        print(f"  #{i:02d} {name:<18} MI={mi_reg[i]:.4f}")

    if y_bin_tr is not None:
        Xmi2, y_bin_mi = _subsample(FTR, y_bin_tr)
        mi_bin = mutual_info_classif(
            Xmi2, y_bin_mi, discrete_features=False, random_state=42
        )
        print("\nTop MI with y_bin:")
        for i, name, _ in _topk(mi_bin.tolist()):
            print(f"  #{i:02d} {name:<18} MI={mi_bin[i]:.4f}")

    # Baselines rapides
    print("\nBaselines (features = dernière timestep):")
    clf = LogisticRegression(max_iter=300, n_jobs=1)
    clf.fit(FTR, y_cls_tr)
    pred_cls = clf.predict(FVA)
    acc = accuracy_score(y_cls_va, pred_cls)
    macro_f1 = f1_score(y_cls_va, pred_cls, average="macro")
    print(
        f"  LogReg y_cls  → acc={acc:.3f}  macroF1={macro_f1:.3f}  "
        f"(baseline majority={maj_rate:.3f})"
    )

    if y_bin_tr is not None and y_bin_va is not None:
        clf_bin = LogisticRegression(max_iter=300, n_jobs=1)
        clf_bin.fit(FTR, y_bin_tr)
        pred_bin = clf_bin.predict(FVA)
        acc_b = accuracy_score(y_bin_va, pred_bin)
        macro_f1_b = f1_score(y_bin_va, pred_bin, average="macro")
        print(f"  LogReg y_bin  → acc={acc_b:.3f}  macroF1={macro_f1_b:.3f}")

    rg = Ridge(alpha=1.0).fit(FTR, y_reg_tr)
    pr = rg.predict(FVA)
    print(
        f"  Ridge  y_reg  → R2={r2_score(y_reg_va, pr):.3f}  "
        f"MAE={mean_absolute_error(y_reg_va, pr):.5f}"
    )

    rv = Ridge(alpha=1.0).fit(FTR, y_vol_tr)
    pv = rv.predict(FVA)
    print(
        f"  Ridge  y_vol  → R2={r2_score(y_vol_va, pv):.3f}  "
        f"MAE={mean_absolute_error(y_vol_va, pv):.5f}"
    )

    # Baseline séquentielle LSTM (optionnelle) sur y_bin
    seq_report = {}
    if args.seq_baseline == "lstm" and y_bin_tr is not None and y_bin_va is not None:
        if not TORCH_AVAILABLE:
            print("[WARN] PyTorch indisponible: baseline LSTM ignorée.")
        else:
            # Standardisation par feature sur tout le train (tous les pas de temps)
            Xtr2d = Xtr.reshape(-1, Xtr.shape[2])
            mu = Xtr2d.mean(axis=0)
            sd = Xtr2d.std(axis=0) + 1e-9
            Xtr_s = (Xtr - mu) / sd
            Xva_s = (Xva - mu) / sd

            # Tensors
            Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32)
            Xva_t = torch.tensor(Xva_s, dtype=torch.float32)
            ytr = (y_bin_tr + 1).astype(np.int64)  # -1,0,1 → 0,1,2
            yva = (y_bin_va + 1).astype(np.int64)
            ytr_t = torch.tensor(ytr, dtype=torch.long)
            yva_t = torch.tensor(yva, dtype=torch.long)

            train_ds = TensorDataset(Xtr_t, ytr_t)
            val_ds = TensorDataset(Xva_t, yva_t)
            train_loader = DataLoader(
                train_ds, batch_size=int(args.batch_size), shuffle=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=int(args.batch_size), shuffle=False
            )

            C = 3
            F = Xtr.shape[2]
            H = int(args.hidden)
            D = float(args.dropout)

            class_counts = np.bincount(ytr, minlength=C)
            w = class_counts.sum() / (class_counts + 1e-9)
            w = w / w.sum()
            class_w = torch.tensor(w, dtype=torch.float32)

            class LSTMCls(nn.Module):
                def __init__(self, fdim: int, hidden: int, ncls: int, dropout: float):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=fdim,
                        hidden_size=hidden,
                        num_layers=1,
                        batch_first=True,
                    )
                    self.drop = nn.Dropout(dropout)
                    self.out = nn.Linear(hidden, ncls)

                def forward(self, x):  # x: (B,T,F)
                    h, _ = self.lstm(x)
                    pooled = h[:, -1, :]
                    return self.out(self.drop(pooled))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = LSTMCls(F, H, C, D).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
            crit = nn.CrossEntropyLoss(weight=class_w.to(device))

            def _eval(loader):
                model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for xb, yb in loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        lg = model(xb)
                        pred = lg.argmax(dim=1).cpu().numpy()
                        preds.append(pred)
                        trues.append(yb.cpu().numpy())
                if preds:
                    p = np.concatenate(preds)
                    t = np.concatenate(trues)
                    p3 = p - 1  # 0,1,2 → -1,0,1
                    t3 = t - 1
                    return (
                        float(accuracy_score(t3, p3)),
                        float(f1_score(t3, p3, average="macro")),
                    )
                return 0.0, 0.0

            best_f1 = 0.0
            for ep in range(int(args.epochs)):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    lg = model(xb)
                    loss = crit(lg, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                acc_v, f1_v = _eval(val_loader)
                best_f1 = max(best_f1, f1_v)
                print(
                    f"  [LSTM] epoch {ep + 1}/{args.epochs} → "
                    f"acc={acc_v:.3f} macroF1={f1_v:.3f}"
                )

            acc_final, f1_final = _eval(val_loader)
            seq_report = {"y_bin": {"lstm": {"acc": acc_final, "macro_f1": f1_final}}}

    # Optionnel: export JSON condensé
    if args.json_out:
        report = {
            "shapes": {"train": list(Xtr.shape), "val": list(Xva.shape)},
            "class_hist": {
                "y_cls": {"train": hist_tr.tolist(), "val": hist_va.tolist()},
                **(
                    {
                        "y_bin": {
                            "train": hist_bin_tr.tolist(),
                            "val": hist_bin_va.tolist(),
                        }
                    }
                    if hist_bin_tr is not None and hist_bin_va is not None
                    else {}
                ),
            },
            "top": {
                "pearson": {
                    "y_cls": [
                        (int(i), feature_names[i], float(corr_cls[i]))
                        for i, _, _ in _topk(corr_cls)
                    ],
                    "y_reg": [
                        (int(i), feature_names[i], float(corr_reg[i]))
                        for i, _, _ in _topk(corr_reg)
                    ],
                    "y_vol": [
                        (int(i), feature_names[i], float(corr_vol[i]))
                        for i, _, _ in _topk(corr_vol)
                    ],
                },
                "spearman": {
                    "y_cls": [
                        (int(i), feature_names[i], float(sp_cls[i]))
                        for i, _, _ in _topk(sp_cls)
                    ],
                    "y_reg": [
                        (int(i), feature_names[i], float(sp_reg[i]))
                        for i, _, _ in _topk(sp_reg)
                    ],
                    **(
                        {
                            "y_bin": [
                                (int(i), feature_names[i], float(sp_bin[i]))
                                for i, _, _ in _topk(sp_bin)
                            ]
                        }
                        if y_bin_tr is not None
                        else {}
                    ),
                },
                "mi": {
                    "y_cls": [
                        (int(i), feature_names[i], float(mi_cls[i]))
                        for i, _, _ in _topk(mi_cls.tolist())
                    ],
                    "y_reg": [
                        (int(i), feature_names[i], float(mi_reg[i]))
                        for i, _, _ in _topk(mi_reg.tolist())
                    ],
                },
            },
            "baselines": {
                "y_cls": {
                    "acc": float(acc),
                    "macro_f1": float(macro_f1),
                    "majority": float(maj_rate),
                },
                "y_reg": {
                    "r2": float(r2_score(y_reg_va, pr)),
                    "mae": float(mean_absolute_error(y_reg_va, pr)),
                },
                "y_vol": {
                    "r2": float(r2_score(y_vol_va, pv)),
                    "mae": float(mean_absolute_error(y_vol_va, pv)),
                },
            },
        }
        if y_bin_tr is not None and y_bin_va is not None:
            report["top"]["pearson"]["y_bin"] = [
                (int(i), feature_names[i], float(corr_bin[i]))
                for i, _, _ in _topk(corr_bin)
            ]
            report["top"]["mi"]["y_bin"] = [
                (int(i), feature_names[i], float(mi_bin[i]))
                for i, _, _ in _topk(mi_bin.tolist())
            ]
            report["baselines"]["y_bin"] = {
                "acc": float(acc_b),
                "macro_f1": float(macro_f1_b),
            }
            if seq_report:
                report["seq_baselines"] = seq_report

        try:
            import json

            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
            print(f"\nRapport JSON écrit → {args.json_out}")
        except OSError as exc:
            print(f"[WARN] Échec écriture JSON: {exc}")


if __name__ == "__main__":
    main()
