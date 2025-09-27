"""logger_config.py"""

from __future__ import annotations

import glob

import json

import pickle

from pathlib import Path

from typing import Optional, Dict, Union, Tuple

import numpy as np

import torch

from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler  # type: ignore

OUTPUT_DIR = Path("./datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANGES_PATH = OUTPUT_DIR / "scaler_ranges.json"


def _to_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Torch → NumPy sans lien au graphe de calcul."""

    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()

    return arr


def _to_same_type(arr_np: np.ndarray, ref: Union[np.ndarray, torch.Tensor]):
    """Reconvertit en Torch si `ref` était Torch, sinon laisse en NumPy."""

    if isinstance(ref, torch.Tensor):
        return torch.from_numpy(arr_np).to(ref.device).type_as(ref)

    return arr_np


class CryptoDataset(Dataset):
    def __init__(self, X, y_cls, y_vol, y_reg):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)

        self.y_vol = torch.tensor(y_vol, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            {
                "cls": self.y_cls[idx],
                "reg": self.y_reg[idx],
                "vol": self.y_vol[idx],
            },
        )

    def __len__(self):
        return len(self.X)


def _load_pickle_dataset(path: str | Path) -> CryptoDataset:
    with open(path, "rb") as fh:
        payload = pickle.load(fh)

    if isinstance(payload, CryptoDataset):
        return payload

    if isinstance(payload, dict):
        required_keys = {"X", "y_cls", "y_vol", "y_reg"}
        if not required_keys.issubset(payload):
            missing = required_keys.difference(payload)
            raise KeyError(f"Clés manquantes dans {path}: {sorted(missing)}")
        return CryptoDataset(
            payload["X"], payload["y_cls"], payload["y_vol"], payload["y_reg"]
        )

    raise TypeError(f"Format de pickle inattendu pour {path}: {type(payload).__name__}")


def scale(
    X: Union[np.ndarray, torch.Tensor],
    is_use_feature_ranges: bool = True,
    feature_ranges_forced: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Dict[int, Tuple[float, float]]]]:
    """
    Mise à l'échelle feature-wise dans [0, 1].

    - Si `is_use_feature_ranges` est True, on prend les bornes pré-enregistrées.
    - Sinon, on calcule min / max sur X et on les affiche pour archivage.

    Garde le même type en sortie qu'en entrée (NumPy <-> Torch).
    """
    # 1) Sauvegarde du type d'origine et conversion NumPy
    X_np = _to_numpy(X)
    n, t, f = X_np.shape
    X_2d = X_np.reshape(-1, f)  # (N*T, F)

    # 2) Détermination des bornes
    if is_use_feature_ranges:
        if feature_ranges_forced is not None:
            feature_ranges = feature_ranges_forced

        feature_min = np.array(
            [feature_ranges[i][0] for i in range(f)], dtype=np.float32
        )
        feature_max = np.array(
            [feature_ranges[i][1] for i in range(f)], dtype=np.float32
        )
    else:
        feature_min = X_2d.min(axis=0).astype(np.float32)
        feature_max = X_2d.max(axis=0).astype(np.float32)

        # Affiche les nouvelles bornes
        new_ranges = {
            i: (float(feature_min[i]), float(feature_max[i])) for i in range(f)
        }
        print("Nouveau feature_ranges à sauvegarder :")
        print(new_ranges)

        feature_ranges = new_ranges

    # 3) Construction manuelle du MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.n_features_in_ = f
    scaler.data_min_ = feature_min
    scaler.data_max_ = feature_max
    scaler.data_range_ = np.where(
        feature_max - feature_min == 0, 1.0, feature_max - feature_min
    )
    scaler.scale_ = 1.0 / scaler.data_range_
    scaler.min_ = -feature_min * scaler.scale_

    # 4) Transformation puis remise en forme
    X_scaled_2d = scaler.transform(X_2d)
    X_scaled_np = X_scaled_2d.reshape(n, t, f)

    # 5) Retour au type d'origine
    return _to_same_type(X_scaled_np, X), feature_ranges


def scale_target(
    y: Union[np.ndarray, torch.Tensor],
    key: str,  # "ret" ou "vol"
    is_use_saved_ranges: bool = True,
    target_ranges_forced: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Dict[str, Tuple[float, float]]]]:
    """
    Met y dans [0,1] en conservant le type. Si `is_use_saved_ranges=False`,
    calcule min/max, affiche un dict qu'on pourra coller dans `target_ranges`.
    """
    assert key in ("ret", "vol"), "key doit être 'ret' ou 'vol'"
    y_np = _to_numpy(y).reshape(-1, 1)  # (N,1)

    ret_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    if is_use_saved_ranges:
        if target_ranges_forced is None:
            raise ValueError(
                "target_ranges_forced must be provided when is_use_saved_ranges=True"
            )
        tr = target_ranges_forced
        y_min, y_max = tr[key]
        ret_ranges = target_ranges_forced
    else:
        y_min, y_max = float(y_np.min()), float(y_np.max())
        print(
            f"--> nouveau range pour '{key}': ({y_min}, {y_max}) "
            "→ ajoutez-le dans target_ranges"
        )
        ret_ranges = {key: (y_min, y_max)}

    # Min-Max scaling manuel (évite de ré-instancier un MinMaxScaler)
    denom = y_max - y_min if y_max != y_min else 1.0
    y_scaled_np = (y_np - y_min) / denom
    y_scaled_np = np.clip(y_scaled_np, 0.0, 1.0).reshape(-1)  # (N,)

    return _to_same_type(y_scaled_np, y), ret_ranges


def scale_dataset():
    # 1) On charge et concatène tous les splits train
    train_files = sorted(glob.glob(str(OUTPUT_DIR / "ticker_dataset_train_*.pickle")))
    X_list, cls_list, vol_list, reg_list = [], [], [], []
    for path in train_files:
        ds = _load_pickle_dataset(path)
        X_list.append(ds.X)
        cls_list.append(ds.y_cls)
        vol_list.append(ds.y_vol)
        reg_list.append(ds.y_reg)
    X_train = torch.cat(X_list, dim=0)
    y_cls_train = torch.cat(cls_list, dim=0)
    y_vol_train = torch.cat(vol_list, dim=0)
    y_reg_train = torch.cat(reg_list, dim=0)
    print(f"→ Train total shape: {X_train.shape}")

    # 2) Même chose pour le val
    val_files = sorted(glob.glob(str(OUTPUT_DIR / "ticker_dataset_val_*.pickle")))
    X_list, cls_list, vol_list, reg_list = [], [], [], []
    for path in val_files:
        ds = _load_pickle_dataset(path)
        X_list.append(ds.X)
        cls_list.append(ds.y_cls)
        vol_list.append(ds.y_vol)
        reg_list.append(ds.y_reg)
    X_val = torch.cat(X_list, dim=0)
    y_cls_val = torch.cat(cls_list, dim=0)
    y_vol_val = torch.cat(vol_list, dim=0)
    y_reg_val = torch.cat(reg_list, dim=0)
    print(f"→ Val   total shape: {X_val.shape}")

    # 3) On re-emballe dans CryptoDataset
    ticker_dataset_train = CryptoDataset(
        X_train.numpy(), y_cls_train.numpy(), y_vol_train.numpy(), y_reg_train.numpy()
    )
    ticker_dataset_val = CryptoDataset(
        X_val.numpy(), y_cls_val.numpy(), y_vol_val.numpy(), y_reg_val.numpy()
    )

    # 4) Ranges basés uniquement sur le train --------------------
    scaled_train_X, feature_ranges_X = scale(
        X=ticker_dataset_train.X,
        is_use_feature_ranges=False,
    )
    ticker_dataset_train.X = torch.as_tensor(scaled_train_X, dtype=torch.float32)
    scaled_val_X, _ = scale(
        X=ticker_dataset_val.X,
        is_use_feature_ranges=True,
        feature_ranges_forced=feature_ranges_X,
    )
    ticker_dataset_val.X = torch.as_tensor(scaled_val_X, dtype=torch.float32)

    scaled_train_y_reg, target_ranges_y_ret = scale_target(
        y=ticker_dataset_train.y_reg,
        key="ret",
        is_use_saved_ranges=False,
    )
    ticker_dataset_train.y_reg = torch.as_tensor(
        scaled_train_y_reg, dtype=torch.float32
    )
    scaled_val_y_reg, _ = scale_target(
        y=ticker_dataset_val.y_reg,
        key="ret",
        is_use_saved_ranges=True,
        target_ranges_forced=target_ranges_y_ret,
    )
    ticker_dataset_val.y_reg = torch.as_tensor(scaled_val_y_reg, dtype=torch.float32)

    scaled_train_y_vol, target_ranges_y_vol = scale_target(
        y=ticker_dataset_train.y_vol,
        key="vol",
        is_use_saved_ranges=False,
    )
    ticker_dataset_train.y_vol = torch.as_tensor(
        scaled_train_y_vol, dtype=torch.float32
    )
    scaled_val_y_vol, _ = scale_target(
        y=ticker_dataset_val.y_vol,
        key="vol",
        is_use_saved_ranges=True,
        target_ranges_forced=target_ranges_y_vol,
    )
    ticker_dataset_val.y_vol = torch.as_tensor(scaled_val_y_vol, dtype=torch.float32)

    if (
        feature_ranges_X is None
        or target_ranges_y_ret is None
        or target_ranges_y_vol is None
    ):
        raise RuntimeError("Impossible de calculer les bornes de scaling.")

    feature_ranges_serializable = {
        str(idx): [float(bounds[0]), float(bounds[1])]
        for idx, bounds in feature_ranges_X.items()
    }
    target_ranges_serializable = {
        "ret": [float(v) for v in target_ranges_y_ret["ret"]],
        "vol": [float(v) for v in target_ranges_y_vol["vol"]],
    }
    ranges_payload = {
        "feature_ranges": feature_ranges_serializable,
        "target_ranges": target_ranges_serializable,
    }
    RANGES_PATH.write_text(json.dumps(ranges_payload, indent=2))

    # 6) On vérifie
    train_min, train_max = ticker_dataset_train.X.min(), ticker_dataset_train.X.max()
    val_min, val_max = ticker_dataset_val.X.min(), ticker_dataset_val.X.max()
    print(f"Après scaling  → Train X ∈ [{train_min:.3f}, {train_max:.3f}]")
    print(f"               Val   X ∈ [{val_min:.3f}, {val_max:.3f}]")

    # 7) Sauvegarde
    with open(OUTPUT_DIR / "ticker_dataset_train_scaled.pickle", "wb") as f:
        pickle.dump(ticker_dataset_train, f)
    with open(OUTPUT_DIR / "ticker_dataset_val_scaled.pickle", "wb") as f:
        pickle.dump(ticker_dataset_val, f)

    # 8) Re-chargement pour sanity check
    td_tr = pickle.load(open(OUTPUT_DIR / "ticker_dataset_train_scaled.pickle", "rb"))
    td_va = pickle.load(open(OUTPUT_DIR / "ticker_dataset_val_scaled.pickle", "rb"))
    print(f"Re-load shapes: train {td_tr.X.shape}, val {td_va.X.shape}")


if __name__ == "__main__":
    scale_dataset()
