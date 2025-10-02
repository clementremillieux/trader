"""Script de création de jeux de données Binance pour le trading crypto."""

from __future__ import annotations

import argparse

import asyncio

from dataclasses import dataclass, field, asdict

from collections import Counter

from datetime import datetime, timezone, timedelta

import os

from pathlib import Path

import pickle

import random

from typing import Optional, List, Dict, Any, Coroutine, Tuple, cast

import json

import math


import httpx

import certifi

import pandas as pd  # type: ignore

import numpy as np

from numpy.typing import NDArray

import torch

from torch.utils.data import Dataset


from config.logger_config import logger


os.environ["CURL_CA_BUNDLE"] = certifi.where()


BASE_URL = "https://api.binance.com"
MAX_CONC = 6
DEFAULT_TIMEOUT = 30.0
DEFAULT_CACHE_TTL_HOURS = int(os.getenv("BINANCE_CACHE_TTL_HOURS", "6"))
CACHE_DIR = Path(".cache_binance")
CACHE_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS: list[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "log_ret",
    "log_ret_4h",
    "log_ret_24h",
    "volatility_24h",
    "volatility_6h",
    "volatility_1h",
    "atr",
    "price_range",
    "momentum",
    "rsi",
    "macd_diff",
    "stoch_k",
    "bollinger_width",
    "volume",
    "v_rel",
    "imbalance",
    "buy_pressure",
    "rv",
    "btc_log_ret_1h",
    "btc_log_ret_4h",
    "eth_log_ret_1h",
    "usdt_volume_rel",
    # Nouvelles features directionnelles utiles aux modèles séquentiels
    "vwap_ratio_24h",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "body_to_range",
    "total_wick_to_range",
    "close_rank_24h",
    "trend_slope_log_24h",
    "ret_zscore_24h",
    "ret_to_atr",
    # contexte marché additionnel
    "corr_ret_btc_24h",
    "corr_ret_eth_24h",
    "taker_buy_quote_ratio",
    # nouvelles features (volatilité basée sur les ranges et bêta marché)
    "parkinson_vol_24h",
    "gk_vol_24h",
    "rs_vol_24h",
    "beta_btc_24h",
    "beta_eth_24h",
    # indicateurs additionnels (légers) calculés sur OHLCV
    "adx_14",
    "mfi_14",
    "cci_20",
    "williamsr_14",
    "chaikin_osc_3_10",
    "obv_z_24h",
    # moyennes roulantes des indicateurs de flux
    "tbqr_ma_24h",
    "buy_press_ma_24h",
    # nouvelles features multi-échelles
    "rv_6h",
    "rv_1h",
    "ofi_1h",
    "ofi_6h",
    "volume_zscore_24h",
]


class _ClientHolder:
    client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    if _ClientHolder.client is None:
        _ClientHolder.client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(DEFAULT_TIMEOUT),
            limits=httpx.Limits(max_connections=MAX_CONC),
            headers={"User-Agent": "trader-dataset-builder/1.0"},
            http2=True,
        )
    return _ClientHolder.client


async def _close_client() -> None:
    if _ClientHolder.client is not None:
        await _ClientHolder.client.aclose()
        _ClientHolder.client = None


def _cache_path(symbol: str, interval: str, bar_needed: int | None) -> Path:
    safe_symbol = symbol.replace("/", "-")
    suffix = "full" if bar_needed is None else str(bar_needed)
    return CACHE_DIR / f"{safe_symbol}_{interval}_{suffix}.pkl"


def _cache_is_fresh(path: Path, ttl_hours: Optional[int]) -> bool:
    if ttl_hours is None:
        return True
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except FileNotFoundError:
        return False
    age = datetime.now(tz=timezone.utc) - modified
    return age <= timedelta(hours=ttl_hours)


def _read_cached_df(path: Path) -> pd.DataFrame | None:
    try:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
    except FileNotFoundError:
        return None
    except (
        OSError,
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
    ) as exc:  # pragma: no cover - log and continue
        logger.warning("Cache read error for %s: %s", path, exc)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None

    if isinstance(cached, pd.DataFrame):
        return cached
    logger.debug("Cache %s contains unexpected type %s", path, type(cached))
    return None


def _write_cache_df(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as fh:
            pickle.dump(df, fh)
        tmp.replace(path)
    except OSError as exc:  # pragma: no cover - log and continue
        logger.warning("Unable to persist cache %s: %s", path, exc)
        if tmp.exists():
            tmp.unlink()


def _load_feature_thresholds(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        logger.warning("Feature thresholds file %s not found", path)
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read feature thresholds %s: %s", path, exc)
        return {}

    if not isinstance(raw, dict):
        logger.warning("Feature thresholds must be a mapping, got %s", type(raw))
        return {}

    allowed_keys = {"min", "max", "mean", "std"}
    cleaned: Dict[str, Dict[str, float]] = {}
    for feature, spec in raw.items():
        if not isinstance(spec, dict):
            logger.debug(
                "Skip thresholds for %s (expected dict, got %s)", feature, type(spec)
            )
            continue
        cleaned_spec: Dict[str, float] = {}
        for key, value in spec.items():
            if key not in allowed_keys:
                continue
            if not isinstance(value, (int, float)):
                continue
            cleaned_spec[key] = float(value)
        if cleaned_spec:
            cleaned[feature] = cleaned_spec
    return cleaned


@dataclass
class DatasetBuildConfig:
    """Configuration complète pour la génération de dataset."""

    window_size: int = 300
    interval: str = "1h"
    max_candles: int = 100_000
    split_hours: int = 1000
    purge_hours: int = 12
    tickers_per_batch: int = 20
    tickers_allow: Optional[str] = (
        None  # CSV de tickers à forcer (ex: ETHBTC,LTCBTC,...)
    )
    start_batch: int = 14
    max_batches: Optional[int] = None
    shuffle_seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("./datasets"))
    cache_ttl_hours: Optional[int] = DEFAULT_CACHE_TTL_HOURS
    disable_cache: bool = False
    resume: bool = False
    out_prefix: str = "ticker_dataset"
    min_train_windows: int = 800
    min_val_windows: int = 200
    dataset_version: str = "crypto_v3_dataset"
    feature_thresholds_path: Optional[Path] = None
    alert_tolerance: float = 0.0
    quality_report_path: Path = field(
        default_factory=lambda: Path("./datasets/quality_report.json")
    )
    # Nouveaux paramètres pour signaux directionnels et fenêtrage
    window_stride: int = 6
    bin_label_h: int = 24  # horizon pour le label binaire up/down
    bin_label_thr_atr: float = (
        1.0  # seuil en ATR pour considérer un mouvement significatif
    )
    produce_binary_cls: bool = True
    # Adapter automatiquement les fenêtres 1h/4h/24h aux intervalles ≠ 1h
    # (ex: 5m → 1h=12 barres, 24h=288 barres ; 1d → 24h=1 barre)
    time_aware_periods: bool = False
    # Limitation de la taille des datasets (cap sur le nombre de fenêtres)
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    min_volume_quantile: Optional[float] = 0.05
    volume_filter_hours: int = 12
    direction_margin: float = 0.0025

    # Nouveau paramètre : taille max d'un shard (nombre de lignes)
    max_rows_per_shard: Optional[int] = None

    @classmethod
    def from_cli(cls, argv: Optional[List[str]] = None) -> "DatasetBuildConfig":
        parser = argparse.ArgumentParser(
            description=(
                "Crée des jeux de données de trading crypto multi-tickers "
                "avec labels triple barrière."
            )
        )
        parser.add_argument(
            "--interval", default="1h", help="Intervalle Binance (ex: 1h)"
        )
        parser.add_argument(
            "--window",
            type=int,
            default=300,
            help="Taille de fenêtre temporelle utilisée pour les features",
        )
        parser.add_argument(
            "--max-candles",
            type=int,
            default=100_000,
            help="Nombre maximum de bougies récupérées par symbole",
        )
        parser.add_argument(
            "--split-hours",
            type=int,
            default=1000,
            help="Nombre d'heures réservées à la validation",
        )
        parser.add_argument(
            "--purge-hours",
            type=int,
            default=12,
            help="Nombre d'heures de buffer entre train/val",
        )
        parser.add_argument(
            "--tickers-per-batch",
            type=int,
            default=20,
            help="Nombre de tickers traités simultanément",
        )
        parser.add_argument(
            "--tickers-allow",
            default=None,
            help="Liste CSV de tickers à utiliser (ex: ETHBTC,LTCBTC,BNBBTC)",
        )
        parser.add_argument(
            "--start-batch",
            type=int,
            default=14,
            help="Indice de lot initial (permet de reprendre).",
        )
        parser.add_argument(
            "--max-batches",
            type=int,
            default=None,
            help="Nombre maximum de lots à traiter (None = tous).",
        )
        parser.add_argument(
            "--shuffle-seed",
            type=int,
            default=42,
            help="Graine pour le mélange des tickers",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("./datasets"),
            help="Répertoire de sortie des fichiers pickle",
        )
        parser.add_argument(
            "--disable-cache",
            action="store_true",
            help="Force la récupération complète des données sans cache",
        )
        parser.add_argument(
            "--cache-ttl",
            type=int,
            default=DEFAULT_CACHE_TTL_HOURS,
            help="Durée de vie du cache OHLC en heures (None = infini)",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Active la reprise automatique en sautant les lots déjà générés",
        )
        parser.add_argument(
            "--out-prefix",
            default="ticker_dataset",
            help="Préfixe des fichiers de sortie",
        )
        parser.add_argument(
            "--min-train-windows",
            type=int,
            default=800,
            help="Nombre minimum de fenêtres train par ticker pour être conservé",
        )
        parser.add_argument(
            "--min-val-windows",
            type=int,
            default=200,
            help="Nombre minimum de fenêtres validation par ticker",
        )
        parser.add_argument(
            "--quality-report",
            type=Path,
            default=Path("./datasets/quality_report.json"),
            help="Fichier JSON de rapport qualité (généré en sortie)",
        )
        parser.add_argument(
            "--dataset-version",
            default="crypto_v3_dataset",
            help="Identifiant de version à inscrire dans le rapport qualité",
        )
        parser.add_argument(
            "--feature-thresholds",
            type=Path,
            default=None,
            help="Fichier JSON décrivant les bornes attendues des features",
        )
        parser.add_argument(
            "--alert-tolerance",
            type=float,
            default=0.0,
            help="Tolérance absolue appliquée lors de la comparaison aux bornes",
        )
        parser.add_argument(
            "--window-stride",
            type=int,
            default=6,
            help="Stride entre fenêtres successives (par défaut 6)",
        )
        parser.add_argument(
            "--bin-label-h",
            type=int,
            default=24,
            help="Horizon (en barres) pour le label binaire up/down",
        )
        parser.add_argument(
            "--bin-label-thr-atr",
            type=float,
            default=1.0,
            help="Seuil en ATR (multiplicateur) pour définir up/down",
        )
        parser.add_argument(
            "--no-produce-binary-cls",
            action="store_true",
            help="Désactive la production du label binaire up/down",
        )
        parser.add_argument(
            "--time-aware-periods",
            action="store_true",
            help=(
                "Adapte les fenêtres 1h/4h/24h en nombre de barres selon l'intervalle"
            ),
        )
        parser.add_argument(
            "--max-train-samples",
            type=int,
            default=None,
            help="Nombre max de fenêtres à conserver pour le split train (cap).",
        )
        parser.add_argument(
            "--max-val-samples",
            type=int,
            default=None,
            help="Nombre max de fenêtres à conserver pour le split val (cap).",
        )
        parser.add_argument(
            "--min-volume-quantile",
            type=float,
            default=0.05,
            help=(
                "Quantile minimum de volume (0-1). Les fenêtres en dessous sont filtrées."
                " Valeur négative pour désactiver."
            ),
        )
        parser.add_argument(
            "--volume-filter-hours",
            type=int,
            default=12,
            help="Fenêtre en heures pour estimer le quantile de volume.",
        )
        parser.add_argument(
            "--direction-margin",
            type=float,
            default=0.0025,
            help=(
                "Marge minimale (rendement relatif) pour départager les mouvements "
                "haussiers et baissiers quand les deux barrières sont touchées. "
                "Augmentez pour filtrer les signaux ambigus."
            ),
        )

        parser.add_argument(
            "--max-rows-per-shard",
            type=int,
            default=None,
            help="Nombre max de lignes par shard pickle (prioritaire sur la taille auto).",
        )

        args = parser.parse_args(argv)

        return cls(
            window_size=args.window,
            interval=args.interval,
            max_candles=args.max_candles,
            split_hours=args.split_hours,
            purge_hours=args.purge_hours,
            tickers_per_batch=args.tickers_per_batch,
            tickers_allow=args.tickers_allow,
            start_batch=args.start_batch,
            max_batches=args.max_batches,
            shuffle_seed=args.shuffle_seed,
            output_dir=args.output_dir,
            cache_ttl_hours=None if args.cache_ttl is None else int(args.cache_ttl),
            disable_cache=args.disable_cache,
            resume=args.resume,
            out_prefix=args.out_prefix,
            min_train_windows=args.min_train_windows,
            min_val_windows=args.min_val_windows,
            dataset_version=args.dataset_version,
            feature_thresholds_path=args.feature_thresholds,
            alert_tolerance=args.alert_tolerance,
            quality_report_path=args.quality_report,
            window_stride=args.window_stride,
            bin_label_h=args.bin_label_h,
            bin_label_thr_atr=args.bin_label_thr_atr,
            produce_binary_cls=not args.no_produce_binary_cls,
            time_aware_periods=args.time_aware_periods,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            min_volume_quantile=(
                None
                if args.min_volume_quantile is None or args.min_volume_quantile < 0
                else args.min_volume_quantile
            ),
            volume_filter_hours=args.volume_filter_hours,
            direction_margin=args.direction_margin,
            max_rows_per_shard=args.max_rows_per_shard,
        )

    def dataset_path(self, split: str, batch_index: int) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{self.out_prefix}_{split}_{batch_index}.pickle"

    def next_batch_index(self) -> int:
        if not self.resume:
            return self.start_batch

        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing_idx: list[int] = []
        for file in self.output_dir.glob(f"{self.out_prefix}_train_*.pickle"):
            try:
                existing_idx.append(int(file.stem.split("_")[-1]))
            except ValueError:
                continue

        if not existing_idx:
            return self.start_batch
        next_idx = max(existing_idx) + 1
        logger.info("Reprise automatique: prochain lot identifié %d", next_idx)
        return next_idx


@dataclass
class DatasetQualityMetrics:
    count: int = 0
    class_hist: Counter = field(default_factory=Counter)
    reg_sum: float = 0.0
    reg_sq_sum: float = 0.0
    vol_sum: float = 0.0
    vol_sq_sum: float = 0.0
    feature_sum: torch.Tensor | None = field(default=None, repr=False)
    feature_sq_sum: torch.Tensor | None = field(default=None, repr=False)
    feature_min: torch.Tensor | None = field(default=None, repr=False)
    feature_max: torch.Tensor | None = field(default=None, repr=False)
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict, repr=False)
    alert_tolerance: float = 0.0

    def update(self, dataset: "CryptoDataset") -> None:
        y_cls = dataset.y_cls.cpu().numpy()
        y_reg = dataset.y_reg.cpu().numpy()
        y_vol = dataset.y_vol.cpu().numpy()

        self.count += int(y_cls.size)
        self.class_hist.update(int(v) for v in y_cls.tolist())
        self.reg_sum += float(y_reg.sum())
        self.reg_sq_sum += float(np.square(y_reg).sum())
        self.vol_sum += float(y_vol.sum())
        self.vol_sq_sum += float(np.square(y_vol).sum())

        if dataset.X.numel() == 0:
            return

        last_step = dataset.X[:, -1, :].detach().cpu().to(torch.float64)
        if last_step.numel() == 0:
            return

        step_sum = last_step.sum(dim=0)
        step_sq_sum = (last_step**2).sum(dim=0)
        step_min = last_step.min(dim=0).values
        step_max = last_step.max(dim=0).values

        if self.feature_sum is None:
            self.feature_sum = step_sum
            self.feature_sq_sum = step_sq_sum
            self.feature_min = step_min
            self.feature_max = step_max
        else:
            self.feature_sum += step_sum
            self.feature_sq_sum += step_sq_sum
            if self.feature_min is None:
                self.feature_min = step_min
            else:
                self.feature_min = torch.minimum(self.feature_min, step_min)
            if self.feature_max is None:
                self.feature_max = step_max
            else:
                self.feature_max = torch.maximum(self.feature_max, step_max)

    def to_dict(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "class_hist": {},
                "reg_mean": None,
                "reg_std": None,
                "vol_mean": None,
                "vol_std": None,
                "features": {},
            }

        reg_mean = self.reg_sum / self.count
        reg_var = max(self.reg_sq_sum / self.count - reg_mean * reg_mean, 0.0)
        vol_mean = self.vol_sum / self.count
        vol_var = max(self.vol_sq_sum / self.count - vol_mean * vol_mean, 0.0)

        features_payload: Dict[str, Dict[str, float]] = {}
        feature_alerts: list[Dict[str, Any]] = []
        if (
            self.feature_sum is not None
            and self.feature_sq_sum is not None
            and self.feature_min is not None
            and self.feature_max is not None
        ):
            feature_mean = self.feature_sum / self.count
            feature_var = torch.clamp(
                self.feature_sq_sum / self.count - feature_mean**2, min=0.0
            )
            feature_std = torch.sqrt(feature_var)
            feature_min = self.feature_min
            feature_max = self.feature_max
            for idx, name in enumerate(FEATURE_COLUMNS):
                features_payload[name] = {
                    "mean": float(feature_mean[idx]),
                    "std": float(feature_std[idx]),
                    "min": float(feature_min[idx]),
                    "max": float(feature_max[idx]),
                }
                expected = self.thresholds.get(name)
                if not expected:
                    continue
                tol = self.alert_tolerance
                stats = features_payload[name]
                if "min" in expected and stats["min"] < expected["min"] - tol:
                    feature_alerts.append(
                        {
                            "feature": name,
                            "metric": "min",
                            "observed": stats["min"],
                            "expected": expected["min"],
                        }
                    )
                if "max" in expected and stats["max"] > expected["max"] + tol:
                    feature_alerts.append(
                        {
                            "feature": name,
                            "metric": "max",
                            "observed": stats["max"],
                            "expected": expected["max"],
                        }
                    )
                if "mean" in expected and abs(stats["mean"] - expected["mean"]) > tol:
                    feature_alerts.append(
                        {
                            "feature": name,
                            "metric": "mean",
                            "observed": stats["mean"],
                            "expected": expected["mean"],
                        }
                    )
                if "std" in expected and abs(stats["std"] - expected["std"]) > tol:
                    feature_alerts.append(
                        {
                            "feature": name,
                            "metric": "std",
                            "observed": stats["std"],
                            "expected": expected["std"],
                        }
                    )

        return {
            "count": self.count,
            "class_hist": {str(k): int(v) for k, v in self.class_hist.items()},
            "reg_mean": reg_mean,
            "reg_std": math.sqrt(reg_var),
            "vol_mean": vol_mean,
            "vol_std": math.sqrt(vol_var),
            "features": features_payload,
            "alerts": feature_alerts,
        }


# ═════════════════ 2 · HTTP helper ═══════════════════════════════
async def _json(path: str, params: Dict[str, Any]) -> Any:
    params = {k: v for k, v in params.items() if v not in (None, "")}

    logger.debug("GET %s %s", path, params)

    client = _get_client()
    r = await client.get(path, params=params)

    r.raise_for_status()

    return r.json()


async def get_tickers() -> Tuple[List[str], Dict[str, str]]:
    """Retourne les tickers hors USDT qui disposent d'une paire libellée en USDT."""

    exchange_info: Dict[str, Any] = await _json("/api/v3/exchangeInfo", {})

    symbols = exchange_info.get("symbols", [])

    trading_pairs = [s for s in symbols if s.get("status") == "TRADING"]

    sym_base_asset: Dict[str, str] = {
        s["symbol"]: s["baseAsset"] for s in trading_pairs
    }

    tickers: List[str] = [s["symbol"] for s in trading_pairs]

    tickers_not_in_usdt = [ticker for ticker in tickers if not ticker.endswith("USDT")]

    tickers_not_in_usdt_but_exist_in_usdt = [
        ticker
        for ticker in tickers_not_in_usdt
        if f"{sym_base_asset[ticker]}USDT" in tickers
    ]

    return tickers_not_in_usdt_but_exist_in_usdt, sym_base_asset


# ──────────────────────────────────────────────────────────────
#  fetch_ohlc — retourne un DataFrame en ordre chronologique
#               avec l’open-time (indice 0) comme horodatage
# ──────────────────────────────────────────────────────────────
async def fetch_ohlc(
    sym: str,
    intv: str = "1h",
    bar_needed: int | None = None,
    *,
    use_cache: bool = True,
    cache_ttl_hours: Optional[int] = DEFAULT_CACHE_TTL_HOURS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    logger.info("↓ %-10s %s", sym, intv)

    cache_path = _cache_path(sym, intv, bar_needed)
    if use_cache and not force_refresh and cache_path.exists():
        if _cache_is_fresh(cache_path, cache_ttl_hours):
            cached_df = _read_cached_df(cache_path)
            if cached_df is not None:
                if bar_needed and len(cached_df) > bar_needed:
                    cached_df = cached_df.tail(bar_needed)
                logger.debug("Cache hit for %s %s (len=%d)", sym, intv, len(cached_df))
                return cached_df.copy()
        else:
            logger.debug("Cache expired for %s %s", sym, intv)

    rows: list[list[Any]] = []
    end_ms: int | None = None
    oldest_ts: int | None = None

    while True:
        params = {"symbol": sym, "interval": intv, "limit": 1000}
        if end_ms is not None:
            params["endTime"] = end_ms

        batch: list[list[Any]] = await _json("/api/v3/klines", params)
        if not batch:
            break

        first_open_time = batch[0][0]
        if oldest_ts is not None and first_open_time >= oldest_ts:
            # on vient de recevoir la même tranche → plus d'historique
            break
        oldest_ts = first_open_time

        rows.extend(batch)  # on accumule vers le passé
        if bar_needed and len(rows) >= bar_needed:
            break
        if len(batch) < 1000:  # début de l’historique
            break
        end_ms = first_open_time - 1  # reculer d’1 ms

    if not rows:
        raise RuntimeError(f"No data returned for {sym} {intv}")

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_vol",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    frame = pd.DataFrame(rows, columns=cols).drop(columns="ignore")

    # mise au bon type (float32) hors timestamps
    # type: ignore[reportUnknownMemberType]
    float_cols = frame.columns.difference(["open_time", "close_time"])
    frame = frame.astype({col: np.float32 for col in list(float_cols)})  # type: ignore

    # horodatage = open_time (utilise assign pour éviter l'affectation directe)
    frame = frame.assign(  # type: ignore
        ts=pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    )
    frame = (
        frame.drop_duplicates(subset="ts")
        .set_index("ts")
        .sort_index()
        .drop(columns=["open_time", "close_time"])
    )

    if bar_needed and len(frame) > bar_needed:
        frame = frame.tail(bar_needed)

    if use_cache:
        _write_cache_df(cache_path, frame)

    logger.info("%s %s → %d bougies uniques", sym, intv, len(frame))
    return frame


class CryptoDataset(Dataset):
    def __init__(
        self,
        X,
        y_cls,
        y_vol,
        y_reg,
        tau,
        class_weights: Optional[List[float]] = None,
        y_bin: Optional[np.ndarray] = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)

        self.y_vol = torch.tensor(y_vol, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.tau = torch.tensor(tau, dtype=torch.float32)
        self.class_weights = class_weights or []
        self.y_bin = (
            torch.tensor(y_bin, dtype=torch.long) if y_bin is not None else None
        )

    def __getitem__(self, idx):
        return (
            self.X[idx],
            {
                "cls": self.y_cls[idx],
                "reg": self.y_reg[idx],
                "vol": self.y_vol[idx],
                "tau": self.tau[idx],
                **({"bin": self.y_bin[idx]} if self.y_bin is not None else {}),
            },
        )

    def __len__(self):
        return len(self.X)


def safe_diff(s: pd.Series) -> np.ndarray:
    """Différence t-t-1, **décalée** pour ne pas utiliser close[t] dans diff[t]."""
    return s.diff().shift(1).fillna(0).to_numpy(dtype=np.float32)


def lagged_log_return(s: pd.Series, period: int = 1) -> np.ndarray:
    """Log-return t - t-period, décalé pour éviter la fuite de données."""
    log_values = np.log(s.astype(np.float64).to_numpy())
    log_series = pd.Series(log_values, index=s.index)
    return log_series.diff(period).shift(1).fillna(0).to_numpy(dtype=np.float32)


def safe_log_ret(s: pd.Series) -> np.ndarray:
    """Log-return t-t-1, décalé idem."""
    return lagged_log_return(s, 1)


def compute_rolling_volatility(signal: NDArray, window: int = 14) -> NDArray:
    # S'assurer que signal est de type float

    signal = signal.astype(float)

    vol = np.empty_like(signal)  # vol aura le même dtype que signal, donc float
    vol[:] = np.nan  # maintenant pas de soucis pour assigner NaN

    for i in range(len(signal)):
        start = max(0, i - window + 1)
        window_slice = signal[start : i + 1]
        vol[i] = np.std(window_slice) if len(window_slice) > 1 else 0.0

    return vol


def compute_momentum(signal: NDArray, period: int = 5) -> NDArray:
    # Convertir signal en float pour éviter les problèmes de type

    signal = signal.astype(float)

    # Maintenant mom sera aussi en float
    mom = np.empty_like(signal)

    mom[:] = np.nan

    for i in range(len(signal)):
        if i >= period:
            mom[i] = signal[i] - signal[i - period]

    return mom


# Exemple : RSI « sécurisé »
def rsi_safe(s: pd.Series, n: int = 14) -> np.ndarray:
    d = s.diff()
    g = d.clip(lower=0).rolling(n, closed="left").mean()
    losses = (-d.clip(upper=0)).rolling(n, closed="left").mean()
    rsi = 1 - 1 / (1 + g / (losses + 1e-9))
    return rsi.shift(1).to_numpy(dtype=np.float32)


def _atr(h: pd.Series, low: pd.Series, c: pd.Series, n: int = 14) -> np.ndarray:
    """Compute the Average True Range (ATR) for given high, low, and close prices."""
    tr = pd.concat(
        [h - low, (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1
    ).max(axis=1)
    return np.array(tr.rolling(n).mean())


def _bollinger_width(s: pd.Series, n: int = 20, k: float = 2.0):
    """Compute the Bollinger Bands width for a given series."""

    ma = s.rolling(n).mean()
    std = s.rolling(n).std()
    return (k * std * 2) / (ma + 1e-9)


def compute_time_features(date_index: pd.DatetimeIndex) -> np.ndarray:
    """Compute unique linear time features for a week, normalized to [0, 1].

    Args:
        date_index (pd.DatetimeIndex): The index of dates to compute features for.

    Returns:
        np.ndarray: A 1D array of normalized linear values representing time over a week.
    """

    h = date_index.hour.values
    d = date_index.dayofweek.values
    out = pd.DataFrame(index=date_index)

    out["hour_sin"] = np.sin(2 * np.pi * h / 24)

    out["hour_cos"] = np.cos(2 * np.pi * h / 24)

    out["dow_sin"] = np.sin(2 * np.pi * d / 7)

    out["dow_cos"] = np.cos(2 * np.pi * d / 7)

    return np.array(out)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _macd(s: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the MACD (Moving Average Convergence Divergence) for a given series."""

    macd = _ema(s, 12) - _ema(s, 26)

    return np.array(macd), np.array(_ema(macd, 9))


def _stoch_k(c: pd.Series, h: pd.Series, low: pd.Series, n: int = 14) -> np.ndarray:
    """Compute the Stochastic %K for a given close, high, and low prices."""
    low_n = low.rolling(n).min()

    high_n = h.rolling(n).max()

    return np.array((c - low_n) / (high_n - low_n + 1e-9))


def compute_bollinger_bands(
    signal: NDArray, window: int = 20, num_std_dev: int = 2
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Compute the Bollinger Bands over a moving window.

    Args:
        signal (NDArray): A 1D or 2D NumPy array of price data.
        window (int, optional): The window size for rolling mean and std. Defaults to 20.
        num_std_dev (int, optional): The number of standard deviations for the
            bands. Defaults to 2.

    Returns:
        tuple of NDArray: (upper_band, mid_band, lower_band) each of shape (n_samples,).
    """
    signal_1d = signal.squeeze().astype(float)

    # Initialize arrays
    upper_band = np.empty_like(signal_1d)
    mid_band = np.empty_like(signal_1d)
    lower_band = np.empty_like(signal_1d)

    upper_band[:] = np.nan
    mid_band[:] = np.nan
    lower_band[:] = np.nan

    for i in range(len(signal_1d)):
        start = max(0, i - window + 1)
        window_slice = signal_1d[start : i + 1]

        mean_val = np.mean(window_slice)
        std_val = np.std(window_slice)

        mid_band[i] = mean_val
        upper_band[i] = mean_val + num_std_dev * std_val
        lower_band[i] = mean_val - num_std_dev * std_val

    return (
        np.asarray(upper_band).reshape(-1, 1),
        np.asarray(mid_band).reshape(-1, 1),
        np.asarray(lower_band).reshape(-1, 1),
    )


def compute_stoch_oscillator(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[NDArray, NDArray]:
    """
    Compute the Stochastic Oscillator (%K and %D) based on high, low, and close prices.

    Args:
        high (NDArray): high prices (1D).
        low (NDArray):  low prices (1D).
        close (NDArray): close prices (1D).
        k_period (int, optional): Period for %K. Defaults to 14.
        d_period (int, optional): Smoothing period for %D (SMA of %K). Defaults to 3.

    Returns:
        tuple of NDArray: (stoch_k, stoch_d) each of shape (n_samples,).
    """
    high_1d = high.squeeze().astype(float)
    low_1d = low.squeeze().astype(float)
    close_1d = close.squeeze().astype(float)

    length = len(close_1d)
    stoch_k = np.empty(length)
    stoch_k[:] = np.nan
    stoch_d = np.empty(length)
    stoch_d[:] = np.nan

    # Compute %K
    for i in range(length):
        start = max(0, i - k_period + 1)
        window_high = high_1d[start : i + 1]
        window_low = low_1d[start : i + 1]

        highest_high = np.max(window_high)
        lowest_low = np.min(window_low)

        if highest_high - lowest_low == 0:
            stoch_k[i] = 0.0
        else:
            stoch_k[i] = (close_1d[i] - lowest_low) / (highest_high - lowest_low)

    # Compute %D (simple moving average of %K)
    for i in range(length):
        start = max(0, i - d_period + 1)
        window_k = stoch_k[start : i + 1]
        stoch_d[i] = np.nanmean(window_k)

    return np.asarray(stoch_k).reshape(-1, 1), np.asarray(stoch_d).reshape(-1, 1)


def compute_obv(signal: NDArray, volume: NDArray) -> NDArray:
    """
    Compute On-Balance volume (OBV) given price (e.g., close) and volume arrays.

    Args:
        signal (NDArray): A 1D array of prices (e.g., closing prices).
        volume (NDArray): A 1D array of volumes corresponding to the prices.

    Returns:
        NDArray: A 1D array of OBV values.
    """
    price_1d = signal.squeeze().astype(float)
    vol_1d = volume.squeeze().astype(float)

    obv = np.zeros_like(price_1d)
    for i in range(1, len(price_1d)):
        if price_1d[i] > price_1d[i - 1]:
            obv[i] = obv[i - 1] + vol_1d[i]
        elif price_1d[i] < price_1d[i - 1]:
            obv[i] = obv[i - 1] - vol_1d[i]
        else:
            obv[i] = obv[i - 1]

    return np.asarray(obv).reshape(-1, 1)


def compute_atr(
    high: NDArray, low: NDArray, close: NDArray, window: int = 14
) -> NDArray:
    """
    Compute the Average True Range (ATR) for the given price arrays.

    Args:
        high (NDArray): high prices (1D).
        low (NDArray): low prices (1D).
        close (NDArray): close prices (1D).
        window (int, optional): The window size for the ATR. Defaults to 14.

    Returns:
        NDArray: A 1D array of ATR values.
    """
    high_1d = high.squeeze().astype(float)
    low_1d = low.squeeze().astype(float)
    close_1d = close.squeeze().astype(float)

    length = len(close_1d)
    if length < 2:
        raise ValueError("Not enough data to compute ATR.")

    # True Range array
    tr = np.empty(length)
    tr[:] = np.nan

    # ATR array
    atr = np.empty(length)
    atr[:] = np.nan

    # True Range for the first period is just (high - low)
    tr[0] = high_1d[0] - low_1d[0]
    atr[0] = tr[0]  # Starting point for ATR

    for i in range(1, length):
        # True Range calculation:
        range1 = high_1d[i] - low_1d[i]
        range2 = abs(high_1d[i] - close_1d[i - 1])
        range3 = abs(low_1d[i] - close_1d[i - 1])
        tr[i] = max(range1, range2, range3)

        # ATR calculation (typical EMA approach)
        if i < window:
            # For initial periods, can use simple average or partial EMA
            atr[i] = np.mean(tr[: i + 1])
        else:
            # ATR(i) = (ATR(i-1) * (window-1) + TR(i)) / window
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window

    # Retourne un vecteur 1D (N,) pour compatibilité directe avec pd.Series
    return np.asarray(atr, dtype=np.float64)


def volumetric_features(df: pd.DataFrame, window: int = 24) -> np.ndarray:
    """
    Crée des features basées sur volume / nombre de trades / pression acheteuse.
      • v_rel         : volume horaire / moyenne mobile J-1
      • qv_rel        : quote_vol horaire / moyenne mobile
      • trade_rate    : nb trades normalisé
      • buy_pressure  : part des takers acheteurs
      • imbalance     : (TBQ − (QV−TBQ)) / QV
    """
    out = pd.DataFrame(index=df.index)

    mean_vol = df["volume"].rolling(window).mean()
    mean_qv = df["quote_vol"].rolling(window).mean()

    out["v_rel"] = df["volume"] / (mean_vol + 1e-9)
    out["qv_rel"] = df["quote_vol"] / (mean_qv + 1e-9)
    out["trade_rate"] = df["num_trades"] / (
        df["num_trades"].rolling(window).mean() + 1e-9
    )
    out["buy_pressure"] = df["taker_buy_base"] / (df["volume"] + 1e-9)

    sell_quote = df["quote_vol"] - df["taker_buy_quote"]
    out["imbalance"] = (df["taker_buy_quote"] - sell_quote) / (df["quote_vol"] + 1e-9)

    return np.array(out)


def micro_volatility(df: pd.DataFrame, span: int = 24) -> np.ndarray:
    """
    Volatilité réalisée & mesures de roughness pour HAR ou hétéroscédasticité.
      • RV  : somme des ret² intra-h sur `span` h
      • BPV : bipower variation
      • QIV : quarticity (proxy « turbulence »)
    On suppose que df est en pas de temps 1h.
    """
    # 1) calcul du log-return sûr, reconverti en Series pour rolling
    log_ret_arr = safe_log_ret(df["close"])  # array de longueur N
    ret = pd.Series(log_ret_arr, index=df.index)

    # 2) calculs rolling
    rv = ret.rolling(span).apply(lambda x: np.sum(x**2), raw=True)
    bpv = (
        ret.abs()
        .shift(1)
        .rolling(span)
        .apply(lambda x: np.sum(x[1:] * x[:-1]), raw=True)
    )
    qiv = ret.rolling(span).apply(lambda x: np.sum(x**4) * span / 3, raw=True)

    # 3) on construit un DataFrame pour garder l’index, si besoin
    out = pd.DataFrame({"rv": rv, "bpv": bpv, "qiv": qiv}, index=df.index)

    # 4) on renvoie un np.ndarray de shape (N, 3)
    return out.to_numpy()


def triple_barrier(
    df: pd.DataFrame,  # DataFrame contenant au moins la colonne "close"
    p_up: float = 3.0,  # multiplicateur du seuil haussier (en ATR ou en %)
    p_dn: float = 3.0,  # multiplicateur du seuil baissier (en ATR ou en %)
    max_h: int = 24,  # horizon maximum (en barres) pour évaluer le trade
    use_atr: bool = True,  # seuils adaptatifs avec l'ATR si True
    direction_margin: float = 0.0,  # marge pour départager les hits combinés
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applique la règle « triple barrière » pour labelliser chaque barre.

    Retourne
    --------
    y_cls : np.ndarray[int8]
        +1  si la barrière haute est touchée avant la basse,
        -1  si la barrière basse est touchée avant la haute,
         0  si aucune barrière n'est touchée dans l'horizon `max_h`.
    y_reg : np.ndarray[float32]
        Le mouvement de prix (en %) au moment du hit ; 0 si aucune touche.
    t_hit : np.ndarray[int16]
        Nombre de barres écoulées jusqu’au toucher (ou `max_h` si rien).
    """
    close = df["close"].to_numpy(dtype=np.float32)

    # ────────────────────────── seuils haut / bas ───────────────────────────
    if use_atr:
        if "atr" not in df.columns:
            raise ValueError("`use_atr=True` mais la colonne 'atr' est absente.")
        thr = df["atr"].to_numpy(dtype=np.float32)
        up_thr = thr * p_up
        dn_thr = thr * p_dn
    else:
        up_thr = close * p_up
        dn_thr = close * p_dn

    n = len(close)
    valid_n = n - max_h  # dernières barres ignorées (pas de futur)

    y_cls = np.zeros(valid_n, dtype=np.int8)
    y_reg = np.zeros(valid_n, dtype=np.float32)
    t_hit = np.full(valid_n, max_h, dtype=np.int16)

    sentinel = max_h + 1  # valeur qui signifie « pas de hit »
    margin = max(0.0, float(direction_margin))

    # ──────────────────────── boucle principale ────────────────────────────
    for t0 in range(valid_n):
        c0 = close[t0]
        up_level = c0 + up_thr[t0]
        dn_level = c0 - dn_thr[t0]

        future = close[t0 + 1 : t0 + max_h + 1]
        future_rel = (
            (future - c0) / c0 if future.size else np.empty(0, dtype=np.float32)
        )

        hits_up = np.flatnonzero(future >= up_level)
        hits_dn = np.flatnonzero(future <= dn_level)

        hit_up = hits_up[0] + 1 if hits_up.size else sentinel
        hit_dn = hits_dn[0] + 1 if hits_dn.size else sentinel

        if future_rel.size:
            max_up_val = float(np.max(future_rel))
            max_up_idx = int(np.argmax(future_rel)) + 1
            max_dn_val = float(np.min(future_rel))
            max_dn_idx = int(np.argmin(future_rel)) + 1
        else:
            max_up_val = float("-inf")
            max_dn_val = float("inf")
            max_up_idx = sentinel
            max_dn_idx = sentinel

        up_pct_thr = up_thr[t0] / c0
        dn_pct_thr = dn_thr[t0] / c0
        meets_up_ext = max_up_val >= up_pct_thr
        meets_dn_ext = -max_dn_val >= dn_pct_thr

        prefer_up = meets_up_ext and (
            not meets_dn_ext or (max_up_val - abs(max_dn_val) >= margin)
        )
        prefer_dn = meets_dn_ext and (
            not meets_up_ext or (abs(max_dn_val) - max_up_val >= margin)
        )

        if prefer_up:
            y_cls[t0] = 1
            y_reg[t0] = max_up_val
            t_hit[t0] = max(1, min(max_up_idx, max_h))
            continue
        if prefer_dn:
            y_cls[t0] = -1
            y_reg[t0] = max_dn_val
            t_hit[t0] = max(1, min(max_dn_idx, max_h))
            continue

        if hit_up < hit_dn:
            y_cls[t0] = 1
            y_reg[t0] = (future[hit_up - 1] - c0) / c0
            t_hit[t0] = hit_up
        elif hit_dn < hit_up:
            y_cls[t0] = -1
            y_reg[t0] = (future[hit_dn - 1] - c0) / c0
            t_hit[t0] = hit_dn
        # sinon y_cls reste 0, y_reg 0, t_hit max_h

    return y_cls, y_reg, t_hit


def _clean_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    • supprime les doublons d'index    (keep='first')
    • remet l'index en ordre croissant (sort_index)
    """
    return df[~df.index.duplicated(keep="first")].sort_index()


def prepare_df(  # mêmes arguments qu'avant
    df_ticker: pd.DataFrame,
    df_usdt: pd.DataFrame,
    df_btc: pd.DataFrame,
    df_eth: pd.DataFrame,
    df_sol: pd.DataFrame,
    df_xrp: pd.DataFrame,
    interval: str,
    time_aware_periods: bool = False,
    *,
    max_h: int = 24,
    vol_h: int = 5,
    # nouveaux paramètres de label binaire
    bin_h: int = 24,
    bin_thr_atr: float = 1.0,
    produce_binary: bool = True,
    min_volume_quantile: Optional[float] = None,
    volume_filter_hours: int = 12,
    direction_margin: float = 0.0025,
) -> pd.DataFrame:
    # Helper pour convertir des heures en nombre de barres selon l'intervalle
    def bars_for_hours(hours: int) -> int:
        if not time_aware_periods:
            # comportement historique:
            # les valeurs 1/4/24 signifient des barres, pas des heures
            return hours
        # parse interval (ex: '5m', '15m', '1h', '4h', '1d')
        try:
            unit = interval[-1].lower()
            val = int(interval[:-1])
        except (ValueError, TypeError, IndexError):
            # fallback conservateur
            return hours
        minutes = 0
        if unit == "m":
            minutes = val
        elif unit == "h":
            minutes = val * 60
        elif unit == "d":
            minutes = val * 60 * 24
        elif unit == "w":
            minutes = val * 60 * 24 * 7
        else:
            return hours
        # nb de barres pour X heures = (X heures en minutes) / (durée d'une barre)
        if minutes <= 0:
            return hours
        bars = int(round((hours * 60) / minutes))
        return max(1, bars)

    # Périodes équivalentes 1h/4h/6h/24h exprimées en barres
    p1h = bars_for_hours(1)
    p4h = bars_for_hours(4)
    p6h = bars_for_hours(6)
    p24h = bars_for_hours(24)
    # 0) nettoyage index ----------------------------------------------------
    df_ticker = df_ticker[~df_ticker.index.duplicated(keep="first")].sort_index()

    # skip si trop court
    if len(df_ticker) <= max_h + vol_h:
        logger.warning("Skip (hist. trop court) : %d lignes", len(df_ticker))
        return pd.DataFrame()

    # 1) ATR ---------------------------------------------------------------
    df_ticker["atr"] = (
        pd.Series(
            _atr(df_ticker["high"], df_ticker["low"], df_ticker["close"], n=14),
            index=df_ticker.index,
        )
        .shift(1)
        .astype(np.float32)
    )

    # 2) triple-barrier -----------------------------------------------------
    y_cls, y_reg, tau = triple_barrier(
        df_ticker,
        p_up=bin_thr_atr,
        p_dn=bin_thr_atr,
        max_h=max_h,
        use_atr=True,
        direction_margin=direction_margin,
    )
    df_ticker = df_ticker.iloc[:-max_h].copy()  # retrait horizon futur
    df_ticker["y_cls"], df_ticker["y_reg"], df_ticker["tau"] = y_cls, y_reg, tau

    # 3) cibles futures (volume et binaire up/down) -------------------------
    df_ticker["y_vol"] = df_ticker["volume"].shift(-vol_h)
    if produce_binary:
        future_close = df_ticker["close"].shift(-bin_h)
        ret_h = (future_close - df_ticker["close"]) / (df_ticker["close"] + 1e-9)
        thr_pct = (df_ticker["atr"] * bin_thr_atr) / (df_ticker["close"] + 1e-9)
        y_bin = np.where(ret_h >= thr_pct, 1, np.where(ret_h <= -thr_pct, -1, 0))
        df_ticker["y_bin"] = y_bin.astype(np.int8)
        trim = max(vol_h, bin_h)
        df_ticker = df_ticker.iloc[:-trim]
    else:
        df_ticker = df_ticker.iloc[:-vol_h]

    # 4) références marché --------------------------------------------------
    for ref in (df_btc, df_usdt, df_eth, df_sol, df_xrp):
        ref.drop_duplicates(inplace=True)

    refs = {
        k: _clean_idx(v).reindex(df_ticker.index, method="ffill")
        for k, v in {
            "btc": df_btc,
            "usdt": df_usdt,
            "eth": df_eth,
            "sol": df_sol,
            "xrp": df_xrp,
        }.items()
    }

    # 6) time-features -------------------------------------------
    df_ticker[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]] = compute_time_features(
        pd.DatetimeIndex(df_ticker.index)
    )

    # 7) dérivées & indicateurs de prix --------------------------
    df_ticker["log_ret"] = pd.Series(
        lagged_log_return(df_ticker["close"]), index=df_ticker.index
    )
    df_ticker["log_ret_4h"] = pd.Series(
        lagged_log_return(df_ticker["close"], p4h), index=df_ticker.index
    )
    df_ticker["log_ret_24h"] = pd.Series(
        lagged_log_return(df_ticker["close"], p24h), index=df_ticker.index
    )
    log_close = pd.Series(
        np.log(df_ticker["close"].astype(np.float64).to_numpy()),
        index=df_ticker.index,
    )
    log_diff_1h = log_close.diff()
    df_ticker["volatility_24h"] = (
        log_diff_1h.rolling(p24h).std().shift(1).fillna(0).astype(np.float32)
    )
    df_ticker["volatility_6h"] = (
        log_diff_1h.rolling(max(p6h, 2))
        .std(ddof=0)
        .shift(1)
        .fillna(0)
        .astype(np.float32)
    )
    df_ticker["volatility_1h"] = (
        log_diff_1h.rolling(max(p1h, 2))
        .std(ddof=0)
        .shift(1)
        .fillna(0)
        .astype(np.float32)
    )
    ret_sq = log_diff_1h.pow(2)
    df_ticker["rv_6h"] = (
        ret_sq.rolling(max(p6h, 2)).sum().shift(1).fillna(0).astype(np.float32)
    )
    df_ticker["rv_1h"] = (
        ret_sq.rolling(max(p1h, 1)).sum().shift(1).fillna(0).astype(np.float32)
    )
    df_ticker["price_range"] = (
        ((df_ticker["high"] - df_ticker["low"]) / (df_ticker["close"] + 1e-9))
        .shift(1)
        .fillna(0)
        .astype(np.float32)
    )
    df_ticker["momentum"] = pd.Series(
        compute_momentum(df_ticker["close"].to_numpy(dtype=np.float32)),
        index=df_ticker.index,
    ).fillna(0)
    # RSI et Stoch K: neutraliser les NaN initiaux par un remplissage « neutre »
    df_ticker["rsi"] = (
        pd.Series(rsi_safe(df_ticker["close"]), index=df_ticker.index)
        .fillna(0.5)
        .astype(np.float32)
    )
    df_ticker["bollinger_width"] = (
        _bollinger_width(df_ticker["close"]).shift(1).fillna(0).astype(np.float32)
    )
    macd, macd_sig = _macd(df_ticker["close"])
    df_ticker["macd_diff"] = (
        pd.Series(macd - macd_sig, index=df_ticker.index).shift(1).fillna(0)
    )
    df_ticker["stoch_k"] = (
        pd.Series(
            _stoch_k(df_ticker["close"], df_ticker["high"], df_ticker["low"]),
            index=df_ticker.index,
        )
        .shift(1)
        .fillna(0.5)
        .astype(np.float32)
    )

    # 7a) Volatilités par range (Parkinson, Garman-Klass, Rogers-Satchell) sur 24h
    # Parkinson
    parkinson_inst = (np.log(df_ticker["high"]) - np.log(df_ticker["low"])) ** 2 / (
        4.0 * np.log(2)
    )
    df_ticker["parkinson_vol_24h"] = (
        parkinson_inst.rolling(p24h).mean().shift(1).fillna(0).astype(np.float32)
    )
    # Garman-Klass
    oc = np.log(df_ticker["close"]) - np.log(df_ticker["open"])  # overnight close-open
    hl = np.log(df_ticker["high"]) - np.log(df_ticker["low"])  # high-low
    gk_inst = 0.5 * hl**2 - (2.0 * np.log(2) - 1) * oc**2
    df_ticker["gk_vol_24h"] = (
        gk_inst.rolling(p24h).mean().shift(1).fillna(0).astype(np.float32)
    )
    # Rogers-Satchell
    u = np.log(df_ticker["high"]) - np.log(df_ticker["close"])  # up move
    d = np.log(df_ticker["low"]) - np.log(df_ticker["close"])  # down move
    rs_inst = u * (np.log(df_ticker["high"]) - np.log(df_ticker["open"])) + d * (
        np.log(df_ticker["low"]) - np.log(df_ticker["open"])  # noqa: W503
    )
    df_ticker["rs_vol_24h"] = (
        rs_inst.rolling(p24h).mean().shift(1).fillna(0).astype(np.float32)
    )

    # 7b) Indicateurs additionnels légers
    h, low_, c, v = (
        df_ticker["high"].astype(float),
        df_ticker["low"].astype(float),
        df_ticker["close"].astype(float),
        df_ticker["volume"].astype(float),
    )
    # ADX(14) simplifié
    up = h.diff()
    dn = -low_.diff()
    plus_dm = (up.where((up > dn) & (up > 0), 0.0)).rolling(14).sum()
    minus_dm = (dn.where((dn > up) & (dn > 0), 0.0)).rolling(14).sum()
    tr1 = h - low_
    tr2 = (h - c.shift()).abs()
    tr3 = (low_ - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).sum() + 1e-9
    plus_di = 100 * (plus_dm / tr)
    minus_di = 100 * (minus_dm / tr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    df_ticker["adx_14"] = dx.rolling(14).mean().shift(1).fillna(0).astype(np.float32)
    # MFI(14)
    tp = (h + low_ + c) / 3.0
    rmf = tp * v
    pos = rmf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg = rmf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    mfr = pos / (neg + 1e-9)
    df_ticker["mfi_14"] = (
        (100 - (100 / (1 + mfr))).shift(1).fillna(50).astype(np.float32)
    )
    # CCI(20)
    tp20 = tp.rolling(20).mean()
    md20 = (tp - tp20).abs().rolling(20).mean() + 1e-9
    df_ticker["cci_20"] = (
        ((tp - tp20) / (0.015 * md20)).shift(1).fillna(0).astype(np.float32)
    )
    # Williams %R(14)
    hh = h.rolling(14).max()
    ll = low_.rolling(14).min()
    df_ticker["williamsr_14"] = (
        (-(hh - c) / (hh - ll + 1e-9) * 100.0).shift(1).fillna(0).astype(np.float32)
    )
    # Chaikin Oscillator (3,10)
    adl = ((2 * c - h - low_) / (h - low_ + 1e-9) * v).fillna(0).cumsum()
    df_ticker["chaikin_osc_3_10"] = (
        (adl.ewm(span=3).mean() - adl.ewm(span=10).mean())
        .shift(1)
        .fillna(0)
        .astype(np.float32)
    )
    # OBV z-score 24h
    obv = (np.sign(c.diff().fillna(0)) * v).fillna(0).cumsum()
    obv_ma = obv.rolling(p24h).mean()
    obv_sd = obv.rolling(p24h).std()
    df_ticker["obv_z_24h"] = (
        ((obv - obv_ma) / (obv_sd + 1e-9)).shift(1).fillna(0).astype(np.float32)
    )

    # 7b) Features directionnelles additionnelles -------------------------
    # VWAP approximatif sur 24h et ratio close/VWAP
    pv = (df_ticker["close"] * df_ticker["volume"]).rolling(p24h).sum()
    v = df_ticker["volume"].rolling(p24h).sum()
    vwap_24h = (pv / (v + 1e-9)).shift(1)
    df_ticker["vwap_ratio_24h"] = (df_ticker["close"] / (vwap_24h + 1e-9)).fillna(1.0)

    # Anatomie des bougies
    open_ = df_ticker["open"].astype(float)
    high_ = df_ticker["high"].astype(float)
    low_ = df_ticker["low"].astype(float)
    close_ = df_ticker["close"].astype(float)
    body = (close_ - open_).astype(np.float32)
    rng = (high_ - low_).replace(0, np.nan)
    max_oc = pd.Series(
        np.maximum(open_.to_numpy(), close_.to_numpy()), index=df_ticker.index
    )
    min_oc = pd.Series(
        np.minimum(open_.to_numpy(), close_.to_numpy()), index=df_ticker.index
    )
    upper_wick = (high_ - max_oc).clip(lower=0)
    lower_wick = (min_oc - low_).clip(lower=0)
    df_ticker["candle_body"] = body.shift(1).fillna(0)
    df_ticker["upper_wick"] = upper_wick.shift(1).fillna(0).astype(np.float32)
    df_ticker["lower_wick"] = lower_wick.shift(1).fillna(0).astype(np.float32)
    df_ticker["body_to_range"] = (
        (body.abs() / (rng + 1e-9)).shift(1).fillna(0).astype(np.float32)
    )
    df_ticker["total_wick_to_range"] = (
        ((upper_wick + lower_wick) / (rng + 1e-9)).shift(1).fillna(0).astype(np.float32)
    )

    # Position du close dans la fenêtre 24h (rank normalisé 0..1)
    rolling_min = close_.rolling(p24h).min()
    rolling_max = close_.rolling(p24h).max()
    df_ticker["close_rank_24h"] = (
        ((close_ - rolling_min) / (rolling_max - rolling_min + 1e-9))
        .shift(1)
        .fillna(0)
        .astype(np.float32)
    )

    # Pente (moindre carrés) sur log prix 24h
    def _rolling_slope(arr: np.ndarray, win: int = 24) -> np.ndarray:
        out = np.full_like(arr, np.nan, dtype=np.float64)
        x = np.arange(win, dtype=np.float64)
        x = (x - x.mean()) / (x.std() + 1e-12)
        for i in range(win, len(arr) + 1):
            y = arr[i - win : i]
            y = (y - y.mean()) / (y.std() + 1e-12)
            out[i - 1] = (x * y).mean()
        return out

    df_ticker["trend_slope_log_24h"] = (
        pd.Series(
            _rolling_slope(np.log(close_.to_numpy() + 1e-9), p24h),
            index=df_ticker.index,
        )
        .shift(1)
        .fillna(0)
        .astype(np.float32)
    )

    # Z-score du rendement 1h sur 24h et rendement vs ATR
    ret1 = df_ticker["log_ret"].astype(np.float32)
    mu = ret1.rolling(p24h).mean()
    sd = ret1.rolling(p24h).std()
    df_ticker["ret_zscore_24h"] = (
        ((ret1 - mu) / (sd + 1e-9)).shift(1).fillna(0).astype(np.float32)
    )
    df_ticker["ret_to_atr"] = (
        ((close_.pct_change()) / (df_ticker["atr"] + 1e-9))
        .shift(1)
        .fillna(0)
        .astype(np.float32)
    )

    # 8) volumétriques & micro-vol -------------------------------
    vf = pd.DataFrame(
        volumetric_features(df_ticker, window=p24h).astype(np.float32),
        index=df_ticker.index,
        columns=["v_rel", "qv_rel", "trade_rate", "buy_pressure", "imbalance"],
    ).shift(1)
    mv = pd.DataFrame(
        micro_volatility(df_ticker, span=p24h),
        index=df_ticker.index,
        columns=["rv", "bpv", "qiv"],
    ).shift(1)
    df_ticker = pd.concat([df_ticker, vf, mv], axis=1)
    df_ticker.drop(
        columns=["qv_rel", "trade_rate", "bpv", "qiv"],
        inplace=True,
        errors="ignore",
    )

    taker_buy_base = (
        df_ticker["taker_buy_base"]
        if "taker_buy_base" in df_ticker.columns
        else pd.Series(0.0, index=df_ticker.index)
    )
    volume_series = df_ticker["volume"].astype(np.float32)
    raw_ofi = (2 * taker_buy_base - volume_series) / (volume_series + 1e-9)
    df_ticker["ofi_1h"] = raw_ofi.shift(1).fillna(0).astype(np.float32)
    df_ticker["ofi_6h"] = (
        raw_ofi.rolling(max(p6h, 2)).mean().shift(1).fillna(0).astype(np.float32)
    )
    vol_mean_24h = volume_series.rolling(p24h).mean()
    vol_std_24h = volume_series.rolling(p24h).std(ddof=0)
    df_ticker["volume_zscore_24h"] = (
        ((volume_series - vol_mean_24h) / (vol_std_24h + 1e-9))
        .shift(1)
        .fillna(0)
        .astype(np.float32)
    )
    # Petit ffill sur quelques colonnes de contexte pour éviter NaN initiaux
    for col in [
        "btc_log_ret_1h",
        "btc_log_ret_4h",
        "eth_log_ret_1h",
        "usdt_volume_rel",
        "corr_ret_btc_24h",
        "corr_ret_eth_24h",
    ]:
        if col in df_ticker.columns:
            df_ticker[col] = df_ticker[col].ffill().fillna(0)

    # 9) market-wide context -------------------------------------
    df_ticker["btc_log_ret_1h"] = pd.Series(
        lagged_log_return(refs["btc"]["close"], p1h), index=df_ticker.index
    )
    df_ticker["btc_log_ret_4h"] = pd.Series(
        lagged_log_return(refs["btc"]["close"], p4h), index=df_ticker.index
    )
    df_ticker["eth_log_ret_1h"] = pd.Series(
        lagged_log_return(refs["eth"]["close"], p1h), index=df_ticker.index
    )
    p6h = bars_for_hours(6)
    df_ticker["usdt_volume_rel"] = (
        (refs["usdt"]["volume"] / (refs["usdt"]["volume"].rolling(p6h).mean() + 1e-9))
        .shift(1)
        .fillna(1.0)
        .astype(np.float32)
    )
    # Corrélations roulantes (24h) avec BTC/ETH — alignement d'index obligatoire
    roll = p24h
    ret_t = pd.Series(lagged_log_return(df_ticker["close"]), index=df_ticker.index)
    ret_b = pd.Series(lagged_log_return(refs["btc"]["close"]))
    ret_e = pd.Series(lagged_log_return(refs["eth"]["close"]))
    # Aligner BTC/ETH sur l'index du ticker (UTC 1h)
    ret_b = ret_b.reindex(df_ticker.index)
    ret_e = ret_e.reindex(df_ticker.index)
    df_ticker["corr_ret_btc_24h"] = (
        ret_t.rolling(roll).corr(ret_b).shift(1).fillna(0).astype(np.float32)
    )
    df_ticker["corr_ret_eth_24h"] = (
        ret_t.rolling(roll).corr(ret_e).shift(1).fillna(0).astype(np.float32)
    )
    # Bêta (régression simple) sur ret_t vs BTC/ETH sur 24h

    def _rolling_beta(y: pd.Series, x: pd.Series, win: int) -> pd.Series:
        cov = y.rolling(win).cov(x)
        var = x.rolling(win).var()
        beta = cov / (var + 1e-12)
        return beta

    df_ticker["beta_btc_24h"] = (
        _rolling_beta(ret_t, ret_b, roll).shift(1).fillna(0).astype(np.float32)
    )
    df_ticker["beta_eth_24h"] = (
        _rolling_beta(ret_t, ret_e, roll).shift(1).fillna(0).astype(np.float32)
    )
    # Ratio taker buy
    tbq = (
        df_ticker["taker_buy_quote"]
        if "taker_buy_quote" in df_ticker.columns
        else pd.Series(0.0, index=df_ticker.index)
    )
    qv = (
        df_ticker["quote_vol"]
        if "quote_vol" in df_ticker.columns
        else pd.Series(0.0, index=df_ticker.index)
    )
    df_ticker["taker_buy_quote_ratio"] = (
        (tbq / (qv + 1e-9)).shift(1).fillna(0).astype(np.float32)
    )
    # Moyennes roulantes 24h sur des flux
    if "taker_buy_quote_ratio" in df_ticker.columns:
        df_ticker["tbqr_ma_24h"] = (
            df_ticker["taker_buy_quote_ratio"]
            .rolling(p24h)
            .mean()
            .shift(1)
            .fillna(0)
            .astype(np.float32)
        )
    if "buy_pressure" in df_ticker.columns:
        df_ticker["buy_press_ma_24h"] = (
            df_ticker["buy_pressure"]
            .rolling(p24h)
            .mean()
            .shift(1)
            .fillna(0)
            .astype(np.float32)
        )
    # 10) purge NaN ESSENTIEL (sans propagation futur → passé)
    required_cols = FEATURE_COLUMNS + ["y_cls", "y_reg", "y_vol", "tau"]
    if produce_binary:
        required_cols.append("y_bin")
    missing_cols = [col for col in required_cols if col not in df_ticker.columns]
    if missing_cols:
        raise KeyError(f"Colonnes de features manquantes: {missing_cols}")

    # Filtre volume faible
    if (
        min_volume_quantile is not None
        and 0 < min_volume_quantile < 1
        and "volume" in df_ticker.columns
    ):
        filter_window = max(1, bars_for_hours(volume_filter_hours))
        rolling_quantile = (
            df_ticker["volume"]
            .rolling(filter_window, min_periods=max(1, filter_window // 2))
            .quantile(min_volume_quantile)
            .shift(1)
        )
        fallback = df_ticker["volume"].quantile(min_volume_quantile)
        rolling_quantile = rolling_quantile.fillna(fallback)
        mask = df_ticker["volume"] >= rolling_quantile
        df_ticker = df_ticker[mask]

    # Sélection des colonnes requises
    df_ticker = df_ticker.loc[:, required_cols]

    # Warmup trim pour éliminer les NaN des fenêtres roulantes initiales
    # Max des fenêtres utilisées ≈ 26 (MACD), 24 (beaucoup d'autres) + décalages → ~50
    warmup = max(50, p24h + 26)
    if len(df_ticker) > warmup:
        df_ticker = df_ticker.iloc[warmup:]

    # Dropna strict après warmup, mais d'abord diagnostic sur les NaN restants
    before_rows = len(df_ticker)
    nan_ratio_pre = df_ticker.isna().mean().sort_values(ascending=False)
    top_nan_pre = nan_ratio_pre[nan_ratio_pre > 0].head(10).to_dict()
    df_ticker = df_ticker.dropna(axis=0, how="any")

    if df_ticker.empty:
        logger.warning(
            "Skip (NaN après filtrage) — avant=%d, cols=%d, top=%s",
            before_rows,
            len(required_cols),
            top_nan_pre,
        )
        return pd.DataFrame()

    # Cast final des types
    df_ticker[FEATURE_COLUMNS] = df_ticker[FEATURE_COLUMNS].astype(np.float32)
    df_ticker["y_cls"] = df_ticker["y_cls"].astype(np.int8)
    df_ticker["y_reg"] = df_ticker["y_reg"].astype(np.float32)
    df_ticker["y_vol"] = df_ticker["y_vol"].astype(np.float32)
    df_ticker["tau"] = df_ticker["tau"].astype(np.float32)
    if "y_bin" in df_ticker.columns:
        df_ticker["y_bin"] = df_ticker["y_bin"].astype(np.int8)

    if df_ticker.isna().any().any():
        raise ValueError("NaN détectés après nettoyage des features")

    logger.info(
        "DF prêt : %d lignes, %d colonnes (0 NaN).", len(df_ticker), df_ticker.shape[1]
    )
    return df_ticker


def split_df_by_time(
    df: pd.DataFrame, split_time: pd.Timestamp, buffer: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpe globale sans fuite de temps."""
    train = df.loc[: split_time - pd.Timedelta(hours=buffer)]
    val = df.loc[split_time + pd.Timedelta(hours=buffer) :]
    return train, val


def make_windows(
    df: pd.DataFrame,
    window: int,
    stride: int = 6,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
]:
    """
    Découpe un DataFrame en fenêtres glissantes.
    Retourne des tableaux vides (shape[0] == 0) si le DF est trop court.
    """
    cols_feat = FEATURE_COLUMNS
    missing_features = [col for col in cols_feat if col not in df.columns]
    if missing_features:
        raise KeyError(f"Colonnes absentes dans le DataFrame : {missing_features}")
    n_feat = len(cols_feat)
    N = len(df)

    if N < window:
        # tableaux vides mais typés → éviter ValueError « need at least one array »
        empty_X = np.empty((0, window, n_feat), dtype=np.float32)
        empty_1 = np.empty((0,), dtype=np.float32)
        empty_i = np.empty((0,), dtype=np.int64)
        return empty_X, empty_1, empty_i, empty_1.copy(), empty_1.copy(), None

    Xs, y_regs, y_clss, y_vols, taus = [], [], [], [], []
    y_bins: list[int] | None = [] if "y_bin" in df.columns else None

    for end_idx in range(window - 1, N, stride):
        win = df.iloc[end_idx - window + 1 : end_idx + 1]
        Xs.append(win[cols_feat].values.astype(np.float32))
        y_regs.append(float(df["y_reg"].iat[end_idx]))
        y_clss.append(int(df["y_cls"].iat[end_idx]))
        y_vols.append(float(df["y_vol"].iat[end_idx]))
        taus.append(float(df["tau"].iat[end_idx]))
        if y_bins is not None:
            y_bins.append(int(df["y_bin"].iat[end_idx]))

    X_arr = np.stack(Xs) if Xs else np.empty((0, window, n_feat), dtype=np.float32)
    y_rg = np.array(y_regs, dtype=np.float32)
    y_cl = np.array(y_clss, dtype=np.int64)
    y_vl = np.array(y_vols, dtype=np.float32)
    y_ta = np.array(taus, dtype=np.float32)

    y_ba = np.array(y_bins, dtype=np.int64) if y_bins is not None else None
    return X_arr, y_rg, y_cl, y_vl, y_ta, y_ba


async def _fetch_reference_assets(
    config: DatasetBuildConfig,
) -> Dict[str, pd.DataFrame]:
    symbols = {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
        "xrp": "XRPUSDT",
        "sol": "SOLUSDT",
    }

    coroutines = [
        fetch_ohlc(
            sym=symbol,
            intv=config.interval,
            bar_needed=config.max_candles,
            use_cache=not config.disable_cache,
            cache_ttl_hours=config.cache_ttl_hours,
        )
        for symbol in symbols.values()
    ]

    logger.info(
        "Téléchargement des actifs de référence (%s)", ", ".join(symbols.values())
    )
    results = await asyncio.gather(*coroutines)
    return dict(zip(symbols.keys(), results))


async def create_ticker_dataset(
    window_size: int,
    window_stride: int,
    ticker_name: str,
    interval: str,
    stock_btc: pd.DataFrame,
    stock_eth: pd.DataFrame,
    stock_xrp: pd.DataFrame,
    stock_sol: pd.DataFrame,
    max_value: int,
    sym_base_asset: Dict[str, str],
    *,
    cache_ttl_hours: Optional[int],
    use_cache: bool,
    split_hours: int,
    purge_hours: int,
    min_train_windows: int,
    min_val_windows: int,
    # labels binaires
    bin_h: int,
    bin_thr_atr: float,
    produce_binary: bool,
    time_aware_periods: bool,
    min_volume_quantile: Optional[float],
    volume_filter_hours: int,
    direction_margin: float,
) -> Optional[Tuple[CryptoDataset, CryptoDataset]]:
    try:
        tasks = [
            fetch_ohlc(
                sym=ticker_name,
                intv=interval,
                bar_needed=max_value,
                use_cache=use_cache,
                cache_ttl_hours=cache_ttl_hours,
            ),
            fetch_ohlc(
                sym=f"{sym_base_asset[ticker_name]}USDT",
                intv=interval,
                bar_needed=max_value,
                use_cache=use_cache,
                cache_ttl_hours=cache_ttl_hours,
            ),
        ]

        (
            stock,
            stock_usdt,
        ) = await asyncio.gather(*tasks)

        if stock is None:
            return None

        logger.info(
            "Ticker %s téléchargé (%d lignes, %s → %s)",
            ticker_name,
            len(stock),
            stock.index[0],
            stock.index[-1],
        )
        logger.debug(
            "Ticker %s base USDT: %d lignes",
            ticker_name,
            len(stock_usdt),
        )

        df = prepare_df(
            df_ticker=stock,
            df_usdt=stock_usdt,
            df_btc=stock_btc,
            df_eth=stock_eth,
            df_sol=stock_sol,
            df_xrp=stock_xrp,
            interval=interval,
            time_aware_periods=time_aware_periods,
            bin_h=bin_h,
            bin_thr_atr=bin_thr_atr,
            produce_binary=produce_binary,
            min_volume_quantile=min_volume_quantile,
            volume_filter_hours=volume_filter_hours,
            direction_margin=direction_margin,
        )

        logger.info("Ticker %s après préparation : %d lignes", ticker_name, len(df))

        if len(df) < window_size:
            logger.warning(
                "Ticker %s insuffisant (%d lignes < fenêtre %d)",
                ticker_name,
                len(df),
                window_size,
            )
            return None

        # découpe temporelle basée sur l'historique du ticker lui-même
        split_time = df.index.max() - pd.Timedelta(hours=split_hours)

        df_train, df_val = split_df_by_time(df, split_time, buffer=purge_hours)

        df_train = df_train.reset_index(drop=True)

        df_val = df_val.reset_index(drop=True)

        logger.info(
            "Ticker %s has %d training samples and %d validation samples.",
            ticker_name,
            len(df_train),
            len(df_val),
        )

        X_tr, y_reg_tr, y_cls_tr, y_vol_tr, tau_tr, y_bin_tr = make_windows(
            df_train, window_size, stride=window_stride
        )

        if X_tr.shape[0] < min_train_windows:
            logger.warning(
                "Ticker %s ignoré : fenêtres train %d < seuil %d",
                ticker_name,
                X_tr.shape[0],
                min_train_windows,
            )
            return None

        X_va, y_reg_va, y_cls_va, y_vol_va, tau_va, y_bin_va = make_windows(
            df_val, window_size, stride=window_stride
        )

        if X_va.shape[0] < min_val_windows:
            logger.warning(
                "Ticker %s ignoré : fenêtres validation %d < seuil %d",
                ticker_name,
                X_va.shape[0],
                min_val_windows,
            )
            return None

        logger.info(
            "Ticker %s processed: Train shape: %s, Val shape: %s",
            ticker_name,
            X_tr.shape,
            X_va.shape,
        )

        hist = torch.bincount(
            torch.from_numpy(y_cls_tr).long() + 1,
            minlength=3,
        ).cpu()
        hist_val = torch.bincount(
            torch.from_numpy(y_cls_va).long() + 1,
            minlength=3,
        ).cpu()
        train_weights = _compute_class_weights(hist.tolist())
        val_weights = _compute_class_weights(hist_val.tolist())
        logger.info(
            "%s : balance train=%s, val=%s",
            ticker_name,
            hist.tolist(),
            hist_val.tolist(),
        )

        return (
            CryptoDataset(
                X=X_tr,
                y_cls=y_cls_tr,
                y_reg=y_reg_tr,
                y_vol=y_vol_tr,
                tau=tau_tr,
                class_weights=train_weights,
                y_bin=y_bin_tr,
            ),
            CryptoDataset(
                X=X_va,
                y_cls=y_cls_va,
                y_reg=y_reg_va,
                y_vol=y_vol_va,
                tau=tau_va,
                class_weights=val_weights,
                y_bin=y_bin_va,
            ),
        )

    except (httpx.HTTPError, RuntimeError, KeyError, ValueError) as e:
        logger.error(
            "Error processing ticker %s, Interval : %s: %s",
            ticker_name,
            interval,
            e,
        )

        return None


async def create_dataset(
    config: DatasetBuildConfig,
    tickers_name: List[str],
    sym_base_asset: Dict[str, str],
    references: Dict[str, pd.DataFrame],
) -> Tuple[Optional[CryptoDataset], Optional[CryptoDataset]]:
    """Construit les datasets train/val pour une liste de tickers."""

    tasks: List[Coroutine[Any, Any, Optional[Tuple[CryptoDataset, CryptoDataset]]]] = []

    for ticker_name in tickers_name:
        logger.info("Traitement du ticker %s", ticker_name)
        tasks.append(
            create_ticker_dataset(
                window_size=config.window_size,
                window_stride=config.window_stride,
                ticker_name=ticker_name,
                interval=config.interval,
                stock_btc=references["btc"],
                stock_eth=references["eth"],
                stock_xrp=references["xrp"],
                stock_sol=references["sol"],
                max_value=config.max_candles,
                sym_base_asset=sym_base_asset,
                cache_ttl_hours=config.cache_ttl_hours,
                use_cache=not config.disable_cache,
                split_hours=config.split_hours,
                purge_hours=config.purge_hours,
                min_train_windows=config.min_train_windows,
                min_val_windows=config.min_val_windows,
                bin_h=config.bin_label_h,
                bin_thr_atr=config.bin_label_thr_atr,
                produce_binary=config.produce_binary_cls,
                time_aware_periods=config.time_aware_periods,
                min_volume_quantile=config.min_volume_quantile,
                volume_filter_hours=config.volume_filter_hours,
                direction_margin=config.direction_margin,
            )
        )

    full_train_dataset: Optional[CryptoDataset] = None
    full_val_dataset: Optional[CryptoDataset] = None

    results = await asyncio.gather(*tasks)

    for result in results:
        if result is None:
            continue

        train_dataset, val_dataset = result

        if full_train_dataset is None:
            full_train_dataset = train_dataset
        else:
            full_train_dataset.X = torch.cat(
                (full_train_dataset.X, train_dataset.X), dim=0
            )
            full_train_dataset.y_cls = torch.cat(
                (full_train_dataset.y_cls, train_dataset.y_cls), dim=0
            )
            full_train_dataset.y_vol = torch.cat(
                (full_train_dataset.y_vol, train_dataset.y_vol), dim=0
            )
            full_train_dataset.y_reg = torch.cat(
                (full_train_dataset.y_reg, train_dataset.y_reg), dim=0
            )
            full_train_dataset.tau = torch.cat(
                (full_train_dataset.tau, train_dataset.tau), dim=0
            )
            t_yb = getattr(train_dataset, "y_bin", None)
            if isinstance(t_yb, torch.Tensor):
                if getattr(full_train_dataset, "y_bin", None) is None:
                    full_train_dataset.y_bin = t_yb  # type: ignore[assignment]
                else:
                    fb = cast(torch.Tensor, full_train_dataset.y_bin)
                    full_train_dataset.y_bin = torch.cat(  # type: ignore[assignment]
                        (fb, t_yb), dim=0
                    )

        if full_val_dataset is None:
            full_val_dataset = val_dataset
        else:
            full_val_dataset.X = torch.cat((full_val_dataset.X, val_dataset.X), dim=0)
            full_val_dataset.y_cls = torch.cat(
                (full_val_dataset.y_cls, val_dataset.y_cls), dim=0
            )
            full_val_dataset.y_vol = torch.cat(
                (full_val_dataset.y_vol, val_dataset.y_vol), dim=0
            )
            full_val_dataset.y_reg = torch.cat(
                (full_val_dataset.y_reg, val_dataset.y_reg), dim=0
            )
            full_val_dataset.tau = torch.cat(
                (full_val_dataset.tau, val_dataset.tau), dim=0
            )
            v_yb = getattr(val_dataset, "y_bin", None)
            if isinstance(v_yb, torch.Tensor):
                if getattr(full_val_dataset, "y_bin", None) is None:
                    full_val_dataset.y_bin = v_yb  # type: ignore[assignment]
                else:
                    fb = cast(torch.Tensor, full_val_dataset.y_bin)
                    full_val_dataset.y_bin = torch.cat(  # type: ignore[assignment]
                        (fb, v_yb), dim=0
                    )

    return full_train_dataset, full_val_dataset


def _subsample_dataset(
    ds: "CryptoDataset", max_n: int, seed: int = 42
) -> "CryptoDataset":
    """Retourne un dataset sous-échantillonné à max_n fenêtres (sans remplacement)."""
    n = len(ds)
    if n <= max_n:
        return ds
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_n, replace=False))
    # Applique l'indice sur chaque tenseur
    ds.X = ds.X[idx]
    ds.y_cls = ds.y_cls[idx]
    ds.y_vol = ds.y_vol[idx]
    ds.y_reg = ds.y_reg[idx]
    ds.tau = ds.tau[idx]
    if isinstance(getattr(ds, "y_bin", None), torch.Tensor):
        ds.y_bin = ds.y_bin[idx]  # type: ignore[assignment]
    return ds


def save_new_batch(
    path: Path | str, batch: CryptoDataset, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, int]:
    """Concatène et persiste un lot de données sur disque.

    Pour éviter l'OverflowError du pickle (> 4 GiB), cette fonction segmente
    automatiquement le lot en "shards" si nécessaire. Les shards sont écrits
    sous forme de fichiers suffixés par _part{idx}.pickle.
    """

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "X": batch.X.detach().cpu().numpy(),
        "y_cls": batch.y_cls.detach().cpu().numpy(),
        "y_vol": batch.y_vol.detach().cpu().numpy(),
        "y_reg": batch.y_reg.detach().cpu().numpy(),
        "tau": batch.tau.detach().cpu().numpy(),
        "meta": metadata.copy() if metadata else {},
    }
    # optionnel: label binaire
    if isinstance(getattr(batch, "y_bin", None), torch.Tensor):
        payload["y_bin"] = batch.y_bin.numpy()  # type: ignore[union-attr]

    # Si un fichier unique existe déjà, on tente la concaténation,
    # sinon on repart de payload
    if target_path.exists() and target_path.stat().st_size > 0:
        try:
            with target_path.open("rb") as fh:
                persisted = pickle.load(fh)
            for key in payload:
                if key == "meta":
                    continue
                persisted[key] = np.concatenate([persisted[key], payload[key]], axis=0)
        except (OSError, pickle.UnpicklingError, EOFError, ValueError) as exc:
            logger.warning("Réécriture du lot %s suite à erreur: %s", target_path, exc)
            persisted = payload
    else:
        persisted = payload

    # Prépare meta (histogrammes/poids/features)
    cls_array = persisted["y_cls"].astype(np.int64)
    class_hist = np.bincount(cls_array + 1, minlength=3)
    class_weights = _compute_class_weights(class_hist.tolist())
    meta = persisted.get("meta", {})
    meta.update(
        {
            "class_hist": class_hist.tolist(),
            "class_weights": class_weights,
            "feature_names": FEATURE_COLUMNS,
        }
    )
    if metadata:
        meta.update(metadata)
    persisted["meta"] = meta

    # Détermine la taille de shard pour maintenir un fichier << 4 GiB
    X = persisted["X"]
    n, T, F = int(X.shape[0]), int(X.shape[1]), int(X.shape[2])
    itemsize = int(X.dtype.itemsize) if hasattr(X, "dtype") else 4
    # budget ≈ 1 GiB pour X par shard (les autres arrays sont négligeables)
    max_bytes = 1_000_000_000
    # Permet d'overrider la taille auto par un paramètre global
    from inspect import currentframe

    frame = currentframe()
    # Recherche du paramètre max_rows_per_shard dans la pile d'appels
    max_rows_per_shard = None
    while frame:
        local_vars = frame.f_locals
        if "self" in local_vars and hasattr(local_vars["self"], "max_rows_per_shard"):
            max_rows_per_shard = getattr(local_vars["self"], "max_rows_per_shard")
            break
        frame = frame.f_back
    if max_rows_per_shard is not None and max_rows_per_shard > 0:
        rows_per_shard = int(max_rows_per_shard)
    else:
        rows_per_shard = max(1, max_bytes // max(1, (T * F * itemsize)))

    def _write_single_file(pth: Path, obj: Dict[str, Any]) -> None:
        with pth.open("wb") as fh:
            pickle.dump(obj, fh)

    if n <= rows_per_shard:
        # Ecriture mono-fichier (remplace le 0B s'il existait)
        try:
            _write_single_file(target_path, persisted)
            lengths = {
                key: len(value)
                for key, value in persisted.items()
                if hasattr(value, "__len__") and key != "meta"
            }
            logger.info("Lot sauvegardé %s → %s", target_path.name, lengths)
            return lengths
        except OverflowError:
            # Fallback exceptionnel: basculer en shards si pickle échoue
            logger.warning(
                "OverflowError sur %s, bascule en sauvegarde par shards", target_path
            )
            # on supprime le fichier vide/corrompu éventuel
            try:
                if target_path.exists():
                    target_path.unlink()
            except OSError:
                pass
            # on continue en mode shards

    # Ecriture par shards
    # Nettoie le fichier cible unique s'il existe (pour éviter confusion)
    try:
        if target_path.exists():
            target_path.unlink()
    except OSError:
        pass

    base = target_path.stem  # ex: ticker_dataset_train_0
    suffix = target_path.suffix  # .pickle
    dirp = target_path.parent

    shard_idx = 0
    start = 0
    while start < n:
        end = min(n, start + rows_per_shard)
        shard = {"meta": meta}
        for key, arr in persisted.items():
            if key == "meta":
                continue
            shard[key] = arr[start:end]
        shard_name = f"{base}_part{shard_idx}{suffix}"
        shard_path = dirp / shard_name
        _write_single_file(shard_path, shard)
        logger.info(
            "Shard sauvegardé %s (%d → %d)", shard_path.name, int(start), int(end)
        )

        # Si le processus est lancé en mode "streaming" (parent uploadant les fichiers),
        # ne PAS supprimer le shard local : le parent (create_full_dataset.py) va
        # détecter, uploader et supprimer le fichier. Sinon, conserver le comportement
        # précédent et supprimer le shard immédiatement.
        streaming_child = os.environ.get("STREAMING_UPLOAD_CHILD", "0") == "1"
        if streaming_child:
            logger.info(
                "Streaming mode enfant détecté — conservation du shard pour l'uploader: %s",
                shard_path,
            )
        else:
            try:
                os.remove(shard_path)
                logger.info("Shard local supprimé: %s", shard_path)
            except Exception as e:
                logger.warning(
                    "Impossible de supprimer le shard local %s: %s", shard_path, e
                )
        shard_idx += 1
        start = end

    lengths = {k: int(n) for k in persisted.keys() if k != "meta"}
    logger.info(
        "Lot shardé %s en %d fichiers (rows/shard≈%d)",
        target_path.name,
        shard_idx,
        int(rows_per_shard),
    )
    return lengths


def _class_histogram(target: torch.Tensor) -> list[int]:
    return torch.bincount(target.long() + 1, minlength=3).cpu().tolist()


def _hist_to_dict(hist: list[int]) -> Dict[str, int]:
    classes = [-1, 0, 1]
    return {str(cls): int(hist[idx]) for idx, cls in enumerate(classes)}


def _compute_class_weights(hist: list[int]) -> list[float]:
    total = float(sum(hist))
    if total == 0:
        return [0.0 for _ in hist]
    raw_weights = [total / count if count > 0 else 0.0 for count in hist]
    weight_sum = sum(raw_weights)
    if weight_sum == 0:
        return [0.0 for _ in hist]
    return [weight / weight_sum for weight in raw_weights]


def _config_report(config: DatasetBuildConfig) -> Dict[str, Any]:
    cfg = asdict(config)
    cfg["output_dir"] = str(cfg["output_dir"])
    cfg["quality_report_path"] = str(cfg["quality_report_path"])
    return cfg


async def main(config: DatasetBuildConfig) -> None:
    """Exécute la génération de datasets par lots."""

    try:
        crypto, sym_base_asset = await get_tickers()
        if not crypto:
            logger.error("Aucun ticker Binance disponible")
            return

        references = await _fetch_reference_assets(config)

        rng = random.Random(config.shuffle_seed)
        crypto_list = list(crypto)
        rng.shuffle(crypto_list)

        # Filtre optionnel par allow-list
        if config.tickers_allow:
            allow = {
                t.strip().upper()
                for t in str(config.tickers_allow).split(",")
                if t.strip()
            }
            before = len(crypto_list)
            crypto_list = [t for t in crypto_list if t.upper() in allow]
            logger.info(
                "Filtre allow-list: %d → %d tickers (%s)",
                before,
                len(crypto_list),
                ", ".join(sorted(allow)),
            )
            if not crypto_list:
                logger.error("Aucun ticker ne correspond à l'allow-list fournie")
                return

        start_batch = config.next_batch_index()
        start_offset = start_batch * config.tickers_per_batch

        if start_offset >= len(crypto_list):
            logger.warning(
                "Indice de départ %d hors limite (tickers=%d)",
                start_offset,
                len(crypto_list),
            )
            return

        processed_batches = 0
        feature_thresholds = _load_feature_thresholds(config.feature_thresholds_path)
        train_metrics = DatasetQualityMetrics(
            thresholds=feature_thresholds,
            alert_tolerance=config.alert_tolerance,
        )
        val_metrics = DatasetQualityMetrics(
            thresholds=feature_thresholds,
            alert_tolerance=config.alert_tolerance,
        )
        batch_summaries: list[Dict[str, Any]] = []
        for batch_no, start_idx in enumerate(
            range(start_offset, len(crypto_list), config.tickers_per_batch)
        ):
            if config.max_batches is not None and batch_no >= config.max_batches:
                logger.info("Stop: limite de lots atteinte (%d)", config.max_batches)
                break

            batch_tickers = crypto_list[
                start_idx : start_idx + config.tickers_per_batch
            ]
            if not batch_tickers:
                break

            current_batch = start_batch + batch_no
            logger.info(
                "Lot %d → %d tickers: %s",
                current_batch,
                len(batch_tickers),
                ", ".join(batch_tickers),
            )

            train_ds, val_ds = await create_dataset(
                config=config,
                tickers_name=batch_tickers,
                sym_base_asset=sym_base_asset,
                references=references,
            )

            if train_ds is None or val_ds is None:
                logger.warning("Lot %d ignoré (dataset vide)", current_batch)
                continue

            logger.info(
                "Lot %d : train=%d, val=%d",
                current_batch,
                len(train_ds),
                len(val_ds),
            )

            # Caps de taille facultatifs pour réduire l'empreinte disque/mémoire
            if (
                config.max_train_samples is not None
                and len(train_ds) > config.max_train_samples
            ):
                train_ds = _subsample_dataset(
                    train_ds,
                    config.max_train_samples,
                    seed=config.shuffle_seed,
                )
                logger.info(
                    "Lot %d : train sous-échantillonné à %d fenêtres",
                    current_batch,
                    int(len(train_ds)),
                )
            if (
                config.max_val_samples is not None
                and len(val_ds) > config.max_val_samples
            ):
                val_ds = _subsample_dataset(
                    val_ds,
                    config.max_val_samples,
                    seed=config.shuffle_seed + 1,
                )
                logger.info(
                    "Lot %d : val sous-échantillonné à %d fenêtres",
                    current_batch,
                    int(len(val_ds)),
                )

            train_hist = _class_histogram(train_ds.y_cls)
            val_hist = _class_histogram(val_ds.y_cls)
            logger.info(
                "Lot %d distribution train=%s val=%s",
                current_batch,
                train_hist,
                val_hist,
            )

            save_new_batch(config.dataset_path("train", current_batch), train_ds)
            save_new_batch(config.dataset_path("val", current_batch), val_ds)
            processed_batches += 1
            train_metrics.update(train_ds)
            val_metrics.update(val_ds)
            batch_summaries.append(
                {
                    "batch": current_batch,
                    "tickers": batch_tickers,
                    "train_count": len(train_ds),
                    "val_count": len(val_ds),
                    "train_hist": _hist_to_dict(train_hist),
                    "val_hist": _hist_to_dict(val_hist),
                }
            )

        logger.info("Génération terminée (%d lots produits)", processed_batches)

        report_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "processed_batches": processed_batches,
            "config": _config_report(config),
            "train": train_metrics.to_dict(),
            "val": val_metrics.to_dict(),
            "batches": batch_summaries,
        }

        try:
            report_path = config.quality_report_path
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as fh:
                json.dump(report_payload, fh, indent=2, ensure_ascii=False)
            logger.info("Rapport qualité écrit dans %s", report_path)
        except OSError as exc:  # pragma: no cover - diagnostic
            logger.warning("Échec écriture rapport qualité: %s", exc)

    finally:
        await _close_client()


if __name__ == "__main__":
    try:
        asyncio.run(main(DatasetBuildConfig.from_cli()))
    except KeyboardInterrupt:
        logger.warning("Interruption par l'utilisateur")
