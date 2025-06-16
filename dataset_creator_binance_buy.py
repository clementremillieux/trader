"""logger_config.py"""

from __future__ import annotations

import asyncio
import os

import pickle
import random

import httpx

from pathlib import Path

import certifi

from typing import Optional, List, Dict, Any, Union, Coroutine, Tuple

import pandas as pd

import numpy as np

from numpy.typing import NDArray

import torch

from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler


from config.logger_config import logger


os.environ["CURL_CA_BUNDLE"] = certifi.where()


API_KEY = "RRx439X2aBvHzrodPRhgNPAw9hyr48lYFqenjNIjWql25a9kuMMcdV7dRnjE9YsU"

API_SECRET = "RKSfisReVgtRa1bMqrzJjAaVhZ6OjAW9ATdLK3XC9gcBkYuCmOvC9ms77od4OoVp"

BASE_URL = "https://api.binance.com"  # HTTPS comme demandé

MAX_CONC = 4

CACHE_DIR = Path(".cache_binance")

CACHE_DIR.mkdir(exist_ok=True)

CLIENT = httpx.AsyncClient(
    base_url=BASE_URL,
    timeout=httpx.Timeout(30.0),
    limits=httpx.Limits(max_connections=MAX_CONC),
)


# ═════════════════ 2 · HTTP helper ═══════════════════════════════
async def _json(path: str, params: Dict[str, Any]) -> Any:
    params = {k: v for k, v in params.items() if v not in (None, "")}

    logger.debug("GET %s %s", path, params)

    r = await CLIENT.get(path, params=params)

    r.raise_for_status()

    return r.json()


def _cache(sym: str, intv: str) -> Path:
    return CACHE_DIR / f"{sym}_{intv}.parquet"


async def get_tickers() -> Tuple[List[str], Dict[str, str]]:
    """Retourne les tickers Binance qui ne sont pas en USDT et qui ont un équivalent en USDT."""

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


async def fetch_ohlc(sym: str, intv: str, bar_needed: int) -> pd.DataFrame:
    """Retourne au moins *BARS_NEEDED* bougies pour (sym,intv)."""

    # p = _cache(sym, intv)

    # if p.exists():
    #     return pd.read_parquet(p)

    logger.info(f"↓ {sym:<10} {intv}")

    rows, start = [], None

    while len(rows) < bar_needed:
        batch = await _json(
            "/api/v3/klines",
            {"symbol": sym, "interval": intv, "limit": 1000, "startTime": start},
        )

        if not batch:
            break
        rows[:0] = batch

        if len(batch) < 1000:
            break

        start = batch[0][0] - 1

    if not rows:
        raise RuntimeError(f"Empty data for {sym} {intv}")

    df = (
        pd.DataFrame(rows)[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        .set_axis(
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "ts",
                "quote_vol",
                "num_trades",
                "taker_buy_base",
                "taker_buy_quote",
            ],
            axis=1,
        )
        .astype(np.float32)
    )

    df["ts"] = pd.to_datetime(df.ts, unit="ms", utc=True)

    df = df.set_index("ts").sort_index().tail(bar_needed)

    # df.to_parquet(p)

    logger.debug(f"{sym} {intv} → {len(df)} rows (cached)")

    return df


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


def safe_diff(arr: pd.Series) -> np.ndarray:
    """Renvoie np.diff(arr) mais padde le premier élément pour conserver la longueur."""
    a = arr.to_numpy()
    diff = np.diff(a, prepend=a[0])
    return diff


def safe_log_ret(arr: pd.Series) -> np.ndarray:
    """Rend la variation logarithmique tout en conservant la longueur (pad au début)."""
    a = arr.to_numpy()
    logp = np.log(a)
    d = np.diff(logp, prepend=logp[0])
    return d


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


def _rsi(s: pd.Series, n: int = 14) -> np.ndarray:
    """Compute the Relative Strength Index (RSI) for a given series."""

    d = s.diff()

    g = d.clip(lower=0).rolling(n).mean()

    l = (-d.clip(upper=0)).rolling(n).mean()

    return np.array(1 - 1 / (1 + g / (l + 1e-9)))


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> np.ndarray:
    """Compute the Average True Range (ATR) for given high, low, and close prices."""
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(
        axis=1
    )
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


def _stoch_k(c: pd.Series, h: pd.Series, l: pd.Series, n: int = 14) -> np.ndarray:
    """Compute the Stochastic %K for a given close, high, and low prices."""
    low_n = l.rolling(n).min()

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
        num_std_dev (int, optional): The number of standard deviations for the bands. Defaults to 2.

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

    return np.asarray(atr).reshape(-1, 1)


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


def balance_binary(
    X: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    y_vol: np.ndarray,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sous-échantillonne X et y pour que chaque classe (0 et 1) soit présente exactement
    min(count(0), count(1)) fois.

    Args:
        X: array de forme (N, ...) contenant vos features.
        y: array de shape (N,) contenant les labels {0,1}.
        seed: graine pour reproductibilité du tirage aléatoire.

    Returns:
        X_bal, y_bal: arrays sous-échantillonnés et mélangés.
    """
    if seed is not None:
        np.random.seed(seed)

    y_int = y_cls

    idx0 = np.where(y_int == 0)[0]
    idx1 = np.where(y_int == 1)[0]
    idx2 = np.where(y_int == -1)[0]

    logger.info(
        f"Balance classes: {len(idx0)} samples for class 0, {len(idx1)} samples for class 1, {len(idx2)} samples for class -1."
    )

    n = min(len(idx0), len(idx1), len(idx2))

    if n == 0:
        return X[:0], y_int[:0], y_reg[:0], y_vol[:0]

    sel = np.concatenate(
        [
            np.random.choice(idx0, n, replace=False),
            np.random.choice(idx1, n, replace=False),
            np.random.choice(idx2, n, replace=False),
        ]
    )

    np.random.shuffle(sel)

    return (
        X[sel],
        y_int[sel],
        y_reg[sel],
        y_vol[sel],
    )


def prepare_df(
    df_ticker: pd.DataFrame,
    df_usdt: pd.DataFrame,
    df_btc: pd.DataFrame,
    df_eth: pd.DataFrame,
    df_sol: pd.DataFrame,
    df_xrp: pd.DataFrame,
    horizon: int,
    gain: float,
) -> pd.DataFrame:
    df_ticker = df_ticker.copy()

    df_ticker["atr"] = _atr(
        df_ticker["high"], df_ticker["low"], df_ticker["close"], n=14
    )

    df_ticker["y_reg"] = (
        df_ticker["close"].shift(-horizon) - df_ticker["close"]
    ) / df_ticker["close"]

    df_ticker["y_vol"] = df_ticker["volume"]

    df_ticker["y_cls"] = 0

    df_ticker.loc[df_ticker["y_reg"] > gain, "y_cls"] = 1

    df_ticker.loc[df_ticker["y_reg"] < -gain, "y_cls"] = -1

    df_ticker["y_cls"] = df_ticker["y_cls"].astype("int8")

    df_ticker = df_ticker.iloc[:-horizon].copy()

    cleans = []
    for other in (df_btc, df_usdt, df_eth, df_sol, df_xrp):
        # déduplication stricte de l’index
        other_clean = other[~other.index.duplicated(keep="first")]
        cleans.append(other_clean)
    df_btc, df_usdt, df_eth, df_sol, df_xrp = cleans

    # 2) On réaligne par forward-fill
    df_btc = df_btc.reindex(df_ticker.index, method="ffill")
    df_usdt = df_usdt.reindex(df_ticker.index, method="ffill")
    df_eth = df_eth.reindex(df_ticker.index, method="ffill")
    df_sol = df_sol.reindex(df_ticker.index, method="ffill")
    df_xrp = df_xrp.reindex(df_ticker.index, method="ffill")

    time_features = compute_time_features(df_ticker.index)

    for i, name in enumerate(["hour_sin", "hour_cos", "dow_sin", "dow_cos"]):
        df_ticker[name] = time_features[:, i]

    df_ticker["close_deriv"] = safe_diff(df_ticker["close"])

    df_ticker["open_deriv"] = safe_diff(df_ticker["open"])

    df_ticker["high_deriv"] = safe_diff(df_ticker["high"])

    df_ticker["low_deriv"] = safe_diff(df_ticker["low"])

    df_ticker["volume_deriv"] = safe_diff(df_ticker["volume"])

    df_ticker["momentum"] = compute_momentum(
        df_ticker["close"].values.reshape(-1, 1),
    )

    df_ticker["rsi"] = _rsi(df_ticker["close"])

    df_ticker["log_ret"] = safe_log_ret(df_ticker["close"])

    df_ticker["bollinger_width"] = _bollinger_width(
        df_ticker["close"],
    )
    df_ticker["macd"], df_ticker["macd_signal"] = _macd(
        df_ticker["close"],
    )

    df_ticker["macd_diff"] = df_ticker["macd"] - df_ticker["macd_signal"]

    df_ticker["stoch_k"] = _stoch_k(
        df_ticker["close"],
        df_ticker["high"],
        df_ticker["low"],
    )

    df_ticker["log_ret_btc"] = safe_log_ret(df_ticker["close"]) - safe_log_ret(
        df_btc["close"]
    )

    vf = volumetric_features(df_ticker)
    for i, name in enumerate(
        ["v_rel", "qv_rel", "trade_rate", "buy_pressure", "imbalance"]
    ):
        df_ticker[name] = vf[:, i]

    mv = micro_volatility(df_ticker)

    cols = ["rv", "bpv", "qiv"]

    for i, col in enumerate(cols):
        df_ticker[col] = mv[:, i]

    df_ticker["btc_close"] = df_btc["close"].values

    df_ticker["btc_volume"] = df_btc["volume"].values

    df_ticker["btc_close_deriv"] = safe_diff(df_btc["close"])

    df_ticker["btc_quote_vol"] = df_btc["quote_vol"].values

    df_ticker["btc_num_trades"] = df_btc["num_trades"].values

    df_ticker["btc_taker_buy_base"] = df_btc["taker_buy_base"].values

    df_ticker["btc_taker_buy_quote"] = df_btc["taker_buy_quote"].values

    df_ticker["eth_close"] = df_eth["close"].values

    df_ticker["eth_volume"] = df_eth["volume"].values

    df_ticker["xrp_close"] = df_xrp["close"].values

    df_ticker["xrp_volume"] = df_xrp["volume"].values

    df_ticker["sol_close"] = df_sol["close"].values

    df_ticker["sol_volume"] = df_sol["volume"].values

    df_ticker["usdt_close"] = df_usdt["close"].values

    df_ticker["usdt_volume"] = df_usdt["volume"].values

    df_ticker["usdt_quote_vol"] = df_usdt["quote_vol"].values

    df_ticker["usdt_num_trades"] = df_usdt["num_trades"].values

    df_ticker["usdt_taker_buy_base"] = df_usdt["taker_buy_base"].values

    df_ticker["usdt_taker_buy_quote"] = df_usdt["taker_buy_quote"].values

    df_ticker = df_ticker.dropna()

    return df_ticker.reset_index(drop=True)


def make_windows(df: pd.DataFrame, window: int, horizon: int):
    """
    Crée des fenêtres glissantes de taille *window* avec horizon de prédiction *horizon*.
    Retourne les features X, et les cibles y_reg, y_cls, y_ret, y_vol.
    """

    cols_feat = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "v_rel",
        "trade_rate",
        "buy_pressure",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_vol",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "close_deriv",
        "open_deriv",
        "high_deriv",
        "low_deriv",
        "volume_deriv",
        "momentum",
        "rsi",
        "log_ret",
        "atr",
        "bollinger_width",
        "macd",
        "macd_signal",
        "macd_diff",
        "stoch_k",
        "log_ret_btc",
        "btc_close",
        "btc_volume",
        "btc_close_deriv",
        "btc_quote_vol",
        "btc_num_trades",
        "btc_taker_buy_base",
        "btc_taker_buy_quote",
        "eth_close",
        "eth_volume",
        "xrp_close",
        "xrp_volume",
        "sol_close",
        "sol_volume",
        "usdt_close",
        "usdt_volume",
        "usdt_quote_vol",
        "usdt_num_trades",
        "usdt_taker_buy_base",
        "usdt_taker_buy_quote",
        "qv_rel",
        "imbalance",
    ]

    Xs, y_regs, y_clss, y_vols = [], [], [], []

    N = len(df)

    for end_idx in range(window - 1, N - horizon):
        win = df.iloc[end_idx - window + 1 : end_idx + 1]

        Xs.append(win[cols_feat].values.astype(np.float32))

        y_regs.append(df["y_reg"].iat[end_idx])

        y_clss.append(df["y_cls"].iat[end_idx])

        y_vols.append(df["y_vol"].iat[end_idx])

    return (
        np.stack(Xs),
        np.array(y_regs, dtype=np.float32),
        np.array(y_clss, dtype=np.int64),
        np.array(y_vols, dtype=np.float32),
    )


async def create_ticker_dataset(
    window_size: int,
    ticker_name: str,
    interval: str,
    n: int,
    stock_btc: pd.DataFrame,
    stock_eth: pd.DataFrame,
    stock_xrp: pd.DataFrame,
    stock_sol: pd.DataFrame,
    max_value: int,
    sym_base_asset: Dict[str, str],
    gain: float,
) -> Optional[Tuple[CryptoDataset, CryptoDataset]]:
    try:
        tasks = [
            fetch_ohlc(sym=ticker_name, intv=interval, bar_needed=max_value),
            fetch_ohlc(
                sym=f"{sym_base_asset[ticker_name]}USDT",
                intv=interval,
                bar_needed=max_value,
            ),
        ]

        (
            stock,
            stock_usdt,
        ) = await asyncio.gather(*tasks)

        if stock is None:
            return None

        assert (
            stock_usdt.shape[0] == stock.shape[0]
            and stock_btc.shape[0] == stock.shape[0]
            and stock_eth.shape[0] == stock.shape[0]
            and stock_sol.shape[0] == stock.shape[0]
            and stock_xrp.shape[0] == stock.shape[0]
        ), (
            f"Data length mismatch for {ticker_name}: "
            f"{len(stock)}, {len(stock_usdt)}, {len(stock_btc)}, "
            f"{len(stock_eth)}, {len(stock_sol)}, {len(stock_xrp)}"
        )

        df = prepare_df(
            df_ticker=stock,
            df_usdt=stock_usdt,
            df_btc=stock_btc,
            df_eth=stock_eth,
            df_sol=stock_sol,
            df_xrp=stock_xrp,
            horizon=n,
            gain=gain,
        )

        df_train = df.iloc[:-2000]

        df_val = df.iloc[-2000:]

        logger.info(
            "Ticker %s has %d training samples and %d validation samples.",
            ticker_name,
            len(df_train),
            len(df_val),
        )

        X_tr, y_reg_tr, y_cls_tr, y_vol_tr = make_windows(df_train, window_size, n)

        X_tr, y_cls_tr, y_reg_tr, y_vol_tr = balance_binary(
            X_tr, y_cls_tr, y_reg_tr, y_vol_tr
        )

        if X_tr.shape[0] == 0:
            logger.warning(
                f"No training data available for ticker {ticker_name} after balancing."
            )
            return None

        X_va, y_reg_va, y_cls_va, y_vol_va = make_windows(df_val, window_size, n)

        idx = np.random.choice(
            X_va.shape[0], size=int(X_va.shape[0] * 1), replace=False
        )

        X_va = X_va[idx]

        y_reg_va = y_reg_va[idx]

        y_cls_va = y_cls_va[idx]

        y_vol_va = y_vol_va[idx]

        logger.info(
            f"Ticker {ticker_name} processed: "
            f"Train shape: {X_tr.shape}, Val shape: {X_va.shape}"
        )

        hist = torch.bincount(
            torch.from_numpy(y_cls_tr).long() + 1,
            minlength=3,
        ).cpu()

        print(f"{ticker_name} : train (–1,0,+1) → {hist.tolist()}")

        hist_val = torch.bincount(
            torch.from_numpy(y_cls_va).long() + 1, minlength=3
        ).cpu()
        print(f"{ticker_name} : val   (–1,0,+1) → {hist_val.tolist()}")

        return (
            CryptoDataset(
                X=X_tr,
                y_cls=y_cls_tr,
                y_reg=y_reg_tr,
                y_vol=y_vol_tr,
            ),
            CryptoDataset(
                X=X_va,
                y_cls=y_cls_va,
                y_reg=y_reg_va,
                y_vol=y_vol_va,
            ),
        )

    except Exception as e:
        logger.error(
            f"Error processing ticker {ticker_name}, Interval : {interval}: {e}",
            exc_info=True,
        )

        return None


async def create_dataset(
    window_size: int,
    tickers_name: List[str],
    interval: str,
    n: int,
    sym_base_asset: Dict[str, str],
    gain: float,
    max_value: int,
) -> Tuple[CryptoDataset, CryptoDataset] | None:
    """
    Constructs the dataset by extracting signals from each ticker's stock data,
    applying derivative calculations if specified, and combining the data into a single array.
    """

    tasks: List[Coroutine] = []

    stock_btc = await fetch_ohlc(sym="BTCUSDT", intv=interval, bar_needed=max_value)
    stock_eth = await fetch_ohlc(sym="ETHUSDT", intv=interval, bar_needed=max_value)
    stock_xrp = await fetch_ohlc(sym="XRPUSDT", intv=interval, bar_needed=max_value)
    stock_sol = await fetch_ohlc(sym="SOLUSDT", intv=interval, bar_needed=max_value)

    for ticker_name in tickers_name:
        logger.info(f"Processing ticker: {ticker_name}")

        tasks.append(
            create_ticker_dataset(
                window_size=window_size,
                ticker_name=ticker_name,
                interval=interval,
                n=n,
                stock_btc=stock_btc,
                stock_eth=stock_eth,
                stock_xrp=stock_xrp,
                stock_sol=stock_sol,
                max_value=max_value,
                sym_base_asset=sym_base_asset,
                gain=gain,
            )
        )

    full_train_dataset: CryptoDataset | None = None

    full_val_dataset: CryptoDataset | None = None

    results: List[Optional[Tuple[CryptoDataset, CryptoDataset]]] = await asyncio.gather(
        *tasks
    )

    for result in results:
        if result is None:
            continue

        train_dataset, val_dataset = result

        if full_train_dataset is None:
            full_train_dataset = train_dataset

        else:
            full_train_dataset.X = torch.cat(
                [full_train_dataset.X, train_dataset.X], dim=0
            )
            full_train_dataset.y_cls = torch.cat(
                [full_train_dataset.y_cls, train_dataset.y_cls], dim=0
            )

            full_train_dataset.y_vol = torch.cat(
                [full_train_dataset.y_vol, train_dataset.y_vol], dim=0
            )
            full_train_dataset.y_reg = torch.cat(
                [full_train_dataset.y_reg, train_dataset.y_reg], dim=0
            )

        if full_val_dataset is None:
            full_val_dataset = val_dataset
        else:
            full_val_dataset.X = torch.cat([full_val_dataset.X, val_dataset.X], dim=0)
            full_val_dataset.y_cls = torch.cat(
                [full_val_dataset.y_cls, val_dataset.y_cls], dim=0
            )

            full_val_dataset.y_vol = torch.cat(
                [full_val_dataset.y_vol, val_dataset.y_vol], dim=0
            )
            full_val_dataset.y_reg = torch.cat(
                [full_val_dataset.y_reg, val_dataset.y_reg], dim=0
            )

    return full_train_dataset, full_val_dataset


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

    if is_use_saved_ranges:
        if target_ranges_forced is not None:
            target_ranges = target_ranges_forced

        y_min, y_max = target_ranges[key]
    else:
        y_min, y_max = float(y_np.min()), float(y_np.max())
        print(
            f"--> nouveau range pour '{key}': ({y_min}, {y_max}) "
            "→ ajoutez-le dans target_ranges"
        )

        target_ranges = {}

        target_ranges[key] = (y_min, y_max)

    # Min-Max scaling manuel (évite de ré-instancier un MinMaxScaler)
    denom = y_max - y_min if y_max != y_min else 1.0
    y_scaled_np = (y_np - y_min) / denom
    y_scaled_np = np.clip(y_scaled_np, 0.0, 1.0).reshape(-1)  # (N,)

    return _to_same_type(y_scaled_np, y), target_ranges


def save_new_batch(path: str, batch: CryptoDataset) -> None:
    """
    Save the new batch to a file.
    """

    try:
        with open(path, "rb") as f:
            dataset: CryptoDataset = pickle.load(f)

        dataset.X = np.concatenate((dataset.X, batch.X), axis=0)

        dataset.y_cls = np.concatenate((dataset.y_cls, batch.y_cls), axis=0)

        dataset.y_vol = np.concatenate((dataset.y_vol, batch.y_vol), axis=0)

        dataset.y_reg = np.concatenate((dataset.y_reg, batch.y_reg), axis=0)

    except FileNotFoundError:
        print("File not found, creating new dataset.")

        dataset = batch

    print(f"Batch X : {len(batch.X)}")

    print(f"Batch Y : {len(batch.y_cls)}")

    print(f"Batch Y : {len(batch.y_vol)}")

    print(f"Batch Y : {len(batch.y_reg)}")

    print(f"Batch Y : {len(batch.y_reg)}")

    print(f"Dataset X : {len(dataset.X)}")

    print(f"Dataset Y : {len(dataset.y_cls)}")

    print(f"Dataset Y : {len(dataset.y_vol)}")

    print(f"Dataset Y : {len(dataset.y_reg)}")

    print(f"Dataset Y : {len(dataset.y_reg)}")

    with open(path, "wb") as f:
        pickle.dump(dataset, f)


async def run():
    """Main function to run the training process."""

    n = 5

    window_size = 300

    interval = "1h"

    max_value = 100000

    gain = 0.02

    crypto, sym_base_asset = await get_tickers()

    crypto = random.sample(crypto, len(crypto))

    print(f"Crypto : {len(crypto)}")

    step = 10

    index_start = 21

    start = index_start * step

    index_saved = index_start

    print("\n\n\n-----------------------------------------\n\n\n")

    for i in range(start, len(crypto), step):
        print(f"{int(i / step)} / {int(len(crypto) / step)}")

        index_saved += 1

        ticker_dataset_train_batch, ticker_dataset_val_batch = await create_dataset(
            tickers_name=crypto[i : i + step],
            interval=interval,
            window_size=window_size,
            n=n,
            sym_base_asset=sym_base_asset,
            gain=gain,
            max_value=max_value,
        )

        if not ticker_dataset_train_batch:
            print(f"Ticker {crypto[i : i + step]} not found")

            continue

        cls_batch = ticker_dataset_train_batch.y_cls

        hist = torch.bincount(cls_batch + 1, minlength=3).cpu()

        print(f"Distribution batch train (–1,0,+1) : {hist.tolist()}")

        hist_val = torch.bincount(ticker_dataset_val_batch.y_cls + 1, minlength=3).cpu()

        print(f"Distribution batch val   (–1,0,+1) : {hist_val.tolist()}")

        save_new_batch(
            path=f"./ticker_dataset_train_{index_saved}.pickle",
            batch=ticker_dataset_train_batch,
        )

        save_new_batch(
            path=f"./ticker_dataset_val_{index_saved}.pickle",
            batch=ticker_dataset_val_batch,
        )

        print("\n\n\n-----------------------------------------\n\n\n")


if __name__ == "__main__":
    asyncio.run(run())
