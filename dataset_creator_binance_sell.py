"""logger_config.py"""

from __future__ import annotations

import os

import httpx

import pickle

import certifi

import asyncio

from typing import Optional, List, Dict, Any, Union, Coroutine, Tuple

import pandas as pd

import numpy as np

from numpy.typing import NDArray

import torch

from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset as TorchDataset

from binance.spot import Spot

from config.logger_config import logger


os.environ["CURL_CA_BUNDLE"] = certifi.where()

API_KEY = "RRx439X2aBvHzrodPRhgNPAw9hyr48lYFqenjNIjWql25a9kuMMcdV7dRnjE9YsU"

API_SECRET = "RKSfisReVgtRa1bMqrzJjAaVhZ6OjAW9ATdLK3XC9gcBkYuCmOvC9ms77od4OoVp"

BASE_URL = "https://api.binance.com"  # HTTPS comme demandé


class BinanceHandler:
    """
    Identique à la version sync mais totalement asynchrone grâce à httpx.AsyncClient.
    Noms de colonnes et signatures strictement inchangés.
    """

    def __init__(self, main_currency: str):
        self.main_currency: str = main_currency
        # ❶ Le client HTTP réutilisable
        self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=30.0, http2=True)

        self.client_sync = Spot(
            api_key=API_KEY,
            api_secret=API_SECRET,
            base_url="https://api.binance.com",
        )

    # --------------------------------------------------------------------- #
    #  Section : wrappers REST                                              #
    # --------------------------------------------------------------------- #
    async def _query(self, path: str, params: dict | None = None) -> list[list]:
        """Requête GET générique sur l'API Binance (endpoint public)."""

        url = path if path.startswith("http") else f"{BASE_URL}{path}"

        resp = await self._client.get(url, params=params)

        resp.raise_for_status()

        return resp.json()

    async def klines(self, symbol: str, interval: str, **kwargs) -> list[list]:
        """
        Kline/Candlestick Data  —  GET /api/v3/klines
        https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data
        """
        if not symbol or not interval:
            raise ValueError("symbol et interval sont obligatoires")

        params = {"symbol": symbol, "interval": interval, **kwargs}

        return await self._query("/api/v3/klines", params)

    # --------------------------------------------------------------------- #
    #  Section : transformation DataFrame                                   #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _klines_to_df(raw: List[List]) -> pd.DataFrame:
        cols = [
            "OpenTime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "CloseTime",
            "QuoteAssetVolume",
            "NumTrades",
            "TakerBuyBaseVolume",
            "TakerBuyQuoteVolume",
            "Ignore",
        ]

        df = pd.DataFrame(raw, columns=cols)
        df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)

        return df.set_index("OpenTime")[
            [
                "Close",
                "Volume",
                "High",
                "Low",
                "Open",
                "QuoteAssetVolume",
                "NumTrades",
                "TakerBuyBaseVolume",
                "TakerBuyQuoteVolume",
            ]
        ].astype(float)

    # --------------------------------------------------------------------- #
    #  Section : méthode publique identique à la version sync               #
    # --------------------------------------------------------------------- #
    async def get_historical_data_v2(
        self, ticker: str, interval: str, max_value: Optional[int] = None
    ) -> pd.DataFrame:
        ticker_pair: str = ticker

        stock: List[List] = await self.klines(
            symbol=ticker_pair,
            interval=interval,
            limit=1000,
        )

        full_stock: List[List] = stock

        while True:
            stock = await self.klines(
                symbol=ticker_pair,
                interval=interval,
                limit=1000,
                startTime=stock[0][0] - 1000 * 60 * 60 * 24,
            )

            full_stock = stock + full_stock

            if len(stock) < 1000:
                break

            if max_value is not None and len(full_stock) >= max_value:
                full_stock = full_stock[-max_value:]
                break

            # await asyncio.sleep(1)

        print(
            f"Ticker : {ticker_pair}, Interval : {interval} => Done with {len(full_stock)} rows"
        )

        return self._klines_to_df(full_stock)

    # --------------------------------------------------------------------- #
    #  Section : nettoyage                                                  #
    # --------------------------------------------------------------------- #
    async def aclose(self):
        """Ferme proprement le client HTTP."""
        await self._client.aclose()


exchange_info: Dict[str, Any] = BinanceHandler(
    main_currency="USDT"
).client_sync.exchange_info()

symbols = exchange_info.get("symbols", [])

trading_pairs = [s for s in symbols if s.get("status") == "TRADING"]

sym_base_asset: Dict[str, str] = {s["symbol"]: s["baseAsset"] for s in trading_pairs}

tickers: List[str] = [s["symbol"] for s in trading_pairs]

print(tickers)

print(len(tickers))

tickers_not_in_usdt = [ticker for ticker in tickers if not ticker.endswith("USDT")]

print(tickers_not_in_usdt)

print(len(tickers_not_in_usdt))

tickers_in_usdt = [ticker for ticker in tickers if ticker.endswith("USDT")]

tickers_not_in_usdt_but_exist_in_usd = [
    ticker
    for ticker in tickers_not_in_usdt
    if f"{sym_base_asset[ticker]}USDT" in tickers
]

print(tickers_not_in_usdt_but_exist_in_usd)

print(len(tickers_not_in_usdt_but_exist_in_usd))


class MultiTaskSignalDataset(TorchDataset):
    """
    MultiTaskSignalDataset class for multi-task learning.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y_cls: torch.Tensor,
        y_ret: torch.Tensor,
        y_vol: torch.Tensor,
    ):
        self.X = X
        self.y_cls = y_cls.long()  # CrossEntropy → int64
        self.y_ret = y_ret
        self.y_vol = y_vol

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            {"cls": self.y_cls[idx], "ret": self.y_ret[idx], "vol": self.y_vol[idx]},
        )


def create_binary_signal(
    signal: Union[np.ndarray, pd.Series], N: int, c: float
) -> np.ndarray:
    """Create a binary signal based on forward returns."""

    signal = np.asarray(signal, dtype=float)

    length = len(signal)

    binary_signal = np.zeros(length, dtype=float)

    smoothing_window = 3

    for i in range(length - N):
        start_idx = i

        end_idx = i + N

        if i >= smoothing_window:
            window = signal[start_idx : end_idx + 1]

            smoothed_current = np.mean(signal[i - smoothing_window : i + 1])

            min_future = np.min(window)

            if min_future < (1.0 - c) * smoothed_current:
                binary_signal[i] = 2

            elif min_future < smoothed_current:
                binary_signal[i] = 1

            else:
                binary_signal[i] = 0
        else:
            window = signal[start_idx : end_idx + 1]

            if np.any(window < (1.0 - c) * signal[i]):
                binary_signal[i] = 2

            elif np.any(window < (1.0 - (c / 2)) * signal[i]):
                binary_signal[i] = 1

            else:
                binary_signal[i] = 0

    return binary_signal


def compute_balanced_indices(y_cls: np.ndarray) -> np.ndarray:
    """Return balanced indices for 3 classes."""

    idx0, idx1, idx2 = (np.where(y_cls == k)[0] for k in (0, 1, 2))

    print(f"0: {len(idx0)}, 1: {len(idx1)}, 2: {len(idx2)}")

    n = min(len(idx0), len(idx1), len(idx2))

    print(f"n: {n}")

    if n == 0:
        return np.array([])

    np.random.shuffle(idx0)

    np.random.shuffle(idx1)

    np.random.shuffle(idx2)

    balanced = np.concatenate([idx0[:n], idx1[:n], idx2[:n]])

    np.random.shuffle(balanced)

    print(f"balanced: {len(balanced)}")

    return balanced


# %%
def create_sliding_window(data: np.ndarray, window_size: int) -> Optional[np.ndarray]:
    """
    Splits the dataset into overlapping packs of the specified size.

    Args:
        data (np.ndarray): The input time series data.
        window_size (int): The size of each sliding window.

    Returns:
        np.ndarray: A 2D array where each row corresponds to a window.
    """

    num_windows = data.shape[0] - window_size + 1

    if num_windows <= 0:
        logger.warning(
            f"Window size {window_size} is larger than the dataset length {data.shape[0]}."
        )

        return None

    return np.array([data[i : i + window_size] for i in range(num_windows)])


class SignalDerivative:
    """Compute the derivative of a signal (instantaneous variation)."""

    @staticmethod
    def compute_derivative(signal: NDArray) -> NDArray:
        # Conversion en array NumPy pour éviter tout problème de Series
        signal_array = np.asarray(signal)
        return np.diff(signal_array, prepend=signal_array[0])


class SignalLogReturn:
    """Compute the log-return of a signal, useful for relative change measurement."""

    @staticmethod
    def compute_log_return(signal: NDArray) -> NDArray:
        # Assurer que le signal soit 1D
        signal_1d = signal.squeeze()  # Convertit (N,1) en (N,)

        log_prices = np.log(signal_1d)

        log_return = np.diff(log_prices, prepend=log_prices[0])

        return log_return


class SignalRollingVolatility:
    """Compute the rolling volatility of a signal over a given window."""

    @staticmethod
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


class SignalMomentum:
    """Compute the momentum over a given period (difference between current and past values)."""

    @staticmethod
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


class SignalRSI:
    """Compute the Relative Strength Index (RSI) of a signal."""

    @staticmethod
    def compute_rsi(signal: np.ndarray, period: int = 14) -> np.ndarray:
        # Ensure signal is a 1D float array
        signal = signal.flatten().astype(float)

        if len(signal) < 2:
            raise ValueError("Signal must have at least 2 data points to compute RSI.")

        # Initialize RSI with NaNs
        rsi = np.empty_like(signal)
        rsi[:] = np.nan

        # Calculate the differences (delta) of consecutive prices
        delta = np.zeros_like(signal)  # delta has the same shape as signal
        delta[1:] = np.diff(signal)  # The first delta is zero

        # Calculate gains and losses
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        # Calculate the initial rolling average for the first period
        if len(signal) <= period:
            raise ValueError(
                f"Signal must have more than {period} data points. Received {len(signal)} points."
            )

        avg_gain = np.mean(gains[:period])

        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            rsi[period] = 100  # If there are no losses, RSI is 100

        else:
            rs = avg_gain / avg_loss

            rsi[period] = 100.0 - (100.0 / (1.0 + rs))

        # Use exponential moving average (EMA) for the subsequent RSI values
        for i in range(period + 1, len(signal)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi[i] = 100  # Avoid division by zero, if no losses, RSI = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        return rsi / 100


class SignalDateTime:
    """Compute unique linear time features for a week, normalized to [0, 1]."""

    @staticmethod
    def compute_time_features(date_index: pd.DatetimeIndex) -> np.ndarray:
        """Compute unique linear time features for a week, normalized to [0, 1].

        Args:
            date_index (pd.DatetimeIndex): The index of dates to compute features for.

        Returns:
            np.ndarray: A 1D array of normalized linear values representing time over a week.
        """

        day_of_week = date_index.dayofweek.values

        hour = date_index.hour.values

        total_hours = day_of_week * 24 + hour

        normalized_features = total_hours / 168

        return normalized_features


class SignalMACD:
    """Compute the Moving Average Convergence Divergence (MACD) of a price signal."""

    @staticmethod
    def compute_macd(
        signal: NDArray, fast: int = 12, slow: int = 26, signal_period: int = 9
    ) -> List[NDArray]:
        """
        Compute the MACD line, signal line, and MACD histogram for the given price signal.

        Args:
            signal (NDArray): A 1D or 2D NumPy array of price data (e.g., closing prices).
            fast (int, optional): The period for the fast EMA. Defaults to 12.
            slow (int, optional): The period for the slow EMA. Defaults to 26.
            signal_period (int, optional): The period for the signal EMA. Defaults to 9.

        Returns:
            NDArray: A 2D array of shape (n_samples, 3) where:
                - column 0 = MACD line
                - column 1 = Signal line
                - column 2 = MACD histogram (macd_line - signal_line)
        """
        # Ensure signal is 1D
        signal_1d = signal.squeeze().astype(float)

        def compute_ema(prices: NDArray, period: int) -> NDArray:
            """Helper function to compute Exponential Moving Average (EMA)."""
            ema = np.zeros_like(prices)
            alpha = 2.0 / (period + 1.0)
            ema[0] = prices[0]
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
            return ema

        # Compute EMAs
        fast_ema = compute_ema(signal_1d, fast)
        slow_ema = compute_ema(signal_1d, slow)

        # MACD line
        macd_line = fast_ema - slow_ema

        # Signal line
        signal_line = compute_ema(macd_line, signal_period)

        # Histogram
        histogram = macd_line - signal_line

        return [
            np.asarray(macd_line).reshape(-1, 1),
            np.asarray(signal_line).reshape(-1, 1),
            np.asarray(histogram).reshape(-1, 1),
        ]


class SignalBollingerBands:
    """Compute Bollinger Bands for a price signal."""

    @staticmethod
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


class SignalStochasticOscillator:
    """Compute the Stochastic Oscillator (%K and %D) for a given price signal."""

    @staticmethod
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
            high (NDArray): High prices (1D).
            low (NDArray):  Low prices (1D).
            close (NDArray): Close prices (1D).
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


class SignalOBV:
    """Compute the On-Balance Volume (OBV) for a given price signal."""

    @staticmethod
    def compute_obv(signal: NDArray, volume: NDArray) -> NDArray:
        """
        Compute On-Balance Volume (OBV) given price (e.g., close) and volume arrays.

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


class SignalATR:
    """Compute the Average True Range (ATR) for a given price signal."""

    @staticmethod
    def compute_atr(
        high: NDArray, low: NDArray, close: NDArray, window: int = 14
    ) -> NDArray:
        """
        Compute the Average True Range (ATR) for the given price arrays.

        Args:
            high (NDArray): High prices (1D).
            low (NDArray): Low prices (1D).
            close (NDArray): Close prices (1D).
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


def make_daily_columns(df_slice: pd.DataFrame, rows: int):
    """Retourne colonnes Open / dérivée (shape = rows, 1).
    Renvoie (None, None) si la fenêtre doit être sautée."""
    if df_slice.empty:
        return None, None  # slice vide → skip fenêtre

    diff = rows - len(df_slice)
    if diff > 0:
        if diff < 100:
            return None, None  # trou <100 → skip
        pad = np.zeros((diff, 1), dtype=float)
        opens = df_slice["Open"].to_numpy().reshape(-1, 1)
        deriv = SignalDerivative.compute_derivative(df_slice["Open"]).reshape(-1, 1)
        open_col = np.vstack((pad, opens))
        deriv_col = np.vstack((pad, deriv))
    else:
        open_col = df_slice["Open"].to_numpy().reshape(-1, 1)
        deriv_col = SignalDerivative.compute_derivative(df_slice["Open"]).reshape(-1, 1)

    return open_col, deriv_col


def make_min_columns(df_slice: pd.DataFrame, rows: int):
    if df_slice.empty:
        return None, None

    diff = rows - len(df_slice)
    if diff > 0:
        pad = np.zeros((diff, 1), dtype=float)
        opens = df_slice["Open"].to_numpy().reshape(-1, 1)
        deriv = SignalDerivative.compute_derivative(df_slice["Open"]).reshape(-1, 1)
        open_col = np.vstack((pad, opens))
        deriv_col = np.vstack((pad, deriv))
    else:
        open_col = df_slice["Open"].to_numpy().reshape(-1, 1)
        deriv_col = SignalDerivative.compute_derivative(df_slice["Open"]).reshape(-1, 1)

    return open_col, deriv_col


async def create_ticker_dataset(
    window_size: int,
    ticker_name: str,
    interval: str,
    c: float,
    n: int,
    momentum_period: int,
    rsi_period: int,
    stock_btc: pd.DataFrame,
    max_value: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    try:
        binance_handler = BinanceHandler(main_currency="USDT")

        tasks = [
            binance_handler.get_historical_data_v2(
                ticker=ticker_name, interval=interval, max_value=max_value
            ),
            binance_handler.get_historical_data_v2(
                ticker=f"{sym_base_asset[ticker_name]}USDT",
                interval=interval,
                max_value=max_value,
            ),
            binance_handler.get_historical_data_v2(
                ticker=ticker_name, interval="1m", max_value=max_value
            ),
            binance_handler.get_historical_data_v2(
                ticker=ticker_name, interval="1d", max_value=max_value
            ),
        ]

        stock, stock_usdt, stock_1d, stock_1m = await asyncio.gather(*tasks)

        if stock is None:
            return None

        if stock.shape[0] < window_size:
            return None

        signal_data = []

        signal_data.append(np.asarray(stock.index).reshape(-1, 1))

        signal_data.append(
            SignalDateTime.compute_time_features(
                stock.index,
            ).reshape(-1, 1)
        )

        signal_data.append(np.asarray(stock["Open"].values).reshape(-1, 1))

        signal_data.append(np.asarray(stock["High"].values).reshape(-1, 1))

        signal_data.append(np.asarray(stock["Low"].values).reshape(-1, 1))

        signal_data.append(np.asarray(stock["Close"].values).reshape(-1, 1))

        signal_data.append(np.asarray(stock["Volume"].values).reshape(-1, 1))

        signal_data.append(np.asarray(stock["QuoteAssetVolume"].values).reshape(-1, 1))

        signal_data.append(np.asarray(stock["NumTrades"].values).reshape(-1, 1))

        signal_data.append(
            np.asarray(stock["TakerBuyBaseVolume"].values).reshape(-1, 1)
        )

        signal_data.append(
            np.asarray(stock["TakerBuyQuoteVolume"].values).reshape(-1, 1)
        )

        signal_data.append(
            SignalDerivative.compute_derivative(stock["Open"]).reshape(-1, 1)
        )

        signal_data.append(
            SignalDerivative.compute_derivative(stock["High"]).reshape(-1, 1)
        )

        signal_data.append(
            SignalDerivative.compute_derivative(stock["Low"]).reshape(-1, 1)
        )

        signal_data.append(
            SignalDerivative.compute_derivative(stock["Close"]).reshape(-1, 1)
        )

        signal_data.append(
            SignalDerivative.compute_derivative(stock["Volume"]).reshape(-1, 1)
        )

        signal_data.append(
            SignalMomentum.compute_momentum(
                stock["Open"].values.reshape(-1, 1),
                momentum_period,
            ).reshape(-1, 1)
        )

        signal_data.append(
            SignalRSI.compute_rsi(
                stock["Open"].values.reshape(-1, 1),
                rsi_period,
            ).reshape(-1, 1)
        )

        signal_data += SignalMACD.compute_macd(
            stock["Open"].values.reshape(-1, 1),
        )

        signal_data += SignalStochasticOscillator.compute_stoch_oscillator(
            stock["High"].values.reshape(-1, 1),
            stock["Low"].values.reshape(-1, 1),
            stock["Close"].values.reshape(-1, 1),
        )

        len_main_stock = len(stock)

        stock_btc = stock_btc[-len_main_stock:]

        if len(stock_btc) < len(stock):
            logger.info("Will pad BTCUSDT with 0 value at the beginning.")

            stock_btc = pd.concat(
                [
                    pd.DataFrame(
                        0,
                        index=pd.date_range(
                            start=stock.index[0], end=stock.index[0], freq="T"
                        ),
                        columns=stock_btc.columns,
                    ),
                    stock_btc,
                ]
            )

        signal_data.append(np.asarray(stock_btc["Open"]).reshape(-1, 1))

        signal_data.append(np.asarray(stock_btc["Volume"]).reshape(-1, 1))

        signal_data.append(
            SignalDerivative.compute_derivative(stock_btc["Open"]).reshape(-1, 1)
        )

        signal_data.append(
            np.asarray(stock_btc["QuoteAssetVolume"].values).reshape(-1, 1)
        )

        signal_data.append(np.asarray(stock_btc["NumTrades"].values).reshape(-1, 1))

        signal_data.append(
            np.asarray(stock_btc["TakerBuyBaseVolume"].values).reshape(-1, 1)
        )

        signal_data.append(
            np.asarray(stock_btc["TakerBuyQuoteVolume"].values).reshape(-1, 1)
        )

        if len(stock_usdt) < len(stock):
            logger.warning("USDT too short")

            return None

        signal_data.append(np.asarray(stock_usdt["Open"]).reshape(-1, 1))

        signal_data.append(np.asarray(stock_usdt["Volume"]).reshape(-1, 1))

        signal_data.append(
            SignalDerivative.compute_derivative(stock_usdt["Open"]).reshape(-1, 1)
        )

        signal_data.append(
            np.asarray(stock_usdt["QuoteAssetVolume"].values).reshape(-1, 1)
        )

        signal_data.append(np.asarray(stock_usdt["NumTrades"].values).reshape(-1, 1))

        signal_data.append(
            np.asarray(stock_usdt["TakerBuyBaseVolume"].values).reshape(-1, 1)
        )

        signal_data.append(
            np.asarray(stock_usdt["TakerBuyQuoteVolume"].values).reshape(-1, 1)
        )

        data_np: np.ndarray = np.concatenate(signal_data, axis=1)

        max_initial_na = max(
            momentum_period,
            rsi_period,
        )

        data_np = data_np[max_initial_na:, :]

        data_np = data_np[:-n,]

        y_data = create_binary_signal(
            signal=stock["Open"][max_initial_na:],
            c=c,
            N=n,
        )[window_size - 1 : -n].reshape(-1, 1)

        mask = (np.roll(y_data, 1) == 0) & (np.roll(y_data, -1) == 0)

        y_data[mask] = 0

        y_cls = y_data.reshape(-1, 1)

        open = stock["Open"].astype(float).values

        volume = stock["Volume"].astype(float).values

        fwd_log_ret = open[n:] - open[:-n]

        fwd_log_ret = fwd_log_ret[max_initial_na:]

        fwd_log_ret = fwd_log_ret[window_size - 1 :]

        y_ret = fwd_log_ret.reshape(-1, 1)

        y_vol = volume.reshape(-1, 1)

        y_vol = y_vol[max_initial_na:]

        y_vol = y_vol[window_size - 1 : -n]

        windowed_data = create_sliding_window(data=data_np, window_size=window_size)

        if windowed_data is None:
            return None

        stock_1d = stock_1d.sort_index()

        stock_1d = stock_1d[~stock_1d.index.duplicated(keep="first")]

        stock_1m = stock_1m.sort_index().loc[~stock_1m.index.duplicated(keep="first")]

        if stock_1m.index.tz is None:
            stock_1m.index = stock_1m.index.tz_localize("UTC")

        new_windowed_data = []

        new_y_cls, new_y_ret, new_y_vol = [], [], []

        for i, window in enumerate(windowed_data):
            last_val = window[-1, 0]

            if isinstance(last_val, pd.Timestamp):
                last_ts = last_val if last_val.tzinfo else last_val.tz_localize("UTC")
            elif isinstance(last_val, np.datetime64):
                last_ts = pd.Timestamp(last_val, tz="UTC")
            else:
                last_ts = pd.to_datetime(last_val, unit="ns", utc=True)

            end_15 = last_ts.floor("15T")

            need_rows = len(window) - 1

            end_daily = (last_ts - pd.Timedelta(days=1)).normalize()

            stock_1d_slice = stock_1d.loc[:end_daily].tail(need_rows)

            stock_1m_slice = stock_1m.loc[:end_15].tail(need_rows)

            open_col, deriv_col = make_daily_columns(stock_1d_slice, need_rows)

            open_col_1m, deriv_col_1m = make_min_columns(stock_1m_slice, need_rows)

            if (
                open_col is None
                or open_col_1m is None
                or deriv_col is None
                or deriv_col_1m is None
            ):
                continue

            new_window = np.concatenate(
                (window[1:, :], open_col, deriv_col, open_col_1m, deriv_col_1m),
                axis=1,
            )

            new_windowed_data.append(new_window)

            new_y_cls.append(y_data[i])

            new_y_ret.append(y_ret[i])

            new_y_vol.append(y_vol[i])

        if not new_windowed_data:
            logger.warning("No valid windows after processing.")

            return None

        windowed_data = np.stack(new_windowed_data, axis=0)

        windowed_data = windowed_data[:, :, 1:]

        windowed_data = windowed_data.astype(np.float32)

        if np.isnan(np.asarray(windowed_data)).any():
            return None

        y_cls = np.asarray(new_y_cls, dtype=np.int8).reshape(-1, 1)

        y_ret = np.asarray(new_y_ret, dtype=np.float32).reshape(-1, 1)

        y_vol = np.asarray(new_y_vol, dtype=np.float32).reshape(-1, 1)

        print(
            f"Tickers : {ticker_name}, Interval : {interval} => X shape: {windowed_data.shape}, y_cls shape: {y_cls.shape}, y_ret shape: {y_ret.shape}, y_vol shape: {y_vol.shape}"
        )

        return windowed_data, y_cls, y_ret, y_vol

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
    c: float,
    n: int,
    momentum_period: int,
    rsi_period: int,
) -> Any:
    """
    Constructs the dataset by extracting signals from each ticker's stock data,
    applying derivative calculations if specified, and combining the data into a single array.
    """
    x_dataset_list: List[NDArray] = []

    y_cls_list, y_ret_list, y_vol_list = [], [], []

    tasks: List[Coroutine] = []

    max_value = 20000

    binance_handler = BinanceHandler(main_currency="USDT")

    stock_btc = await binance_handler.get_historical_data_v2(
        ticker="BTCUSDT", interval=interval, max_value=max_value
    )

    for ticker_name in tickers_name:
        logger.info(f"Processing ticker: {ticker_name}")

        tasks.append(
            create_ticker_dataset(
                window_size=window_size,
                ticker_name=ticker_name,
                interval=interval,
                c=c,
                n=n,
                momentum_period=momentum_period,
                rsi_period=rsi_period,
                stock_btc=stock_btc,
                max_value=max_value,
            )
        )

    results = await asyncio.gather(*tasks)

    for res in results:
        if res is None:
            continue

        x_dataset, y_cls, y_ret, y_vol = res

        if x_dataset is None or y_cls is None or y_ret is None or y_vol is None:
            continue

        if x_dataset.shape[0] == y_cls.shape[0] == y_ret.shape[0] == y_vol.shape[0]:
            x_dataset_list.append(x_dataset)

            y_cls_list.append(y_cls)

            y_ret_list.append(y_ret)

            y_vol_list.append(y_vol)

    if not x_dataset_list:
        logger.warning("No valid datasets found.")

        return None

    X = np.concatenate(x_dataset_list)

    print(f"X shape: {X.shape}")

    y_cls = np.concatenate(y_cls_list).flatten()

    y_ret = np.concatenate(y_ret_list).flatten()

    y_vol = np.concatenate(y_vol_list).flatten()

    idx = compute_balanced_indices(y_cls)

    if len(idx) == 0:
        logger.warning("No balanced indices found.")

        return None

    X = X[idx]

    y_cls = y_cls[idx]

    y_ret = y_ret[idx]

    y_vol = y_vol[idx]

    X_t = torch.tensor(X, dtype=torch.float32)

    y_cls = torch.tensor(y_cls, dtype=torch.long)

    y_ret = torch.tensor(y_ret, dtype=torch.float32)

    y_vol = torch.tensor(y_vol, dtype=torch.float32)

    return MultiTaskSignalDataset(X_t, y_cls, y_ret, y_vol)


feature_ranges = {
    0: (0.0, 0.9940476417541504),
    1: (0.0, 106458.5234375),
    2: (0.0, 106581.734375),
    3: (24865.142578125, 108240.0859375),
    4: (0.0, 35653705728.0),
    5: (1.2170149332746405e-08, 110011.2578125),
    6: (-19908.34375, 19450.6640625),
    7: (-19687.2890625, 18940.6328125),
    8: (0.0, 1.0),
    9: (-3587.860107421875, 2443.761962890625),
    10: (-3145.3359375, 2052.35693359375),
    11: (-1569.955810546875, 1052.1256103515625),
    12: (0.0, 1.0),
    13: (0.0, 1.0),
    14: (0.0, 132717666304.0),
    15: (-126326300672.0, 101510324224.0),
}


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


target_ranges = {"ret": (-12911.5625, 18669.859375), "vol": (0.0, 35677515776.0)}


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


def save_new_batch(path: str, batch: Any) -> None:
    """
    Save the new batch to a file.
    """

    try:
        with open(path, "rb") as f:
            dataset = pickle.load(f)

        dataset.X = np.concatenate((dataset.X, batch.X), axis=0)

        dataset.y_cls = np.concatenate((dataset.y_cls, batch.y_cls), axis=0)

        dataset.y_vol = np.concatenate((dataset.y_vol, batch.y_vol), axis=0)

        dataset.y_ret = np.concatenate((dataset.y_ret, batch.y_ret), axis=0)

    except FileNotFoundError:
        print("File not found, creating new dataset.")

        dataset = batch

    print(f"Batch X : {len(batch.X)}")

    print(f"Batch Y : {len(batch.y_cls)}")

    print(f"Batch Y : {len(batch.y_vol)}")

    print(f"Batch Y : {len(batch.y_ret)}")

    print(f"Dataset X : {len(dataset.X)}")

    print(f"Dataset Y : {len(dataset.y_cls)}")

    print(f"Dataset Y : {len(dataset.y_vol)}")

    print(f"Dataset Y : {len(dataset.y_ret)}")

    with open(path, "wb") as f:
        pickle.dump(dataset, f)


async def run():
    """Main function to run the training process."""

    c = 0.05

    n = 5

    window_size = 300

    interval = "1h"

    crypto = tickers_not_in_usdt_but_exist_in_usd

    split_dataset = int(len(crypto) * 0.9)

    tickers_name_train = crypto[:split_dataset]

    tickers_name_val = crypto[split_dataset:]

    print(f"Crypto : {len(crypto)}")

    print(f"Ticker train : {len(tickers_name_train)}")

    print(f"Ticker val : {len(tickers_name_val)}")

    step = 5

    start = 0 * step

    index_saved = start + 1

    print("\n\n\n-----------------------------------------\n\n\n")

    for i in range(start, len(tickers_name_train), step):
        print(f"{int(i / step)} / {int(len(tickers_name_train) / step)}")

        if i > 10:
            index_saved += 1

        ticker_dataset_train_batch = await create_dataset(
            tickers_name=tickers_name_train[i : i + step],
            interval=interval,
            window_size=window_size,
            c=c,
            n=n,
            momentum_period=5,
            rsi_period=5,
        )

        if not ticker_dataset_train_batch:
            print(f"Ticker {tickers_name_train[i : i + step]} not found")

            continue

        save_new_batch(
            path=f"./ticker_dataset_train__sell_{index_saved}.pickle",
            batch=ticker_dataset_train_batch,
        )

        print("\n\n\n-----------------------------------------\n\n\n")

    start = 0

    index_saved = start + 1

    for i in range(start, len(tickers_name_val), step):
        print(tickers_name_val[i : i + step])

        index_saved += 1

        try:
            ticker_dataset_val_batch = await create_dataset(
                tickers_name=tickers_name_val[i : i + step],
                interval=interval,
                window_size=window_size,
                c=c,
                n=n,
                momentum_period=5,
                rsi_period=5,
            )

            if not ticker_dataset_val_batch:
                print(f"Ticker {tickers_name_val[i : i + step]} not found")

                continue

            save_new_batch(
                path=f"./ticker_dataset_val_sell_{index_saved}.pickle",
                batch=ticker_dataset_val_batch,
            )

        except Exception as e:
            print(e)


def scale_dataset():
    with open("./ticker_dataset_train_sell.pickle", "rb") as f:
        ticker_dataset_train = pickle.load(f)

    # Load the validation dataset
    with open("./ticker_dataset_val_sell.pickle", "rb") as f:
        ticker_dataset_val = pickle.load(f)

    print(f"Train shape : {ticker_dataset_train.X.shape}")

    print(f"Val shape : {ticker_dataset_val.X.shape}")

    ticker_dataset_train.X, feature_ranges = scale(
        X=ticker_dataset_train.X, is_use_feature_ranges=False
    )

    ticker_dataset_val.X, _ = scale(
        X=ticker_dataset_val.X,
        is_use_feature_ranges=True,
        feature_ranges_forced=feature_ranges,
    )

    ticker_dataset_train.y_ret, target_ranges_ret = scale_target(
        ticker_dataset_train.y_ret, key="ret", is_use_saved_ranges=False
    )

    ticker_dataset_val.y_ret, _ = scale_target(
        ticker_dataset_val.y_ret,
        key="ret",
        is_use_saved_ranges=True,
        target_ranges_forced=target_ranges_ret,
    )

    ticker_dataset_train.y_vol, target_ranges_vol = scale_target(
        ticker_dataset_train.y_vol, key="vol", is_use_saved_ranges=False
    )

    ticker_dataset_val.y_vol, _ = scale_target(
        ticker_dataset_val.y_vol,
        key="vol",
        is_use_saved_ranges=True,
        target_ranges_forced=target_ranges_vol,
    )

    print(f"Max X train : {np.max(np.asarray(ticker_dataset_train.X))}")

    print(f"Min X train : {np.min(np.asarray(ticker_dataset_train.X))}")

    print(f"Max X val : {np.max(np.asarray(ticker_dataset_val.X))}")

    print(f"Min X val : {np.min(np.asarray(ticker_dataset_val.X))}")

    print(f"Max Y cls train : {np.max(np.asarray(ticker_dataset_train.y_cls))}")

    print(f"Min Y cls train : {np.min(np.asarray(ticker_dataset_train.y_cls))}")

    print(f"Max Y cls val : {np.max(np.asarray(ticker_dataset_val.y_cls))}")

    print(f"Min Y cls val : {np.min(np.asarray(ticker_dataset_val.y_cls))}")

    print(f"Max Y ret train : {np.max(np.asarray(ticker_dataset_train.y_ret))}")

    print(f"Min Y ret train : {np.min(np.asarray(ticker_dataset_train.y_ret))}")

    print(f"Max Y ret val : {np.max(np.asarray(ticker_dataset_val.y_ret))}")

    print(f"Min Y ret val : {np.min(np.asarray(ticker_dataset_val.y_ret))}")

    print(f"Max Y vol train : {np.max(np.asarray(ticker_dataset_train.y_vol))}")

    print(f"Min Y vol train : {np.min(np.asarray(ticker_dataset_train.y_vol))}")

    print(f"Max Y vol val : {np.max(np.asarray(ticker_dataset_val.y_vol))}")

    print(f"Min Y vol val : {np.min(np.asarray(ticker_dataset_val.y_vol))}")

    print(f"Train shape : {ticker_dataset_train.X.shape}")

    print(f"Val shape : {ticker_dataset_val.X.shape}")

    pickle.dump(
        ticker_dataset_train,
        open("./ticker_dataset_train_sell_scaled.pickle", "wb"),
    )

    pickle.dump(
        ticker_dataset_val,
        open("./ticker_dataset_val_sell_scaled.pickle", "wb"),
    )

    with open("./ticker_dataset_train_sell_scaled.pickle", "rb") as f:
        ticker_dataset_train_scaled = pickle.load(f)

    # Load the validation dataset
    with open("./ticker_dataset_val_sell_scaled.pickle", "rb") as f:
        ticker_dataset_val_scaled = pickle.load(f)

    print(f"Train shape : {ticker_dataset_train_scaled.X.shape}")

    print(f"Val shape : {ticker_dataset_val_scaled.X.shape}")


if __name__ == "__main__":
    asyncio.run(run())

    # scale_dataset()
