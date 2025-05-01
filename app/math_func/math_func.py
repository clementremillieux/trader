"""Compute various mathematical functions for signals."""

from typing import List, Union
import numpy as np

import pandas as pd

from numpy.typing import NDArray

from sklearn.preprocessing import MinMaxScaler

import torch


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


class Scaler:
    """Mise à l'échelle des données dans [0, 1] pour chaque feature."""

    feature_ranges = {
        0: (0.0476190485060215, 0.6726190447807312),
        1: (0.10119999945163727, 77230.0),
        2: (-3940.0, 9880.0),
        3: (-5420.0, 9520.0),
        4: (0.001832408714108169, 0.9999982714653015),
        5: (-2711.1240234375, 1537.8070068359375),
        6: (-1943.899169921875, 1330.8485107421875),
        7: (-801.3096923828125, 825.0460205078125),
        8: (0.0, 1.0),
        9: (0.0, 1.0),
        10: (0.0, 1197713920.0),
        11: (-1048523776.0, 1166991104.0),
    }

    def _to_numpy(self, arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Torch → NumPy sans lien au graphe de calcul."""
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()

        return arr

    def _to_same_type(self, arr_np: np.ndarray, ref: Union[np.ndarray, torch.Tensor]):
        """Reconvertit en Torch si `ref` était Torch, sinon laisse en NumPy."""

        if isinstance(ref, torch.Tensor):
            return torch.from_numpy(arr_np).to(ref.device).type_as(ref)
        return arr_np

    def scale(
        self, X: Union[np.ndarray, torch.Tensor], is_use_feature_ranges: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Scale data to [0, 1] using MinMaxScaler.
        """

        X_np = self._to_numpy(X)

        n, t, f = X_np.shape

        X_2d = X_np.reshape(-1, f)

        if is_use_feature_ranges:
            feature_min = np.array(
                [self.feature_ranges[i][0] for i in range(f)], dtype=np.float32
            )
            feature_max = np.array(
                [self.feature_ranges[i][1] for i in range(f)], dtype=np.float32
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

        scaler = MinMaxScaler(feature_range=(0, 1))

        scaler.n_features_in_ = f

        scaler.data_min_ = feature_min

        scaler.data_max_ = feature_max

        scaler.data_range_ = np.where(
            feature_max - feature_min == 0, 1.0, feature_max - feature_min
        )

        scaler.scale_ = 1.0 / scaler.data_range_

        scaler.min_ = -feature_min * scaler.scale_

        X_scaled_2d = scaler.transform(X_2d)

        X_scaled_np = X_scaled_2d.reshape(n, t, f)

        return self._to_same_type(X_scaled_np, X)
