"""DatasetCreator class to create datasets for machine learning."""

import asyncio

from typing import Any, Dict, Optional, Tuple

import httpx

import numpy as np

from numpy.typing import NDArray

import pandas as pd

import torch

from torch.utils.data import Dataset

from config.logger_config import logger


class CryptoDataset(Dataset):
    """CryptoDataset class to handle cryptocurrency datasets for machine learning."""

    def __init__(self, X):
        """Initializes the CryptoDataset with features and labels."""

        self.X = torch.tensor(X, dtype=torch.float32)

    def __getitem__(self, idx):
        return (self.X[idx],)

    def __len__(self):
        return len(self.X)


class DatasetCreator:
    """DatasetCreator class to create datasets for machine learning."""

    def __init__(self):
        """Initializes the DatasetCreator with a BinanceHandler instance."""

        BASE_URL = "https://api.binance.com"

        MAX_CONC = 4

        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=MAX_CONC),
        )

    async def _json(self, path: str, params: Dict[str, Any]) -> Any:
        params = {k: v for k, v in params.items() if v not in (None, "")}

        logger.debug("GET %s %s", path, params)

        r = await self.client.get(path, params=params)

        r.raise_for_status()

        return r.json()

    async def fetch_ohlc(self, sym: str, intv: str, bar_needed: int) -> pd.DataFrame:
        """Retourne au moins *BARS_NEEDED* bougies pour (sym,intv)."""

        logger.info(f"↓ {sym:<10} {intv}")

        rows, start = [], None

        while len(rows) < bar_needed:
            batch = await self._json(
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

        logger.debug(f"{sym} {intv} → {len(df)} rows (cached)")

        return df

    def safe_diff(self, arr: pd.Series) -> np.ndarray:
        """Renvoie np.diff(arr) mais padde le premier élément pour conserver la longueur."""

        a = arr.to_numpy()
        diff = np.diff(a, prepend=a[0])
        return diff

    def safe_log_ret(self, arr: pd.Series) -> np.ndarray:
        """Rend la variation logarithmique tout en conservant la longueur (pad au début)."""

        a = arr.to_numpy()
        logp = np.log(a)
        d = np.diff(logp, prepend=logp[0])
        return d

    def compute_rolling_volatility(self, signal: NDArray, window: int = 14) -> NDArray:
        # S'assurer que signal est de type float
        signal = signal.astype(float)

        vol = np.empty_like(signal)  # vol aura le même dtype que signal, donc float
        vol[:] = np.nan  # maintenant pas de soucis pour assigner NaN

        for i in range(len(signal)):
            start = max(0, i - window + 1)
            window_slice = signal[start : i + 1]
            vol[i] = np.std(window_slice) if len(window_slice) > 1 else 0.0

        return vol

    def compute_momentum(self, signal: NDArray, period: int = 5) -> NDArray:
        # Convertir signal en float pour éviter les problèmes de type
        signal = signal.astype(float)

        # Maintenant mom sera aussi en float
        mom = np.empty_like(signal)

        mom[:] = np.nan

        for i in range(len(signal)):
            if i >= period:
                mom[i] = signal[i] - signal[i - period]

        return mom

    def _rsi(self, s: pd.Series, n: int = 14) -> np.ndarray:
        """Compute the Relative Strength Index (RSI) for a given series."""

        d = s.diff()

        g = d.clip(lower=0).rolling(n).mean()

        l = (-d.clip(upper=0)).rolling(n).mean()

        return np.array(1 - 1 / (1 + g / (l + 1e-9)))

    def _atr(self, h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> np.ndarray:
        """Compute the Average True Range (ATR) for given high, low, and close prices."""
        tr = pd.concat(
            [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
        ).max(axis=1)
        return np.array(tr.rolling(n).mean())

    def _bollinger_width(self, s: pd.Series, n: int = 20, k: float = 2.0):
        """Compute the Bollinger Bands width for a given series."""

        ma = s.rolling(n).mean()
        std = s.rolling(n).std()
        return (k * std * 2) / (ma + 1e-9)

    def compute_time_features(self, date_index: pd.DatetimeIndex) -> np.ndarray:
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

    def _ema(self, s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    def _macd(self, s: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the MACD (Moving Average Convergence Divergence) for a given series."""

        macd = self._ema(s, 12) - self._ema(s, 26)

        return np.array(macd), np.array(self._ema(macd, 9))

    def _stoch_k(
        self, c: pd.Series, h: pd.Series, l: pd.Series, n: int = 14
    ) -> np.ndarray:
        """Compute the Stochastic %K for a given close, high, and low prices."""
        low_n = l.rolling(n).min()

        high_n = h.rolling(n).max()

        return np.array((c - low_n) / (high_n - low_n + 1e-9))

    def compute_bollinger_bands(
        self, signal: NDArray, window: int = 20, num_std_dev: int = 2
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
        self,
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

    def compute_obv(self, signal: NDArray, volume: NDArray) -> NDArray:
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
        self, high: NDArray, low: NDArray, close: NDArray, window: int = 14
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

    def volumetric_features(self, df: pd.DataFrame, window: int = 24) -> np.ndarray:
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
        out["imbalance"] = (df["taker_buy_quote"] - sell_quote) / (
            df["quote_vol"] + 1e-9
        )

        return np.array(out)

    def micro_volatility(self, df: pd.DataFrame, span: int = 24) -> np.ndarray:
        """
        Volatilité réalisée & mesures de roughness pour HAR ou hétéroscédasticité.
        • RV  : somme des ret² intra-h sur `span` h
        • BPV : bipower variation
        • QIV : quarticity (proxy « turbulence »)
        On suppose que df est en pas de temps 1h.
        """
        # 1) calcul du log-return sûr, reconverti en Series pour rolling
        log_ret_arr = self.safe_log_ret(df["close"])  # array de longueur N
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

    def prepare_df(
        self,
        df_ticker: pd.DataFrame,
        df_usdt: pd.DataFrame,
        df_btc: pd.DataFrame,
        df_eth: pd.DataFrame,
        df_sol: pd.DataFrame,
        df_xrp: pd.DataFrame,
    ) -> pd.DataFrame:
        df_ticker = df_ticker.copy()

        df_ticker["atr"] = self._atr(
            df_ticker["high"], df_ticker["low"], df_ticker["close"], n=14
        )

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

        time_features = self.compute_time_features(df_ticker.index)

        for i, name in enumerate(["hour_sin", "hour_cos", "dow_sin", "dow_cos"]):
            df_ticker[name] = time_features[:, i]

        df_ticker["close_deriv"] = self.safe_diff(df_ticker["close"])

        df_ticker["open_deriv"] = self.safe_diff(df_ticker["open"])

        df_ticker["high_deriv"] = self.safe_diff(df_ticker["high"])

        df_ticker["low_deriv"] = self.safe_diff(df_ticker["low"])

        df_ticker["volume_deriv"] = self.safe_diff(df_ticker["volume"])

        df_ticker["momentum"] = self.compute_momentum(
            df_ticker["close"].values.reshape(-1, 1),
        )

        df_ticker["rsi"] = self._rsi(df_ticker["close"])

        df_ticker["log_ret"] = self.safe_log_ret(df_ticker["close"])

        df_ticker["bollinger_width"] = self._bollinger_width(
            df_ticker["close"],
        )
        df_ticker["macd"], df_ticker["macd_signal"] = self._macd(
            df_ticker["close"],
        )

        df_ticker["macd_diff"] = df_ticker["macd"] - df_ticker["macd_signal"]

        df_ticker["stoch_k"] = self._stoch_k(
            df_ticker["close"],
            df_ticker["high"],
            df_ticker["low"],
        )

        df_ticker["log_ret_btc"] = self.safe_log_ret(
            df_ticker["close"]
        ) - self.safe_log_ret(df_btc["close"])

        vf = self.volumetric_features(df_ticker)
        for i, name in enumerate(
            ["v_rel", "qv_rel", "trade_rate", "buy_pressure", "imbalance"]
        ):
            df_ticker[name] = vf[:, i]

        mv = self.micro_volatility(df_ticker)

        cols = ["rv", "bpv", "qiv"]

        for i, col in enumerate(cols):
            df_ticker[col] = mv[:, i]

        df_ticker["btc_close"] = df_btc["close"].values

        df_ticker["btc_volume"] = df_btc["volume"].values

        df_ticker["btc_close_deriv"] = self.safe_diff(df_btc["close"])

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

    def make_windows(self, df: pd.DataFrame, window: int):
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

        Xs = []

        N = len(df)

        for end_idx in range(window - 1, N):
            win = df.iloc[end_idx - window + 1 : end_idx + 1]

            Xs.append(win[cols_feat].values.astype(np.float32))

        return np.stack(Xs)

    async def create_ticker_dataset(
        self,
        window_size: int,
        ticker_name: str,
        interval: str,
        stock_btc: pd.DataFrame,
        stock_eth: pd.DataFrame,
        stock_xrp: pd.DataFrame,
        stock_sol: pd.DataFrame,
        max_value: int,
        sym_base_asset: Dict[str, str],
    ) -> Optional[CryptoDataset]:
        """
        Creates a dataset for a specific ticker by fetching its OHLC data and preparing it for training.
        """

        print(f"Creating dataset for {ticker_name} with interval {interval}")

        try:
            tasks = [
                self.fetch_ohlc(sym=ticker_name, intv=interval, bar_needed=max_value),
                self.fetch_ohlc(
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

            df_train = self.prepare_df(
                df_ticker=stock,
                df_usdt=stock_usdt,
                df_btc=stock_btc,
                df_eth=stock_eth,
                df_sol=stock_sol,
                df_xrp=stock_xrp,
            )

            logger.info(
                "Ticker %s has %d training samples",
                ticker_name,
                len(df_train),
            )

            X_tr = self.make_windows(df_train, window_size)

            if X_tr.shape[0] == 0:
                logger.warning(
                    f"No training data available for ticker {ticker_name} after balancing."
                )
                return None

            logger.info(f"Ticker {ticker_name} processed: Train shape: {X_tr.shape}")

            return CryptoDataset(X=X_tr)

        except Exception as e:
            logger.error(
                f"Error processing ticker {ticker_name}, Interval : {interval}: {e}",
                exc_info=True,
            )

            return None

    async def create_dataset(
        self,
        window_size: int,
        ticker_name: str,
        interval: str,
        sym_base_asset: Dict[str, str],
        max_value: int,
    ) -> CryptoDataset | None:
        """
        Constructs the dataset by extracting signals from each ticker's stock data,
        applying derivative calculations if specified, and combining the data into a single array.
        """

        stock_btc = await self.fetch_ohlc(
            sym="BTCUSDT", intv=interval, bar_needed=max_value
        )
        stock_eth = await self.fetch_ohlc(
            sym="ETHUSDT", intv=interval, bar_needed=max_value
        )
        stock_xrp = await self.fetch_ohlc(
            sym="XRPUSDT", intv=interval, bar_needed=max_value
        )
        stock_sol = await self.fetch_ohlc(
            sym="SOLUSDT", intv=interval, bar_needed=max_value
        )

        return await self.create_ticker_dataset(
            window_size=window_size,
            ticker_name=ticker_name,
            interval=interval,
            stock_btc=stock_btc,
            stock_eth=stock_eth,
            stock_xrp=stock_xrp,
            stock_sol=stock_sol,
            max_value=max_value,
            sym_base_asset=sym_base_asset,
        )
