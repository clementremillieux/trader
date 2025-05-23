"""DatasetCreator class to create datasets for machine learning."""

import asyncio

from typing import Dict, Optional

import numpy as np

import pandas as pd

from torch.utils.data import Dataset as TorchDataset

import torch

from app.binance_handler.binance_handler import BinanceHandler

from app.math_func.math_func import (
    SignalDateTime,
    SignalDerivative,
    SignalMACD,
    SignalMomentum,
    SignalRSI,
    SignalStochasticOscillator,
)

from config.logger_config import logger


class SignalDataset(TorchDataset):
    """
    Custom dataset for signal classification.

    Args:
        X (torch.Tensor): Input signals of shape (num_samples, seq_length, num_features).
        y (torch.Tensor): Labels of shape (num_samples,).
    """

    def __init__(self, X: torch.Tensor):
        self.X = X

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


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


class DatasetCreator:
    """DatasetCreator class to create datasets for machine learning."""

    def create_sliding_window(
        self, data: np.ndarray, window_size: int
    ) -> Optional[np.ndarray]:
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

    async def create_dataset(
        self,
        window_size: int,
        ticker_name: str,
        interval: str,
        momentum_period: int,
        rsi_period: int,
        nb_windows: int,
        sym_base_asset: Dict[str, str],
    ) -> Optional[SignalDataset]:
        """
        Constructs the dataset by extracting signals from each ticker's stock data,
        applying derivative calculations if specified, and combining the data into a single array.
        """

        max_value = 400

        binance_handler = BinanceHandler(main_currency="USDT")

        stock_btc = await binance_handler.get_historical_data_v2(
            ticker="BTCUSDT", interval=interval, max_value=max_value
        )

        if not ticker_name.endswith("USDC"):
            ticker_name = ticker_name + "USDC"

        try:
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
                    ticker=ticker_name, interval="1m", max_value=max_value * 16
                ),
                binance_handler.get_historical_data_v2(
                    ticker=ticker_name, interval="1d", max_value=max_value
                ),
            ]

            stock, stock_usdt, stock_1d, stock_1m = await asyncio.gather(*tasks)

            if stock is None:
                logger.warning("Stock data is None for ticker %s", ticker_name)

                return None

            if stock_1d is None:
                logger.warning("Stock 1d data is None for ticker %s", ticker_name)

                return None

            if stock_1m is None:
                logger.warning("Stock 1m data is None for ticker %s", ticker_name)

                return None

            if stock_usdt is None:
                logger.warning("Stock USDT data is None for ticker %s", ticker_name)

                return None

            if stock_btc is None:
                logger.warning("Stock BTC data is None for ticker %s", ticker_name)

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

            signal_data.append(
                np.asarray(stock["QuoteAssetVolume"].values).reshape(-1, 1)
            )

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

            signal_data.append(
                np.asarray(stock_usdt["NumTrades"].values).reshape(-1, 1)
            )

            signal_data.append(
                np.asarray(stock_usdt["TakerBuyBaseVolume"].values).reshape(-1, 1)
            )

            signal_data.append(
                np.asarray(stock_usdt["TakerBuyQuoteVolume"].values).reshape(-1, 1)
            )

            data_np: np.ndarray = np.concatenate(signal_data, axis=1)

            logger.info("DatasetCreator => Data shape: %s", data_np.shape)

            max_initial_na = max(
                momentum_period,
                rsi_period,
            )

            data_np = data_np[max_initial_na:, :]

            windowed_data = self.create_sliding_window(
                data=data_np, window_size=window_size
            )

            logger.info(
                "DatasetCreator => Data shape windowed: %s", windowed_data.shape
            )

            if windowed_data is None:
                return None

            stock_1d = stock_1d.sort_index()

            stock_1d = stock_1d[~stock_1d.index.duplicated(keep="first")]

            stock_1m = stock_1m.sort_index().loc[
                ~stock_1m.index.duplicated(keep="first")
            ]

            if stock_1m.index.tz is None:
                stock_1m.index = stock_1m.index.tz_localize("UTC")

            new_windowed_data = []

            for i, window in enumerate(windowed_data):
                last_val = window[-1, 0]

                if isinstance(last_val, pd.Timestamp):
                    last_ts = (
                        last_val if last_val.tzinfo else last_val.tz_localize("UTC")
                    )

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

            if not new_windowed_data:
                logger.warning("No valid windows after processing.")

                return None

            windowed_data = np.stack(new_windowed_data, axis=0)

            windowed_data = windowed_data[:, :, 1:]

            windowed_data = windowed_data.astype(np.float32)

            x = torch.tensor(windowed_data, dtype=torch.float32)

            dataset = SignalDataset(x[-nb_windows:])

            return dataset

        except Exception as e:
            logger.error(
                "Error processing ticker %s : %s", ticker_name, e, exc_info=True
            )

            return None
