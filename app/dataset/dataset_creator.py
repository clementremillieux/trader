"""DatasetCreator class to create datasets for machine learning."""

from typing import List, Optional

import numpy as np

import pandas as pd

from numpy.typing import NDArray

from torch.utils.data import Dataset as TorchDataset

import torch

from app.tickers.schemas import DatasetSignal

from app.tickers.tickers_handler import TickerHandler

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


class DatasetCreator:
    """DatasetCreator class to create datasets for machine learning."""

    async def get_stock_data(
        self,
        ticker_name: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """
        Downloads stock data for the given tickers using the TickerHandler.
        """

        try:
            ticker = TickerHandler(name=ticker_name)

            await ticker.download_data(interval=interval)

            return ticker.data.stock

        except Exception as _:
            logger.warning(
                "DATASET CREATOR => Error downloading data for %s",
                ticker_name,
            )

        return None

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

    async def get_signals(
        self,
        signals: List[DatasetSignal],
        momentum_period: int,
        rsi_period: int,
        stock: pd.DataFrame,
    ) -> Optional[NDArray]:
        """
        Downloads stock data for the given tickers using the TickerHandler.
        """

        signal_data = []

        signal_data.append(
            SignalDateTime.compute_time_features(
                stock.index,
            ).reshape(-1, 1)
        )

        for signal in signals:
            if signal.column_name in stock.columns:
                signal_data.append(
                    np.asarray(stock[signal.column_name].values).reshape(-1, 1)
                )

                if signal.is_derivative:
                    derivative: NDArray = SignalDerivative.compute_derivative(
                        stock[signal.column_name]
                    )

                    signal_data.append(derivative.reshape(-1, 1))

                if signal.is_financial and signal.column_name.lower() in [
                    "close",
                ]:
                    signal_data.append(
                        SignalMomentum.compute_momentum(
                            stock[signal.column_name].values.reshape(-1, 1),
                            momentum_period,
                        ).reshape(-1, 1)
                    )

                    signal_data.append(
                        SignalRSI.compute_rsi(
                            stock[signal.column_name].values.reshape(-1, 1),
                            rsi_period,
                        ).reshape(-1, 1)
                    )

                    signal_data += SignalMACD.compute_macd(
                        stock[signal.column_name].values.reshape(-1, 1),
                    )

                    signal_data += SignalStochasticOscillator.compute_stoch_oscillator(
                        stock["High"].values.reshape(-1, 1),
                        stock["Low"].values.reshape(-1, 1),
                        stock["Close"].values.reshape(-1, 1),
                    )

        return np.concatenate(signal_data, axis=1)

    async def create_dataset(
        self,
        window_size: int,
        ticker_name: str,
        interval: str,
        signals: List[DatasetSignal],
        momentum_period: int,
        rsi_period: int,
        nb_windows: int,
    ) -> Optional[SignalDataset]:
        """
        Constructs the dataset by extracting signals from each ticker's stock data,
        applying derivative calculations if specified, and combining the data into a single array.
        """

        try:
            stock: Optional[pd.DataFrame] = await self.get_stock_data(
                ticker_name=ticker_name,
                interval=interval,
            )

            if stock is None or stock.shape[0] < window_size:
                logger.warning(
                    "Ticker %s has no data or not enough data for window size %d.",
                    ticker_name,
                    window_size,
                )

                return None

            signal_data: Optional[NDArray] = await self.get_signals(
                signals=signals,
                momentum_period=momentum_period,
                rsi_period=rsi_period,
                stock=stock,
            )

            if signal_data is None:
                logger.warning(
                    "Ticker %s has no signals or not enough data for window size %d.",
                    ticker_name,
                    window_size,
                )

                return None

            ticker_data = self.create_sliding_window(
                data=signal_data, window_size=window_size
            )

            x = torch.tensor(ticker_data, dtype=torch.float32)

            dataset = SignalDataset(x[-nb_windows:])

            return dataset

        except Exception as e:
            logger.error(
                "Error processing ticker %s : %s", ticker_name, e, exc_info=True
            )

            return None
