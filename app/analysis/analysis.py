"""Analysis class for managing trading operations."""

from typing import List, Optional

import numpy

import pandas as pd

import torch

from torch import Tensor

import matplotlib.pyplot as plt

from app.model.model import Runner

from app.math_func.math_func import Scaler

from app.tickers.schemas import DatasetSignal

from app.dataset.dataset_creator import DatasetCreator, SignalDataset

from app.analysis.schemas import AnalysisOutput, AnalysisState

from config.logger_config import logger


class Analysis:
    """Analysis class for managing trading operations."""

    def __init__(
        self,
        model_path: str,
        signals: List[DatasetSignal],
        num_historical_features: int,
        encoder_length: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
        lstm_layers: int = 1,
        n_heads: int = 2,
        num_attention_layers: int = 3,
        pooling_type: str = "attn",
        patch_size: int = 128,
    ) -> None:
        self.dataset_creator = DatasetCreator()

        self.signals = signals

        self.runner = Runner(
            model_path=model_path,
            num_historical_features=num_historical_features,
            encoder_length=encoder_length,
            hidden_size=hidden_size,
            dropout=dropout,
            lstm_layers=lstm_layers,
            n_heads=n_heads,
            num_attention_layers=num_attention_layers,
            pooling_type=pooling_type,
            patch_size=patch_size,
        )

    def fill_nan_with_neighbors(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Fill NaN values in a tensor by forward and backward filling.
        """

        x_filled = x.clone()

        mask = torch.isnan(x_filled)

        for t in range(1, x_filled.size(dim)):
            idx_cur = [slice(None)] * x_filled.dim()

            idx_prev = [slice(None)] * x_filled.dim()

            idx_cur[dim] = t

            idx_prev[dim] = t - 1

            cur = x_filled[tuple(idx_cur)]

            prev = x_filled[tuple(idx_prev)]

            m = mask[tuple(idx_cur)]

            x_filled[tuple(idx_cur)][m] = prev[m]

        for t in range(x_filled.size(dim) - 2, -1, -1):
            idx_cur = [slice(None)] * x_filled.dim()

            idx_next = [slice(None)] * x_filled.dim()

            idx_cur[dim] = t

            idx_next[dim] = t + 1

            cur = x_filled[tuple(idx_cur)]

            nxt = x_filled[tuple(idx_next)]

            m = torch.isnan(cur)

            x_filled[tuple(idx_cur)][m] = nxt[m]

        return x_filled

    async def analyze(
        self,
        ticker: str,
        window_size: int,
        interval: str,
        momentum_period: int,
        rsi_period: int,
        nb_windows: int,
        name: str,
    ) -> AnalysisOutput:
        """Analyse le ticker et renvoie BUY, SELL ou HOLD."""

        dataset = await self.dataset_creator.create_dataset(
            window_size=window_size,
            ticker_name=ticker,
            interval=interval,
            signals=self.signals,
            momentum_period=momentum_period,
            rsi_period=rsi_period,
            nb_windows=nb_windows,
        )

        if dataset is None:
            logger.warning(
                "ANALYZE => (%s) Dataset is None for ticker: %s", name, ticker
            )

            return AnalysisOutput(state=AnalysisState.HOLD)

        nan_mask = torch.isnan(dataset.X)

        num_nans = nan_mask.sum().item()

        logger.info("ANALYZE => There is %d NaNs in the dataset", num_nans)

        if num_nans > 30:
            logger.warning(
                "ANALYZE => (%s) Too many NaNs in the dataset for ticker: %s",
                name,
                ticker,
            )

            return AnalysisOutput(state=AnalysisState.HOLD)

        coords = nan_mask.nonzero(as_tuple=True)

        dim1_idx = coords[1]

        if bool((dim1_idx >= 10).any()):
            logger.warning(
                "ANALYZE => (%s) NaNs are too far in the dataset for ticker: %s",
                name,
                ticker,
            )

            return AnalysisOutput(state=AnalysisState.HOLD)

        dataset.X = self.fill_nan_with_neighbors(dataset.X)

        nan_mask = torch.isnan(dataset.X)

        num_nans = nan_mask.sum().item()

        logger.info(
            "ANALYZE => There is %d NaNs in the dataset after filling ", num_nans
        )

        if num_nans > 0:
            logger.warning(
                "ANALYZE => (%s) There are still NaNs in the dataset for ticker: %s",
                name,
                ticker,
            )

            return AnalysisOutput(state=AnalysisState.HOLD)

        X = Scaler().scale(X=dataset.X)

        logits = self.runner.run(data=X)

        probs = logits.cpu().detach().numpy()

        p0 = probs[0]

        diff2_1 = p0[2] - p0[1]

        diff2_0 = p0[2] - p0[0]

        diff0_1 = p0[0] - p0[1]

        diff0_2 = p0[1] - p0[2]

        arg = p0.argmax()

        logger.info("ANALYZE => (%s) [%s] probs = %s", name, ticker, p0)

        logger.info(
            "ANALYZE =>\t- (%s) [%s] diff2_1 = %.4f (threshold=%s)",
            name,
            ticker,
            diff2_1,
            2,
        )

        logger.info(
            "ANALYZE =>\t- (%s) [%s] diff2_0  = %.4f (threshold=%s)",
            name,
            ticker,
            diff2_0,
            2,
        )

        logger.info(
            "ANALYZE =>\t- (%s) [%s] Argmax = %d",
            name,
            ticker,
            arg,
        )

        if arg == 2 and diff2_1 > 2 and diff2_0 > 2 and p0[2] > 2:
            logger.info("ANALYZE =>\t- (%s) [%s] ANALYZE RESULT : BUY", name, ticker)

            return AnalysisOutput(state=AnalysisState.BUY)

        if arg == 0 and diff0_1 > 2 and diff0_2 > 2 and p0[0] > 1:
            logger.info("ANALYZE =>\t- (%s) [%s] ANALYZE RESULT : SELL", name, ticker)

            return AnalysisOutput(state=AnalysisState.SELL)

        logger.info("ANALYZE =>\t- (%s) [%s] ANALYZE RESULT : HOLD", name, ticker)

        return AnalysisOutput(state=AnalysisState.HOLD)

    async def plot_last_windows(
        self,
        ticker: str,
        window_size: int,
        interval: str,
        days: str,
        momentum_period: int,
        rsi_period: int,
        nb_windows: int,
        name: str = "plot",
    ) -> None:
        """
        Infère sur les 'nb_windows' dernières fenêtres pour le ticker,
        puis trace :
         - en bleu   : le prix de clôture ('Close')
         - en rouge  : P(classe=0)
         - en vert   : P(classe=1)
         - en orange : P(classe=2)
        """

        # 1. Création du dataset
        dataset: Optional[SignalDataset] = await self.dataset_creator.create_dataset(
            window_size=window_size,
            ticker_name=ticker,
            interval=interval,
            signals=self.signals,
            momentum_period=momentum_period,
            rsi_period=rsi_period,
            nb_windows=nb_windows,
        )

        if dataset is None:
            logger.error("PLOT => (%s) Dataset None for %s", name, ticker)

            return

        X_scaled: Tensor = Scaler().scale(X=dataset.X)

        logits: Tensor = self.runner.run(data=X_scaled)

        probs = logits.cpu().detach().numpy()

        stock_data: Optional[pd.DataFrame] = await self.dataset_creator.get_stock_data(
            ticker_name=ticker,
            interval=interval,
        )

        if stock_data is None:
            logger.error("PLOT => (%s) Close signal data None for %s", name, ticker)

            return

        close_signal = stock_data["Close"].values[-nb_windows:]

        x = numpy.arange(nb_windows)

        fig, ax1 = plt.subplots()

        ax1.plot(x, close_signal, label="Close", linewidth=2, color="red")

        ax1.set_xlabel("Index de la fenêtre")

        ax1.set_ylabel("Prix Close")

        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2 = ax1.twinx()

        ax2.plot(x, probs[:, 0], label="P(0)", linestyle="-.")
        ax2.plot(x, probs[:, 1], label="P(1)", linestyle="--")
        ax2.plot(x, probs[:, 2], label="P(2)", linestyle=":")
        ax2.set_ylabel("Probabilité")

        # Légende commune
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

        plt.title(
            f"{ticker} – Close vs Probabilités (dern. {nb_windows} fenêtres) [du {stock_data.index[-nb_windows]} au {stock_data.index[-1]}"
        )
        plt.tight_layout()
        plt.show()
