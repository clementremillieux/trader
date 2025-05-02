"""Analysis class for managing trading operations."""

from os import close
from typing import List, Optional

import numpy

import matplotlib.pyplot as plt
import pandas as pd
from torch import Tensor

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

    async def analyze(
        self,
        ticker: str,
        window_size: int,
        interval: str,
        days: str,
        momentum_period: int,
        rsi_period: int,
        nb_windows: int,
        nb_2: int,
        nb_last_2: int,
        distance_0: float,
        distance_1: float,
        name: str,
    ) -> AnalysisOutput:
        """Analyse le ticker et renvoie BUY, SELL ou HOLD."""

        dataset = await self.dataset_creator.create_dataset(
            window_size=window_size,
            ticker_name=ticker,
            interval=interval,
            days=days,
            signals=self.signals,
            momentum_period=momentum_period,
            rsi_period=rsi_period,
            nb_windows=nb_windows,
        )

        if dataset is None:
            logger.error("ANALYZE => (%s) Dataset is None for ticker: %s", name, ticker)

            return AnalysisOutput(state=AnalysisState.HOLD)

        X = Scaler().scale(X=dataset.X)

        logits = self.runner.run(data=X)

        probs = logits.cpu().detach().numpy()

        preds = numpy.argmax(probs, axis=1)

        diff2_1 = probs[:, 2] - probs[:, 1]

        diff2_0 = probs[:, 2] - probs[:, 0]

        strong2 = (diff2_1 > distance_1) & (diff2_0 > distance_0)

        count_strong2 = int(numpy.sum(strong2))

        last_all_strong2 = (strong2.size >= nb_last_2) and numpy.all(
            strong2[-nb_last_2:]
        )

        if count_strong2 > nb_2 and last_all_strong2:
            logger.info("ANALYZE => (%s) ANALYZE RESULT [%s]: BUY", name, ticker)
            return AnalysisOutput(state=AnalysisState.BUY)

        if preds.size >= 5 and numpy.all(preds[-5:] == 0):
            logger.info("ANALYZE => (%s) ANALYZE RESULT [%s]: SELL", name, ticker)

            return AnalysisOutput(state=AnalysisState.SELL)

        logger.info("ANALYZE => (%s) ANALYZE RESULT [%s]: HOLD", name, ticker)

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
            days=days,
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
            days=days,
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
