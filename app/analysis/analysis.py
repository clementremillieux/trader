"""Analysis class for managing trading operations."""

from typing import List

import numpy

from app.model.model import Runner

from app.math_func.math_func import Scaler

from app.tickers.schemas import DatasetSignal

from app.dataset.dataset_creator import DatasetCreator

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
