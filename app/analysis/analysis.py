"""Analysis class for managing trading operations."""

from typing import Dict

import torch

import numpy as np

from app.model.model import Runner

from app.math_func.math_func import Scaler

from app.dataset.dataset_creator import DatasetCreator

from app.analysis.schemas import AnalysisOutput, AnalysisState

from config.logger_config import logger


class Analysis:
    """Analysis class for managing trading operations."""

    def __init__(
        self,
        model_path: str,
    ) -> None:
        self.dataset_creator = DatasetCreator()

        self.runner = Runner(
            model_path=model_path,
        )

    async def analyze(
        self,
        ticker: str,
        window_size: int,
        name: str,
        sym_base_asset: Dict[str, str],
    ) -> AnalysisOutput:
        """Analyse le ticker et renvoie BUY, SELL ou HOLD."""

        if not ticker.endswith("USDC"):
            ticker = ticker + "USDC"

        dataset = await self.dataset_creator.create_dataset(
            window_size=window_size,
            ticker_name=ticker,
            sym_base_asset=sym_base_asset,
            interval="1h",
            max_value=1000,
        )

        if dataset is None:
            logger.warning(
                "ANALYZE => (%s) Dataset is None for ticker: %s", name, ticker
            )

            return AnalysisOutput(state=AnalysisState.HOLD)

        X = Scaler().scale(X=dataset.X)

        logits_buy: torch.Tensor = self.runner.run(data=X)

        probs_buy: np.ndarray = logits_buy.cpu().detach().numpy()

        def _is_buy(p: np.ndarray) -> bool:
            """Conditions BUY pour une ligne de proba (index 0: SELL, 1: HOLD, 2: BUY)."""

            logger.info(
                "ANALYZE =>\t- (%s) [%s] PROBS BUY : %s ; DIFF 0: %s",
                name,
                ticker,
                p,
                p[2] - p[0],
            )

            return p[2] - p[0] > 5

        def _is_sell(p: np.ndarray) -> bool:
            """Conditions SELL pour une ligne de proba."""

            logger.info(
                "ANALYZE =>\t- (%s) [%s] PROBS SELL : %s; DIFF 0: %s",
                name,
                ticker,
                p,
                p[2] - p[0],
            )

            return p[0] - p[2] > 0

        watch_buy = probs_buy[-1:]

        if all(_is_buy(p) for p in watch_buy):
            logger.info("ANALYZE =>\t- (%s) [%s] ANALYZE RESULT : BUY", name, ticker)

            return AnalysisOutput(state=AnalysisState.BUY)

        if all(_is_sell(p) for p in watch_buy):
            logger.info("ANALYZE =>\t- (%s) [%s] ANALYZE RESULT : SELL", name, ticker)

            return AnalysisOutput(state=AnalysisState.SELL)

        logger.info("ANALYZE =>\t- (%s) [%s] ANALYZE RESULT : HOLD", name, ticker)

        return AnalysisOutput(state=AnalysisState.HOLD)
