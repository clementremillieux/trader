"""Analysis class for managing trading operations."""

from typing import List, Dict

import torch

import numpy as np

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
        sell_model_path: str,
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

        self.sell_runner = Runner(
            model_path=sell_model_path,
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
        sym_base_asset: Dict[str, str],
        size_watch_buy: int = 1,
        size_watch_sell: int = 2,
    ) -> AnalysisOutput:
        """Analyse le ticker et renvoie BUY, SELL ou HOLD."""

        dataset = await self.dataset_creator.create_dataset(
            window_size=window_size,
            ticker_name=ticker,
            interval=interval,
            momentum_period=momentum_period,
            rsi_period=rsi_period,
            nb_windows=nb_windows,
            sym_base_asset=sym_base_asset,
        )

        if dataset is None:
            logger.warning(
                "ANALYZE => (%s) Dataset is None for ticker: %s", name, ticker
            )

            return AnalysisOutput(state=AnalysisState.HOLD)

        X = Scaler().scale(X=dataset.X)

        logits_buy: torch.Tensor = self.runner.run(data=X)

        probs_buy: np.ndarray = logits_buy.cpu().detach().numpy()

        logits_sell: torch.Tensor = self.sell_runner.run(data=X)

        probs_sell: np.ndarray = logits_sell.cpu().detach().numpy()

        def _is_buy(p: np.ndarray) -> bool:
            """Conditions BUY pour une ligne de proba (index 0: SELL, 1: HOLD, 2: BUY)."""

            logger.info("ANALYZE =>\t- (%s) [%s] PROBS BUY : %s", name, ticker, p)

            diff2_1 = p[2] - p[1]

            diff2_0 = p[2] - p[0]

            arg = p.argmax()

            return arg == 2 and diff2_1 > 4 and diff2_0 > 4 and p[2] > 4

        def _is_sell(p: np.ndarray) -> bool:
            """Conditions SELL pour une ligne de proba."""

            logger.info(
                "ANALYZE =>\t- (%s) [%s] PROBS SELL : %s",
                name,
                ticker,
                p,
            )

            diff2_1 = p[2] - p[1]

            diff2_0 = p[2] - p[0]

            arg = p.argmax()

            return arg == 2 and diff2_1 > 4 and diff2_0 > 4 and p[2] > 4

        watch_buy = (
            probs_buy[-size_watch_buy:]
            if size_watch_buy <= len(probs_buy)
            else probs_buy
        )

        watch_sell = (
            probs_sell[-size_watch_sell:]
            if size_watch_sell <= len(probs_sell)
            else probs_sell
        )

        if any(_is_sell(p) for p in watch_sell):
            logger.info("ANALYZE =>\t- (%s) [%s] ANALYZE RESULT : SELL", name, ticker)

            return AnalysisOutput(state=AnalysisState.SELL)

        if all(_is_buy(p) for p in watch_buy):
            logger.info("ANALYZE =>\t- (%s) [%s] ANALYZE RESULT : BUY", name, ticker)

            return AnalysisOutput(state=AnalysisState.BUY)

        logger.info("ANALYZE =>\t- (%s) [%s] ANALYZE RESULT : HOLD", name, ticker)

        return AnalysisOutput(state=AnalysisState.HOLD)
