"""Cryto tester"""

from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from app.analysis.schemas import AnalysisOutput
from app.binance_handler.binance_handler import BinanceHandler
from app.dataset.dataset_creator import DatasetCreator
from app.math_func.math_func import Scaler
from app.model.model import Runner


from config.logger_config import logger


class Tester:
    """Tester class for managing trading operations."""

    def __init__(
        self,
    ) -> None:
        self.window_size = 300

        self.interval = "1h"

        self.momentum_period = 5

        self.rsi_period = 5

        self.model_path = "app/model/model_epoch_crypto_10.pth"

        self.sell_model_path = "app/model/model_epoch_crypto_sell_97.pth"

        self.num_historical_features = 40

        self.encoder_length = self.window_size

        self.hidden_size = 1024

        self.dropout = 0.6

        self.lstm_layers = 4

        self.n_heads = 8

        self.num_attention_layers = 8

        self.patch_size = int(self.window_size / 8)

        self.pooling_type = "attn"

        self.runner = Runner(
            model_path=self.model_path,
            num_historical_features=self.num_historical_features,
            encoder_length=self.encoder_length,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            lstm_layers=self.lstm_layers,
            n_heads=self.n_heads,
            num_attention_layers=self.num_attention_layers,
            pooling_type=self.pooling_type,
            patch_size=self.patch_size,
        )

        self.sell_runner = Runner(
            model_path=self.sell_model_path,
            num_historical_features=self.num_historical_features,
            encoder_length=self.encoder_length,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            lstm_layers=self.lstm_layers,
            n_heads=self.n_heads,
            num_attention_layers=self.num_attention_layers,
            pooling_type=self.pooling_type,
            patch_size=self.patch_size,
        )

        self.dataset_creator = DatasetCreator()

        self.binance_handler = BinanceHandler(main_currency="USDT")

        self.symbols_dict: Dict[str, str] = self.binance_handler.get_symbols_dict()

    async def test(self, name: str, ticker: str, nb_windows: int) -> None:
        """Test the model with the given ticker."""

        dataset = await self.dataset_creator.create_dataset(
            window_size=self.window_size,
            ticker_name=ticker,
            interval=self.interval,
            momentum_period=self.momentum_period,
            rsi_period=self.rsi_period,
            nb_windows=nb_windows,
            sym_base_asset=self.symbols_dict,
            max_value=nb_windows,
        )

        if dataset is None:
            logger.warning(
                "ANALYZE => (%s) Dataset is None for ticker: %s", name, ticker
            )

            return

        X = Scaler().scale(X=dataset.X)

        logger.info(
            "ANALYZE =>\t- (%s) [%s] DATASET SHAPE: %s, MIN: %s, MAX: %s",
            name,
            ticker,
            dataset.X.shape,
            dataset.X.min(),
            dataset.X.max(),
        )

        logits_buy: torch.Tensor = self.runner.run(data=X)

        probs_buy: np.ndarray = logits_buy.cpu().detach().numpy()

        logits_sell: torch.Tensor = self.sell_runner.run(data=X)

        probs_sell: np.ndarray = logits_sell.cpu().detach().numpy()

        max_value = 1000

        stock: pd.DataFrame = await self.binance_handler.get_historical_data_v2(
            ticker=f"{self.symbols_dict[ticker]}USDC",
            interval=self.interval,
            max_value=max_value,
        )

        if stock is None or stock.empty:
            logger.warning(
                "ANALYZE => (%s) Stock dataframe is None/empty for ticker: %s",
                name,
                ticker,
            )
            return

        close_prices = stock["Close"].values.astype(float)

        # ------------------------------------------------------------------ #
        # 1) Alignement des longueurs                                        #
        # ------------------------------------------------------------------ #
        pred_len = probs_buy.shape[0]  # == nb_windows que vous avez demandé

        # 2) On prend les derniers pred_len cours de clôture
        close_prices = close_prices[-pred_len:]

        # 3) Même chose pour les tableaux de probas
        probs_buy = probs_buy[-pred_len:]
        probs_sell = probs_sell[-pred_len:]

        # (facultatif) Sécurité :
        assert len(close_prices) == len(probs_buy) == len(probs_sell), (
            "Décalage données ↔ prédictions"
        )

        # ------------------------------------------------------------------ #
        # 2) Fonctions utilitaires                                           #
        # ------------------------------------------------------------------ #
        def scale_minus_one_one(arr: np.ndarray) -> np.ndarray:
            """Ramène arr dans l’intervalle [-1 ; 1] en préservant le ratio des amplitudes."""
            max_abs = np.abs(arr).max()
            return arr / max_abs if max_abs else arr

        # ------------------------------------------------------------------ #
        # 3) Signaux et probabilités                                         #
        # ------------------------------------------------------------------ #
        d_margin_buy = 5  # ge pour le signal d’achat

        min_prob_buy = 1

        d_margin_sell = 5  # ge pour le signal de vente

        min_prob_sell = 1

        sell_prob_scaled = (
            MinMaxScaler(feature_range=(-1, 1))
            .fit_transform(probs_sell[:, 2].reshape(-1, 1))
            .flatten()
        )

        buy_signal = (
            ((probs_buy[:, 2] - probs_buy[:, 0]) > d_margin_buy)  # > classe 0
            & ((probs_buy[:, 2] - probs_buy[:, 1]) > d_margin_buy)  # > classe 1
            & (probs_buy[:, 2] > min_prob_buy)
            & (probs_buy[:, 0] < 0)  # positif
        ).astype(int)

        sell_signal = (
            (
                (probs_buy[:, 0] - probs_buy[:, 2]) > d_margin_sell
            )  # marge 0.1 (ou d_margin)
            & ((probs_buy[:, 0] - probs_buy[:, 1]) > d_margin_sell)  # > classe 1
            & (probs_buy[:, 0] > min_prob_sell)  # positif
        ).astype(int)

        buy_prob_scaled = scale_minus_one_one(
            probs_buy[:, 2]
        )  # classe 2 mise à l’échelle

        sell_prob_scaled = scale_minus_one_one(
            probs_buy[:, 0]
        )  # classe 0 mise à l’échelle

        # Prix de clôture normalisé pour qu’il “tienne” visuellement avec les probas
        close_scaled = (
            MinMaxScaler(feature_range=(-1, 1))
            .fit_transform(close_prices.reshape(-1, 1))
            .flatten()
        )

        # ------------------------------------------------------------------ #
        # 4) Tracé                                                          #
        # ------------------------------------------------------------------ #
        plt.figure(figsize=(14, 6))

        # Prix de clôture
        plt.plot(
            close_scaled, label=f"{ticker} Close (norm.)", linewidth=1.5, color="black"
        )

        # BUY : signal plein + proba pointillée
        plt.plot(
            buy_signal, label="Buy signal", linewidth=1, linestyle="-", color="green"
        )

        plt.plot(
            buy_prob_scaled,
            label="Buy prob. (scaled)",
            linewidth=1,
            linestyle="--",
            color="green",
        )

        # SELL : signal plein + proba pointillée
        plt.plot(
            sell_signal, label="Sell signal", linewidth=1, linestyle="-", color="red"
        )

        plt.plot(
            sell_prob_scaled,
            label="Sell prob. (scaled)",
            linewidth=1,
            linestyle="--",
            color="orange",
        )

        plt.title(f"{ticker} – Probabilités vs. cours de clôture ({self.interval})")
        plt.xlabel("Pas de temps (bougies)")
        plt.ylabel("Valeur normalisée")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=":")
        plt.tight_layout()


async def main() -> None:
    tester = Tester()

    tickers: List[str] = list(
        set((tester.binance_handler.get_tickers_existing_both_main_currency_usdt()))
    )[:5]

    for ticker in tickers:
        await tester.test(name="Test", ticker=ticker, nb_windows=1000)

    plt.show()

    # If you added an aclose() to BinanceHandler:
    # await tester.binance_handler.aclose()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
