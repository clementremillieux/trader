# tester.py  ────────────────────────────────────────────────────────────
"""Crypto tester – ultra-clean, factorised version.

Fonctions clés
==============
* Tester.test(...)  →  renvoie (min_var, max_var, avg_var) par ticker
* StatsCollector    →  agrège les métriques sur tous les tickers
* _compute_variations()  →  calcule les Δ% buy→sell (plancher −5 %)

Usage
=====
python -m app.tester.tester
"""

from __future__ import annotations

import random

from typing import Dict, List, Tuple


import numpy as np

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from app.binance_handler.binance_handler import BinanceHandler
from app.dataset.dataset_creator import DatasetCreator
from app.math_func.math_func import Scaler
from app.model.model import Runner
from config.logger_config import logger


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────
def scale_minmax(arr: np.ndarray, feature_range: Tuple[int, int]) -> np.ndarray:
    """Min-max scaling wrapper."""
    return (
        MinMaxScaler(feature_range=feature_range)
        .fit_transform(arr.reshape(-1, 1))
        .ravel()
    )


def scale_minus_one_one(arr: np.ndarray) -> np.ndarray:
    """Scale to [-1, 1] by max abs value (keeps ratios)."""
    max_abs = np.abs(arr).max()
    return arr / max_abs if max_abs else arr


# ───────── helpers.py ─────────
def compound_return(vars_: np.ndarray) -> float:  # déjà présent
    return (np.prod(1 + vars_ / 100) - 1) * 100 if vars_.size else 0.0


def flat_pnl(vars_: np.ndarray, capital: float, n_slices: int) -> float:
    """P&L absolu, sans ré-investissement (stake = capital/N_slices)."""
    if vars_.size == 0:
        return 0.0
    stake = capital / n_slices
    return stake * vars_.sum() / 100  # somme simple sur les %


def compute_variations(
    buy_idx: np.ndarray,
    sell_idx: np.ndarray,
    prices: np.ndarray,
    tester: Tester,
    floor: float = -5,
) -> Tuple[np.ndarray, int, int]:
    """
    Δ% buy → sell avec stop-loss.

    ▸ Appariement séquentiel "premier vert" → "premier rouge suivant".
    ▸ Si, AVANT le rouge, la perte atteint <= floor %, on déclenche
      un stop-loss et on enregistre exactement `floor`.

    Args
    ----
    buy_idx  : indices des croix vertes (buy).
    sell_idx : indices des croix rouges (sell).
    prices   : série 1-D des cours de clôture.
    floor    : stop-loss (négatif), ex. -5 → coupe la position à -5 %.

    Returns
    -------
    np.ndarray des variations (en %) avec stop-loss appliqué.
    """
    buy_sorted = np.sort(buy_idx)
    sell_sorted = np.sort(sell_idx)

    i = j = 0
    variations: List[float] = []

    nb_ticker_success = 0

    nb_ticker_failed = 0

    while i < len(buy_sorted) and j < len(sell_sorted):
        # avance jusqu’au 1er rouge STRICTEMENT après ce vert
        while j < len(sell_sorted) and sell_sorted[j] <= buy_sorted[i]:
            j += 1
        if j == len(sell_sorted):  # plus de rouge disponible
            break

        b = buy_sorted[i]
        s = sell_sorted[j]
        price_b = prices[b]

        # -------- stop-loss : calcule la pire variation atteinte avant 's' --------
        window = prices[b + 1 : s + 1]  # toutes les bougies intermédiaires
        if window.size:  # (par sécurité)
            min_var = ((window.min() - price_b) / price_b) * 100.0
            if min_var <= floor:  # stop-loss déclenché
                variations.append(floor)
                i += 1
                j += 1
                tester.failed_trades += 1

                nb_ticker_failed += 1

                continue

        # -------- pas de stop-loss : variation normale (clampée à floor) --------
        var = ((prices[s] - price_b) / price_b) * 100.0
        variations.append(max(var, floor))

        if var >= 0:
            tester.succes_trades += 1

            nb_ticker_success += 1

        else:
            tester.failed_trades += 1

            nb_ticker_failed += 1

        i += 1
        j += 1

    return np.asarray(variations, dtype=float), nb_ticker_success, nb_ticker_failed


# ───────── stats_collector.py ─────────
class StatsCollector:
    """Agrège résultats + variations brutes + nb. de trades."""

    def __init__(self, capital: float, n_slices: int) -> None:
        self.capital0 = capital
        self.n_slices = n_slices
        self._rows: list[tuple[str, float, float, float, int]] = []
        self._all_vars: list[float] = []

    def add(self, ticker: str, vars_: np.ndarray) -> None:
        """Enregistre métriques + variations d’un ticker."""
        if vars_.size == 0:
            return
        self._all_vars.extend(vars_.tolist())
        self._rows.append((ticker, vars_.min(), vars_.max(), vars_.mean(), len(vars_)))

    # ---- propriétés ----
    @property
    def rows(self) -> list[tuple[str, float, float, float, int]]:
        return self._rows

    @property
    def total_trades(self) -> int:
        return sum(row[4] for row in self._rows)

    # ---- métriques globales ----
    def global_mean(self) -> tuple[float, float, float]:
        if not self._rows:
            return 0.0, 0.0, 0.0
        arr = np.asarray(self._rows)[:, 1:4].astype(float)  # min/max/avg
        return tuple(arr.mean(axis=0))

    def global_compound(self) -> float:
        return compound_return(np.asarray(self._all_vars))

    def flat_final_capital(self) -> float:
        pnl = flat_pnl(np.asarray(self._all_vars), self.capital0, self.n_slices)
        return self.capital0 + pnl


def filter_short_runs(signal: np.ndarray, min_run: int) -> np.ndarray:
    """
    Remplace par 0 toute séquence de 1 dont la longueur est < min_run.
    Parcours de gauche à droite.

    Parameters
    ----------
    signal  : array 1-D de 0/1 (dtype int ou bool)
    min_run : longueur minimale pour garder la séquence

    Returns
    -------
    array du même type/shape que signal
    """
    sig = signal.copy()
    n = len(sig)
    i = 0

    while i < n:
        if sig[i] == 1:
            # début d'un run
            start = i
            while i < n and sig[i] == 1:
                i += 1
            run_len = i - start
            if run_len < min_run:
                sig[start:i] = 0  # supprime le run trop court
        else:
            i += 1
    return sig


# ────────────────────────────────────────────────────────────────────────
# Tester
# ────────────────────────────────────────────────────────────────────────
class Tester:
    """Tester class for managing trading operations."""

    # ----- Hyper-params constant for all tickers ----- #
    window_size = 300

    interval = "1h"

    model_path = "app/model/model_epoch_crypto_sell_87.pth"

    success: int = 0

    failed: int = 0

    succes_trades: int = 0

    failed_trades: int = 0

    # ----- Init ----- #
    def __init__(self) -> None:
        self.encoder_length = self.window_size
        self.runner = Runner(
            model_path=self.model_path,
        )

        self.dataset_creator = DatasetCreator()
        self.binance_handler = BinanceHandler(main_currency="USDC")
        self.symbols_dict: Dict[str, str] = self.binance_handler.get_symbols_dict()

    # ----- Core async test ----- #
    async def test(self, name: str, ticker: str, nb_windows: int) -> np.ndarray | None:
        """Run inference, plot, and return (min, max, mean) Δ% for *ticker*."""

        logger.info(
            "=== %s | %s | start ==================================", name, ticker
        )

        # 1) Dataset
        dataset = await self.dataset_creator.create_dataset(
            window_size=self.window_size,
            ticker_name=ticker,
            interval=self.interval,
            sym_base_asset=self.symbols_dict,
            max_value=nb_windows,
        )

        if dataset is None:
            logger.warning("ANALYZE => (%s) Dataset is None for %s", name, ticker)
            return None

        X = Scaler().scale(dataset.X)

        logger.info("ANALYZE => (%s) Max X : %s, Min X : %s", name, X.max(), X.min())

        logits = self.runner.run(X)

        probs_buy = logits.softmax(dim=1).cpu().numpy()

        stock = await self.binance_handler.get_historical_data_v2(
            ticker=f"{self.symbols_dict[ticker]}USDC",
            interval=self.interval,
            max_value=nb_windows,
        )

        if stock is None or stock.empty:
            logger.warning("ANALYZE => (%s) Empty prices for %s", name, ticker)

            return None

        close_prices = stock["Close"].astype(float).values[-probs_buy.shape[0] :]

        delta_buy = 0.20

        delta_sell = 0.20

        buy_signal = (probs_buy[:, 2] - probs_buy[:, 0] > delta_buy).astype(int)

        sell_signal = (probs_buy[:, 0] - probs_buy[:, 2] > delta_sell).astype(int)

        MIN_RUN = 1

        MIN_SELL = 0

        buy_signal = filter_short_runs(buy_signal, MIN_RUN)

        sell_signal = filter_short_runs(sell_signal, MIN_SELL)

        buy_cross = np.where(np.diff(buy_signal, prepend=0) == 1)[0]

        sell_cross = np.where(np.diff(sell_signal, prepend=0) == 1)[0]

        # 5) Δ% buy→sell
        variations, nb_ticker_success, nb_ticker_failed = compute_variations(
            buy_cross,
            sell_cross,
            close_prices,
            self,
        )

        scale_value = np.max(probs_buy)

        if variations.size == 0:  # ← couper AVANT min/avg/max
            logger.info("► %s | aucun trade clôturé", ticker)
            logger.info(
                "=== %s | %s | end ====================================", name, ticker
            )

            self._plot(
                ticker=ticker,
                close_scaled=scale_minmax(close_prices, (-1, 1)) * scale_value,
                buy_signal=buy_signal * scale_value,
                sell_signal=sell_signal * scale_value,
                buy_prob=probs_buy[:, 2],
                hold_prob=probs_buy[:, 1],
                sell_prob=probs_buy[:, 0],
                buy_cross=buy_cross,
                sell_cross=sell_cross,
                stats=(0, 0, 0, variations.size, 0),
            )

            return variations

        min_v, max_v, avg_v = variations.min(), variations.max(), variations.mean()

        comp_v = compound_return(variations)

        if comp_v < 0:
            self.failed += 1

        else:
            self.success += 1

        logger.info(
            "► %s | trades=%d [succes=%d; failed=%d] | Δ.min=%.2f; Δ.avg=%.2f; Δ.max=%.2f; Δ.comp=%.2f; min price=%.4f; max price=%.4f",
            ticker,
            variations.size,
            nb_ticker_success,
            nb_ticker_failed,
            min_v,
            avg_v,
            max_v,
            comp_v,
            close_prices.min(),
            close_prices.max(),
        )

        # 6) Plot
        self._plot(
            ticker=ticker,
            close_scaled=scale_minmax(close_prices, (-1, 1)) * scale_value,
            buy_signal=buy_signal * scale_value,
            sell_signal=sell_signal * scale_value,
            buy_prob=probs_buy[:, 2],
            hold_prob=probs_buy[:, 1],
            sell_prob=probs_buy[:, 0],
            buy_cross=buy_cross,
            sell_cross=sell_cross,
            stats=(min_v, max_v, avg_v, variations.size, comp_v),
        )

        return variations

    # ----- Internal plotting helper ----- #
    @staticmethod
    def _plot(
        *,
        ticker: str,
        close_scaled: np.ndarray,
        buy_signal: np.ndarray,
        sell_signal: np.ndarray,
        buy_prob: np.ndarray,
        hold_prob: np.ndarray,
        sell_prob: np.ndarray,
        buy_cross: np.ndarray,
        sell_cross: np.ndarray,
        stats: Tuple[float, float, float, int, float],
    ) -> None:
        min_v, max_v, avg_v, nb_trade, comp_v = stats
        plt.figure(figsize=(14, 6))

        # prix
        plt.plot(close_scaled, label=f"{ticker} Close (norm.)", color="black")

        # signaux + probas
        plt.plot(buy_signal, label="Buy signal", color="green")
        plt.plot(buy_prob, label="Buy prob.", linestyle="--", color="green")
        plt.plot(hold_prob, label="Hold prob.", linestyle="--", color="blue")
        plt.plot(sell_signal, label="Sell signal", color="red")
        plt.plot(sell_prob, label="Sell prob.", linestyle="--", color="orange")

        # croix
        plt.scatter(
            buy_cross,
            close_scaled[buy_cross],
            marker="x",
            s=80,
            color="green",
            label="Buy cross",
        )
        plt.scatter(
            sell_cross,
            close_scaled[sell_cross],
            marker="x",
            s=80,
            color="red",
            label="Sell cross",
        )

        # stats in title
        plt.title(
            f"{ticker} – {nb_trade} trades : total={comp_v:.2f}%, min={min_v:.2f}%, avg={avg_v:.2f}%, max={max_v:.2f}% – {Tester.interval}"
        )
        plt.xlabel("Timesteps")
        plt.ylabel("Scaled value")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=":")
        plt.tight_layout()


# ────────────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────────────
# ───────── main() mise à jour ─────────
async def main() -> None:
    INITIAL_CAPITAL = 1_000

    N_SLICES = 20

    NB_HOURS = int(24 * 30.5)

    tester = Tester()

    stats = StatsCollector(INITIAL_CAPITAL, N_SLICES)

    tickers = list(
        tester.binance_handler.get_tickers_existing_both_main_currency_usdt()
    )

    k = len(tickers)

    tickers = random.sample(tickers, k=50)  # pour limiter le nombre de tickers

    for t in tickers:
        vars_ = await tester.test(name="Test", ticker=t, nb_windows=NB_HOURS)

        if vars_ is not None:
            stats.add(t, vars_)

        # — résumé global après chaque ticker —
        gmin, gmax, gavg = stats.global_mean()
        gtot_compounded = stats.global_compound()
        final_capital = stats.flat_final_capital()
        flat_return_pct = (final_capital / INITIAL_CAPITAL - 1) * 100
        total_trades = stats.total_trades

        cap_str = f"{final_capital:,.2f}"
        ratio = (
            tester.success / (tester.success + tester.failed)
            if (tester.success + tester.failed)
            else 0.0
        )

        ratio_trades = (
            tester.succes_trades / (tester.succes_trades + tester.failed_trades)
            if (tester.succes_trades + tester.failed_trades)
            else 0.0
        )

        logger.info(
            "\n>>> GLOBAL after %d tickers (%d trades, stake = X/N)"
            "\n    Δ.min(avg tick)    : %.2f %%"
            "\n    Δ.max(avg tick)    : %.2f %%"
            "\n    Δ.moyen tick       : %.2f %%"
            "\n    Δ.composé global   : %.2f %%"
            "\n    Capital final      : %s  (%.2f %%)"
            "\n    Ratio tickers      : %.2f (%d/%d)"
            "\n    Ratio trades       : %.2f (%d/%d)\n",
            len(stats.rows),  # %d
            total_trades,  # %d
            gmin,
            gmax,
            gavg,  # %.2f
            gtot_compounded,  # %.2f
            cap_str,  # %s
            flat_return_pct,  # %.2f
            ratio,  # %.2f
            tester.success,  # %d
            tester.failed,  # %d
            ratio_trades,  # %.2f
            tester.succes_trades,  # %d
            tester.failed_trades,  # %d
        )

    all_vars = np.asarray(stats._all_vars, dtype=float)

    if all_vars.size:  # sécurité
        # --- histogramme des gains/pertes ----------------------------------
        plt.figure(figsize=(10, 5))  # 1 plot → 1 figure
        plt.hist(all_vars, bins=40)
        plt.title("Répartition des variations par trade")
        plt.xlabel("Δ % par trade")
        plt.ylabel("Nombre de trades")
        plt.grid(True, linestyle=":")

        # --- courbe du capital cumulé --------------------------------------
        stake = INITIAL_CAPITAL / N_SLICES
        capital_curve = INITIAL_CAPITAL + np.cumsum(stake * all_vars / 100)

        plt.figure(figsize=(10, 5))  # autre figure, pas de subplot
        plt.plot(capital_curve)
        plt.title("Évolution du capital (stake fixe)")
        plt.xlabel("Trades (ordre chronologique)")
        plt.ylabel("Capital (€)")
        plt.grid(True, linestyle=":")

    plt.show()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
