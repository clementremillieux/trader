"""Trader class for Alpaca trading."""

import asyncio

import json

import math

from pathlib import Path

import random

import threading

from typing import Dict, List, Optional

from app.analysis.analysis import Analysis

from app.tickers.schemas import DatasetSignal

from app.trader.schemas import PortfolioValue

from app.analysis.schemas import AnalysisOutput, AnalysisState

from app.binance_handler.binance_handler import BinanceHandler

from config.logger_config import logger


class Trader:
    """
    Automated trader that monitors positions and makes new trades based on analysis signals.

    Attributes:
        client (AlpacaAccountClient): Account client for data and orders.
        config (dict): Trader configuration parameters.
        analysis (Any): External analysis module with `run(ticker)` method.
        logger (logging.Logger): Logger instance.
    """

    def __init__(self, main_currency: str) -> None:
        """
        Initialize the Trader.

        Args:
            client (AlpacaAccountClient): Alpaca account client.
            config (dict): Configuration containing:
                - tickers (List[str]): Symbols to watch.
                - stop_loss_pct (float): Max drawdown pct before closing.
                - tranche_pct (float): Percent of portfolio to invest per buy.
            analysis (Any): Module with `run(ticker: str) -> bool`.
        """

        self.main_currency: str = main_currency

        self.stop_loss_pct: float = 0.065

        self.tranche_pct: float = 0.075

        self.window_size = 1000

        self.interval = "1h"

        self.momentum_period = 5

        self.rsi_period = 5

        self.window_size = 1000

        self.client = BinanceHandler(main_currency=main_currency)

        signals = [
            DatasetSignal(
                column_name="Close",
                is_derivative=True,
                is_normalize=True,
                is_financial=True,
            ),
            DatasetSignal(
                column_name="Volume",
                is_derivative=True,
                is_normalize=True,
                is_financial=False,
            ),
        ]

        self.analysis = Analysis(
            model_path="app/model/model_epoch_crypto_176.pth",
            signals=signals,
            num_historical_features=12,
            encoder_length=self.window_size,
            hidden_size=1024,
            dropout=0.5,
            lstm_layers=4,
            n_heads=8,
            num_attention_layers=4,
            patch_size=int(self.window_size / 8),
        )

        self.persistence_path = Path("./app/trader/high_values.json")

        if self.persistence_path.exists():
            self.highs = json.loads(self.persistence_path.read_text())

        else:
            self.highs = {}

        self.tickers: List[str] = self.client.get_all_tickers()

    async def _save_highs(self):
        """Save highs to a temporary file and replace the original."""

        for i in range(3):
            try:
                tmp = self.persistence_path.with_suffix(".tmp")

                tmp.write_text(json.dumps(self.highs))

                tmp.replace(self.persistence_path)

                return

            except Exception as e:
                logger.error("TRADER => Error saving highs [%d/%d]: %s", i, 3, e)

        try:
            tmp.unlink()

        except Exception as e:
            logger.error("TRADER => Error deleting temp file: %s", e)

    @staticmethod
    def floor_decimals(x: float, decimals: int) -> float:
        """
        Round down a float to a specified number of decimal places.
        Args:
            x (float): The number to round down.
            decimals (int): The number of decimal places to keep.
        Returns:
            float: The rounded down number.
        """

        factor = 10**decimals

        return math.floor(x * factor) / factor

    async def monitor_positions(self) -> None:
        """
        Check open positions for drawdown and close if loss exceeds threshold.
        """

        positions = self.client.get_positions() or []

        self.highs: Dict[str, float] = json.loads(self.persistence_path.read_text())

        for pos in positions:
            symbol = pos.symbol

            curr = float(pos.price)

            if symbol not in self.highs:
                self.highs[symbol] = curr

                await self._save_highs()

            high = self.highs.get(symbol, pos.price)

            if curr > high:
                self.highs[symbol] = curr

                await self._save_highs()

            drawdown = (curr - self.highs[symbol]) / self.highs[symbol]

            logger.info(
                "TRADER => %s drawdown = %.2f%% (vs high %.2f)",
                symbol,
                drawdown * 100,
                self.highs[symbol],
            )

            if drawdown < -self.stop_loss_pct:
                logger.info(
                    "TRADER => %s hit trailing stop (%.2f%%). Closing.",
                    symbol,
                    self.stop_loss_pct * 100,
                )

                self.client.submit_order(
                    symbol=symbol,
                    quantity=self.floor_decimals(pos.qty, 5),
                    side="SELL",
                    order_type="MARKET",
                )

                continue

            try:
                signal: AnalysisOutput = await self.analysis.analyze(
                    ticker=symbol,
                    window_size=self.window_size,
                    interval=self.interval,
                    momentum_period=self.momentum_period,
                    rsi_period=self.rsi_period,
                    nb_windows=5,
                    nb_2=4,
                    nb_last_2=1,
                    distance_0=2,
                    distance_1=4,
                    name="monitor",
                )

                if signal.state == AnalysisState.SELL:
                    logger.info(
                        "TRADER => Position %s triggered SELL signal. Closing for %.10f%%",
                        symbol,
                        pos.qty,
                    )

                    self.client.submit_order(
                        symbol=symbol,
                        quantity=self.floor_decimals(pos.qty, 5),
                        side="SELL",
                        order_type="MARKET",
                    )

            except Exception as e:
                logger.error("TRADER => Analysis error for %s: %s", symbol, e)

                continue

        await asyncio.sleep(60)

    async def scan_and_trade(self) -> None:
        """
        Scan tickers, run analysis, and place new buy orders.
        """

        tickers = self.tickers

        random.shuffle(tickers)

        logger.info("TRADER => Scanning %d tickers...", len(tickers))

        tranche = self.tranche_pct

        portfolio_value: Optional[PortfolioValue] = self.client.get_portfolio()

        if portfolio_value is None:
            logger.error("TRADER => Portfolio value is None. Exiting.")

            return

        total_value = portfolio_value.total_value

        buying_power = portfolio_value.buying_power

        owned = {
            p.symbol
            for p in (self.client.get_positions() or [])
            if p.price * p.qty > 0.1
        }

        for index, sym in enumerate(tickers):
            logger.info("TRADER => Analyzing %s [%d/%d]", sym, index, len(tickers))

            if (
                sym in owned
                or f"{sym}{self.main_currency}" in owned
                or sym.replace(self.main_currency, "") in owned
            ):
                logger.info("TRADER => Already own %s. Skipping.", sym)

                continue

            try:
                signal: AnalysisOutput = await self.analysis.analyze(
                    ticker=sym,
                    window_size=self.window_size,
                    interval=self.interval,
                    momentum_period=self.momentum_period,
                    rsi_period=self.rsi_period,
                    nb_windows=5,
                    nb_2=4,
                    nb_last_2=1,
                    distance_0=2,
                    distance_1=4,
                    name="scan",
                )

            except Exception as e:
                logger.error(
                    "TRADER => Analysis error for %s: %s", sym, e, exc_info=True
                )

                continue

            if signal.state == AnalysisState.BUY:
                invest_amt = total_value * tranche

                if buying_power < invest_amt:
                    logger.warning(
                        "TRADER => Insufficient buying power for %s: need %.2f, have %.2f",
                        sym,
                        invest_amt,
                        buying_power,
                    )

                    continue

                ticker_price: Optional[float] = self.client.get_ticker_price(sym)

                if ticker_price is None:
                    logger.warning(
                        "TRADER => Ticker price for %s is None. Skipping.",
                        sym,
                    )

                    continue

                qty = round(invest_amt / ticker_price, 5)

                logger.info(
                    "TRADER => Placing BUY for %s, amount=%.2f [%.4f]",
                    sym,
                    invest_amt,
                    qty,
                )

                self.client.submit_order(
                    symbol=sym,
                    quantity=qty,
                    side="BUY",
                    order_type="MARKET",
                )

                buying_power -= invest_amt

                self.highs[sym] = 0

                await self._save_highs()

    def _run_monitor_in_thread(self):
        """
        Run the monitor_positions function in a separate OS thread.
        """

        while True:
            try:
                asyncio.run(self.monitor_positions())

            except Exception as e:
                logger.error("TRADER => Monitor error: %s", e, exc_info=True)

    def _run_scan_in_thread(self):
        """
        Launch the scan_and_trade function in a separate OS thread.
        """
        while True:
            try:
                asyncio.run(self.scan_and_trade())

            except Exception as e:
                logger.error("TRADER => Scan error: %s", e, exc_info=True)

    def run(self):
        """
        Launch two separate OS threads:
          - one for monitor_positions
          - one for scan_and_trade
        """

        monitor_thread = threading.Thread(
            target=self._run_monitor_in_thread, name="monitor-thread", daemon=True
        )

        scan_thread = threading.Thread(
            target=self._run_scan_in_thread, name="scan-thread", daemon=True
        )

        monitor_thread.start()

        scan_thread.start()

        monitor_thread.join()

        scan_thread.join()


if __name__ == "__main__":
    trader = Trader(main_currency="USDC")

    trader.run()
