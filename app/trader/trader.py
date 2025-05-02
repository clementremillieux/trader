"""Trader class for Alpaca trading."""

import asyncio

import json
from pathlib import Path
import threading

from typing import List

from alpaca.trading.enums import OrderSide

from app.trader.tickers import TICKERS

from app.analysis.analysis import Analysis

from app.tickers.schemas import DatasetSignal

from app.alpaca.alpaca_handler import AlpacaAccountClient

from app.analysis.schemas import AnalysisOutput, AnalysisState

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

    def __init__(
        self,
    ) -> None:
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

        self.tickers: List[str] = TICKERS

        self.stop_loss_pct: float = 0.04

        self.tranche_pct: float = 0.025

        self.window_size = 1000

        self.interval = "1h"

        self.days = "500"

        self.momentum_period = 5

        self.rsi_period = 5

        self.window_size = 1000

        self.client = AlpacaAccountClient(
            api_key="PKPAPAB1JWAA5PLJSFZ4",
            secret_key="HTMo7Lj68OpsXsiTkKIn6JV4FYlWtNtqEbd22vlv",
            paper=True,
        )

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
            model_path="app/model/model_epoch_4_96.pth",
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

    async def _save_highs(self):
        """Save highs to a temporary file and replace the original."""

        tmp = self.persistence_path.with_suffix(".tmp")

        tmp.write_text(json.dumps(self.highs))

        tmp.replace(self.persistence_path)

    async def monitor_positions(self) -> None:
        """
        Check open positions for drawdown and close if loss exceeds threshold.
        """

        positions = self.client.get_positions() or []

        self.highs = json.loads(self.persistence_path.read_text())

        for pos in positions:
            symbol = pos.symbol

            curr = float(pos.current_price)

            if symbol not in self.highs:
                self.highs[symbol] = float(pos.avg_entry_price)

                await self._save_highs()

            high = self.highs.get(symbol, float(pos.avg_entry_price))

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
                    qty=float(pos.qty),
                    side=OrderSide.SELL,
                    order_type="market",
                )

            try:
                signal: AnalysisOutput = await self.analysis.analyze(
                    ticker=symbol,
                    window_size=self.window_size,
                    interval=self.interval,
                    days=self.days,
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
                        "TRADER => Position %s triggered SELL signal. Closing.",
                        symbol,
                    )

                    self.client.submit_order(
                        symbol=symbol,
                        qty=float(pos.qty),
                        side=OrderSide.SELL,
                        order_type="market",
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

        tranche = self.tranche_pct

        account = self.client.get_account()

        portfolio_value = float(account.portfolio_value)

        buying_power = float(account.buying_power)

        owned = {p.symbol for p in (self.client.get_positions() or [])}

        for sym in tickers:
            if sym in owned:
                continue

            try:
                signal: AnalysisOutput = await self.analysis.analyze(
                    ticker=sym,
                    window_size=self.window_size,
                    interval=self.interval,
                    days=self.days,
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
                invest_amt = portfolio_value * tranche

                if buying_power < invest_amt:
                    logger.warning(
                        "TRADER => Insufficient buying power for %s: need %.2f, have %.2f",
                        sym,
                        invest_amt,
                        buying_power,
                    )
                    continue

                qty = invest_amt

                logger.info(
                    "TRADER => Placing buy for %s, amount=%.2f", sym, invest_amt
                )

                self.client.submit_order(
                    symbol=sym,
                    notional=qty,
                    side=OrderSide.BUY,
                    order_type="market",
                )

                buying_power -= invest_amt

    def _run_monitor_in_thread(self):
        """
        Run the monitor_positions function in a separate OS thread.
        """

        while True:
            try:
                asyncio.run(self.monitor_positions())

            except Exception as e:
                logger.error("TRADER => Monitor error: %s", e, exc_info=True)

        asyncio.run(self.monitor_positions())

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
    trader = Trader()

    trader.run()
