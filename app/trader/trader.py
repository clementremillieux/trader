"""Trader class for Alpaca trading."""

import asyncio

from datetime import datetime, timedelta

import json

import math

from pathlib import Path

import random

import threading

from typing import Dict, List, Optional

from pydantic import BaseModel

from app.analysis.analysis import Analysis

from app.tickers.schemas import DatasetSignal

from app.trader.schemas import PortfolioValue, Position

from app.analysis.schemas import AnalysisOutput, AnalysisState

from app.binance_handler.binance_handler import BinanceHandler

from config.logger_config import logger


class PositionSaved(BaseModel):
    """
    Represents a position in the trading system.
    """

    symbol: str

    buy_price: float

    high_price: float

    last_sell: datetime


class PositionsSaved(BaseModel):
    """
    Represents a collection of positions in the trading system.
    """

    positions: Dict[str, PositionSaved]


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

        self.size_watch_buy: int = 1

        self.size_watch_sell: int = 3

        self.stop_loss_pct_base: float = 0.05

        self.stop_loss_pct_high: float = 0.015

        self.tranche_pct: float = 0.075

        self.window_size = 300

        self.interval = "1h"

        self.momentum_period = 5

        self.rsi_period = 5

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
            model_path="app/model/model_epoch_crypto_14.pth",
            signals=signals,
            num_historical_features=40,
            encoder_length=self.window_size,
            hidden_size=1024,
            dropout=0.6,
            lstm_layers=4,
            n_heads=8,
            num_attention_layers=8,
            patch_size=int(self.window_size / 8),
        )

        self.persistence_path = Path("./app/trader/high_values.json")

        self.tickers: List[str] = (
            self.client.get_tickers_existing_both_main_currency_usdt()
        )

        self.symbols_dict: Dict[str, str] = self.client.get_symbols_dict()

        logger.info(
            "TRADER => Found %d tradable tickers",
            len(self.tickers),
        )

        print(self.tickers)

        print(self.symbols_dict)

    async def _save_positions(self, positions: PositionsSaved) -> None:
        """Save positions to a temporary file and replace the original."""

        for i in range(3):
            try:
                tmp = self.persistence_path.with_suffix(".tmp")

                tmp.write_text(positions.model_dump_json())

                tmp.replace(self.persistence_path)

                return

            except Exception as e:
                logger.error("TRADER => Error saving positions [%d/%d]: %s", i, 3, e)

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

        positions: List[Position] = self.client.get_positions() or []

        for pos in positions:
            symbol: str = pos.symbol

            positions_saved = PositionsSaved(
                **json.loads(self.persistence_path.read_text())
            )

            current_price: float = float(pos.price)

            if symbol not in positions_saved.positions:
                positions_saved.positions[symbol] = PositionSaved(
                    symbol=symbol,
                    buy_price=current_price,
                    high_price=current_price,
                    last_sell=datetime.now(),
                )

                await self._save_positions(positions_saved)

            else:
                if current_price > positions_saved.positions[symbol].high_price:
                    positions_saved.positions[symbol].high_price = current_price

                    await self._save_positions(positions_saved)

                drawdown_base: float = (
                    current_price - positions_saved.positions[symbol].buy_price
                ) / positions_saved.positions[symbol].buy_price

                logger.info(
                    "TRADER => %s drawdown base = %.6f%% (vs base %.6f)",
                    symbol,
                    drawdown_base * 100,
                    positions_saved.positions[symbol].buy_price,
                )

                drawdown_high: float = (
                    current_price - positions_saved.positions[symbol].high_price
                ) / positions_saved.positions[symbol].high_price

                logger.info(
                    "TRADER => %s drawdown high = %.6f%% (vs high %.6f)",
                    symbol,
                    drawdown_high * 100,
                    positions_saved.positions[symbol].high_price,
                )

                if drawdown_base < -self.stop_loss_pct_base:
                    logger.info(
                        "TRADER => %s hit trailing base stop (%.6f%%). Closing.",
                        symbol,
                        self.stop_loss_pct_base * 100,
                    )

                    self.client.submit_order(
                        symbol=symbol,
                        quantity=self.floor_decimals(pos.qty, 5),
                        side="SELL",
                        order_type="MARKET",
                    )

                    continue

                if (
                    drawdown_high < -self.stop_loss_pct_high
                    and 0.05 > drawdown_base > 0.03
                ):
                    logger.info(
                        "TRADER => %s hit trailing high stop (%.2f%%). Closing.",
                        symbol,
                        self.stop_loss_pct_high * 100,
                    )

                    self.client.submit_order(
                        symbol=symbol,
                        quantity=self.floor_decimals(pos.qty, 5),
                        side="SELL",
                        order_type="MARKET",
                    )

                    continue

                if (
                    drawdown_high < -self.stop_loss_pct_high * 2
                    and 0.1 > drawdown_base > 0.5
                ):
                    logger.info(
                        "TRADER => %s hit trailing high stop (%.2f%%). Closing.",
                        symbol,
                        self.stop_loss_pct_high * 100,
                    )

                    self.client.submit_order(
                        symbol=symbol,
                        quantity=self.floor_decimals(pos.qty, 5),
                        side="SELL",
                        order_type="MARKET",
                    )

                    continue

                if drawdown_high < -self.stop_loss_pct_high * 3 and drawdown_base > 0.1:
                    logger.info(
                        "TRADER => %s hit trailing high stop (%.2f%%). Closing.",
                        symbol,
                        self.stop_loss_pct_high * 100,
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
                    nb_windows=max(self.size_watch_buy, self.size_watch_sell),
                    name="monitor",
                    sym_base_asset=self.symbols_dict,
                    size_watch_buy=self.size_watch_buy,
                    size_watch_sell=self.size_watch_sell,
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
                logger.error(
                    "TRADER => Analysis error for %s: %s", symbol, e, exc_info=True
                )

                continue

        # await asyncio.sleep(60)

    async def scan_and_trade(self) -> None:
        """
        Scan tickers, run analysis, and place new buy orders.
        """

        try:
            positions_saved = PositionsSaved(
                **json.loads(self.persistence_path.read_text())
            )

        except Exception as e:
            logger.error("TRADER => Error loading positions: %s", e)

            positions_saved = PositionsSaved(positions={})

            await self._save_positions(positions_saved)

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

            positions_saved = PositionsSaved(
                **json.loads(self.persistence_path.read_text())
            )

            if sym in positions_saved.positions and positions_saved.positions[
                sym
            ].last_sell > datetime.now() - timedelta(minutes=30):
                logger.info(
                    "TRADER => %s was sold recently. Skipping.",
                    sym,
                )

                continue

            try:
                signal: AnalysisOutput = await self.analysis.analyze(
                    ticker=sym,
                    window_size=self.window_size,
                    interval=self.interval,
                    momentum_period=self.momentum_period,
                    rsi_period=self.rsi_period,
                    name="scan",
                    sym_base_asset=self.symbols_dict,
                    nb_windows=max(self.size_watch_buy, self.size_watch_sell),
                    size_watch_buy=self.size_watch_buy,
                    size_watch_sell=self.size_watch_sell,
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

                positions_saved = PositionsSaved(
                    **json.loads(self.persistence_path.read_text())
                )

                if sym in positions_saved.positions:
                    positions_saved.positions[sym] = PositionSaved(
                        symbol=sym,
                        buy_price=0,
                        high_price=0,
                        last_sell=datetime.now(),
                    )

                    await self._save_positions(positions=positions_saved)

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
