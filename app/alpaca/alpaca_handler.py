"""Alpaca handler module."""

from typing import List, Optional

from alpaca.common import RawData

from alpaca.trading.client import TradingClient

from alpaca.trading.models import TradeAccount, ClosePositionResponse, Position

from alpaca.trading.requests import (
    OrderRequest,
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
    GetAssetsRequest,
    GetOrdersRequest,
    CancelOrderResponse,
)

from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    QueryOrderStatus,
    AssetClass,
)

from config.logger_config import logger


class AlpacaAccountClient:
    """
    Wrapper around Alpaca TradingClient to manage account, assets, and orders.

    Attributes:
        client (TradingClient): Alpaca TradingClient instance.
        logger (logging.Logger): Logger configured with lazy formatting.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ):
        """
        Initialize the AlpacaAccountClient.

        Args:
            api_key (str): Your Alpaca API key.
            secret_key (str): Your Alpaca secret key.
            paper (bool, optional): Whether to use paper trading. Defaults to True.
            base_url (Optional[str], optional): Custom Alpaca API URL. Defaults to None.
        """

        logger.info("ALPACA => Initializing TradingClient (paper=%s)", paper)

        self.client = TradingClient(api_key=api_key, paper=paper, secret_key=secret_key)

        logger.info("AlpacaAccountClient initialized successfully.")

    def get_account(self) -> Optional[TradeAccount]:
        """
        Retrieve account details.
        """
        logger.info("Fetching account details.")

        account_data: TradeAccount | RawData = self.client.get_account()

        if isinstance(account_data, TradeAccount):
            logger.info(
                "ALPACA => Account details retrieved successfully : %s",
                account_data.model_dump_json(indent=2),
            )

            return account_data

        else:
            logger.error("ALPACA => Failed to retrieve account details.")

            return None

    def get_all_assets(
        self,
        status: Optional[str] = None,
        asset_class: AssetClass = AssetClass.US_EQUITY,
        exchange: Optional[str] = None,
    ):
        """
        Get all assets matching filters.

        Args:
            status (Optional[str]): AssetStatus filter (e.g., 'active').
            asset_class (AssetClass): AssetClass enum.
            exchange (Optional[str]): Exchange filter (e.g., 'NASDAQ').
        """
        logger.info(
            "ALPACA => Fetching assets: status=%s, class=%s, exchange=%s",
            status,
            asset_class,
            exchange,
        )

        params = GetAssetsRequest(
            asset_class=asset_class,
            status=status,
            exchange=exchange,
        )

        return self.client.get_all_assets(params)

    def get_orders(
        self,
        status: QueryOrderStatus = QueryOrderStatus.ALL,
        side: Optional[OrderSide] = None,
    ):
        """
        Retrieve orders with optional filtering.

        Args:
            status (QueryOrderStatus): Filter orders by status.
            side (Optional[OrderSide]): Filter by buy/sell.
        """

        logger.info("ALPACA =>Fetching orders: status=%s, side=%s", status, side)

        params = GetOrdersRequest(status=status, side=side.value if side else None)

        return self.client.get_orders(filter=params)

    def submit_order(
        self,
        symbol: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        side: OrderSide = OrderSide.BUY,
        order_type: str = "market",
        time_in_force: TimeInForce = TimeInForce.DAY,
        **kwargs,
    ):
        """
        Submit a new order.

        Args:
            symbol (str): The symbol to trade (e.g., 'AAPL').
            qty (Optional[float]): Quantity of shares (for stock) or units (for crypto).
            notional (Optional[float]): Order size in USD.
            side (OrderSide): BUY or SELL.
            order_type (str): Type of order: 'market', 'limit', 'stop', 'trailing_stop'.
            time_in_force (TimeInForce): Time in force.
            **kwargs: Additional parameters per order type (e.g., limit_price, stop_price, trail_percent).

        Returns:
            Order object from Alpaca.
        """

        if notional:
            notional = round(notional, 2)

        logger.info(
            "ALPACA =>Submitting %s order: symbol=%s, qty=%s, notional=%s, tif=%s, extra=%s",
            order_type,
            symbol,
            qty,
            notional,
            time_in_force,
            kwargs,
        )

        order_data: Optional[OrderRequest] = None

        order_type = order_type.lower()

        if order_type == "market":
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                notional=notional,
                side=side.value,
                time_in_force=time_in_force,
            )

        elif order_type == "limit":
            order_data = LimitOrderRequest(
                symbol=symbol,
                limit_price=kwargs.get("limit_price"),
                qty=qty,
                notional=notional,
                side=side.value,
                time_in_force=time_in_force,
            )

        elif order_type == "stop":
            order_data = StopOrderRequest(
                symbol=symbol,
                stop_price=kwargs.get("stop_price"),
                qty=qty,
                notional=notional,
                side=side.value,
                time_in_force=time_in_force,
            )

        elif order_type == "trailing_stop":
            order_data = TrailingStopOrderRequest(
                symbol=symbol,
                trail_percent=kwargs.get("trail_percent"),
                qty=qty,
                notional=notional,
                side=side.value,
                time_in_force=time_in_force,
            )

        else:
            logger.info("ALPACA =>nsupported order type: %s", order_type)

            raise ValueError(f"Unsupported order type: {order_type}")

        return self.client.submit_order(order_data)

    def cancel_all_orders(self) -> Optional[List[CancelOrderResponse]]:
        """
        Cancel all open orders.
        """
        logger.info("ALPACA =>Cancelling all orders.")

        res: List[CancelOrderResponse] | RawData = self.client.cancel_orders()

        if isinstance(res, List):
            for order in res:
                logger.info(
                    "ALPACA => Cancelled order: %s",
                    order.model_dump_json(indent=2),
                )

            logger.info("ALPACA => All orders cancelled successfully.")

            return res

        else:
            logger.error("ALPACA => Failed to cancel orders.")

            return None

    def get_positions(self) -> Optional[List[Position]]:
        """
        Retrieve all open positions.
        """

        logger.info("ALPACA =>Fetching positions.")

        res: List[Position] | RawData = self.client.get_all_positions()

        if isinstance(res, List):
            for position in res:
                logger.info(
                    "ALPACA => Position: %s",
                    position.model_dump_json(indent=2),
                )

            logger.info("ALPACA => All positions retrieved successfully.")

            return res

        else:
            logger.error("ALPACA => Failed to retrieve positions.")

            return None

    def close_all_positions(
        self, cancel_orders: bool = True
    ) -> Optional[List[ClosePositionResponse]]:
        """
        Close all open positions, optionally canceling open orders first.

        Args:
            cancel_orders (bool): Whether to cancel open orders.
        """
        logger.info("ALPACA =>Closing all positions (cancel_orders=%s)", cancel_orders)

        res: List[ClosePositionResponse] | RawData = self.client.close_all_positions(
            cancel_orders=cancel_orders
        )

        if isinstance(res, List):
            for position in res:
                logger.info(
                    "ALPACA => Closed position: %s",
                    position.model_dump_json(indent=2),
                )

            logger.info("ALPACA => All positions closed successfully.")

            return res

        else:
            logger.error("ALPACA => Failed to close all positions.")

            return None
