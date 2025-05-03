"""Coinbase Advanced-Py SDK wrapper for account management, orders, and market data."""

from typing import List, Optional

from coinbase.rest import RESTClient

from coinbase.websocket import WSClient

from coinbase.rest.types.common_types import (
    Amount,
)

from coinbase.rest.types.portfolios_types import (
    Portfolio,
    ListPortfoliosResponse,
)

from coinbase.rest.types.perpetuals_types import (
    GetPerpetualsPortfolioSummaryResponse,
)


from config.logger_config import logger


class CoinbaseAccountClient:
    """
    Wrapper around Coinbase Advanced-Py SDK to manage account, orders, and market data.

    Attributes:
        client (RESTClient): Coinbase REST API client instance.
        ws_client (WSClient): Coinbase WebSocket client instance.
        logger (logging.Logger): Logger for debug and info messages.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
    ):
        """
        Initialize the CoinbaseAccountClient.

        Args:
            api_key (str): Your Coinbase CDP API key.
            api_secret (str): Your Coinbase CDP API secret.
            sandbox (bool): If True, connect to sandbox environment (not implemented here).
        """
        self.client = RESTClient(
            api_key=api_key,
            api_secret=api_secret,
        )

    def get_porfolio_value(self) -> Optional[PortfolioValue]:
        """
        Retrieve the total portfolio value in USD.
        """
        logger.info("COINBASE => Fetching portfolio value.")

        portfolios: ListPortfoliosResponse = self.client.get_portfolios()

        if not portfolios.portfolios:
            logger.warning("COINBASE => No portfolios found.")

            return None

        for portfolio in portfolios.portfolios:
            logger.info(
                "COINBASE => Portfolio: %s, UUID: %s",
                portfolio.name,
                portfolio.uuid,
            )

        portfolio: Optional[Portfolio] = (
            portfolios.portfolios[0] if portfolios else None
        )

        if not portfolio:
            logger.warning("COINBASE => No portfolio found.")

            return None

        if not portfolio.uuid:
            logger.warning("COINBASE => Portfolio UUID not found.")

            return None

        portfolio_summary: GetPerpetualsPortfolioSummaryResponse = (
            self.client.get_perps_portfolio_summary(portfolio_uuid=portfolio.uuid)
        )

        if not portfolio_summary:
            logger.warning("COINBASE => Portfolio summary not found.")

            return None

        if not portfolio_summary.summary:
            logger.warning("COINBASE => Portfolio summary data not found.")

            return None

        if not portfolio_summary.summary.total_balance:
            logger.warning("COINBASE => Portfolio total balance not found.")

            return None

        total_balance: Amount = portfolio_summary.summary.total_balance

        logger.info(
            "COINBASE => Portfolio total balance: %s %s",
            total_balance.value,
            total_balance.currency,
        )

        if not total_balance.value:
            logger.warning("COINBASE => Portfolio total balance value not found.")

            return None

        if not portfolio_summary.summary.buying_power:
            logger.warning("COINBASE => Portfolio buying power not found.")

            return None

        buying_power: Amount = portfolio_summary.summary.buying_power

        if not buying_power.value:
            logger.warning("COINBASE => Portfolio buying power value not found.")

            return None

        logger.info(
            "COINBASE => Portfolio buying power: %s %s",
            buying_power.value,
            total_balance.currency,
        )

        return PortfolioValue(
            total_value=float(total_balance.value),
            buying_power=float(buying_power.value),
        )

    # def get_all_assets(
    #     self,
    #     status: Optional[str] = None,
    #     asset_class: AssetClass = AssetClass.US_EQUITY,
    #     exchange: Optional[str] = None,
    # ) -> Optional[List[dict]]:
    #     """
    #     Coinbase does not support asset listing; return account balances instead.
    #     Args mirror Alpaca signature for compatibility.
    #     """
    #     logger.info(
    #         "COINBASE => Fetching assets (balances) status=%s, class=%s, exchange=%s",
    #         status,
    #         asset_class,
    #         exchange,
    #     )
    #     return self.get_account()

    # def get_orders(
    #     self,
    #     status: QueryOrderStatus = QueryOrderStatus.ALL,
    #     side: Optional[OrderSide] = None,
    # ) -> Optional[List[dict]]:
    #     """
    #     Retrieve orders with optional filtering.
    #     """
    #     logger.info("COINBASE => Fetching orders: status=%s, side=%s", status, side)
    #     # Coinbase Advanced API list orders
    #     orders = self.client.get_orders()
    #     orders_list = orders.to_dict() if hasattr(orders, "to_dict") else orders
    #     if side:
    #         return [o for o in orders_list if o.get("side", "").upper() == side.value]
    #     return orders_list

    # def submit_order(
    #     self,
    #     symbol: str,
    #     qty: Optional[float] = None,
    #     notional: Optional[float] = None,
    #     side: OrderSide = OrderSide.BUY,
    #     order_type: str = "market",
    #     time_in_force: TimeInForce = TimeInForce.DAY,
    #     **kwargs,
    # ) -> dict:
    #     """
    #     Submit a new order to Coinbase.

    #     Args match Alpaca signature for compatibility:
    #         symbol (str): Market ID, e.g., 'BTC-USD'.
    #         qty (float): Asset amount (base size).
    #         notional (float): Quote amount for market orders.
    #         side (OrderSide): BUY or SELL.
    #         order_type (str): 'market' or 'limit'.
    #         time_in_force (TimeInForce): Not supported by Coinbase SDK.
    #         **kwargs: Additional parameters like limit_price.
    #     """
    #     logger.info(
    #         "COINBASE => Submitting %s order: symbol=%s, qty=%s, notional=%s, extra=%s",
    #         order_type,
    #         symbol,
    #         qty,
    #         notional,
    #         kwargs,
    #     )
    #     order_type = order_type.lower()
    #     client_order_id = kwargs.get("client_order_id", "")
    #     if order_type == "market":
    #         if side == OrderSide.BUY:
    #             return self.client.market_order_buy(
    #                 client_order_id=client_order_id,
    #                 product_id=symbol,
    #                 quote_size=str(notional or 0),
    #             ).to_dict()
    #         else:
    #             # SELL uses size param
    #             return self.client.market_order_sell(
    #                 client_order_id=client_order_id,
    #                 product_id=symbol,
    #                 size=str(qty or 0),
    #             ).to_dict()
    #     elif order_type == "limit":
    #         price = kwargs.get("limit_price")
    #         if side == OrderSide.BUY:
    #             return self.client.limit_order_buy(
    #                 client_order_id=client_order_id,
    #                 product_id=symbol,
    #                 price=str(price),
    #                 size=str(qty or 0),
    #             ).to_dict()
    #         else:
    #             return self.client.limit_order_sell(
    #                 client_order_id=client_order_id,
    #                 product_id=symbol,
    #                 price=str(price),
    #                 size=str(qty or 0),
    #             ).to_dict()
    #     else:
    #         logger.error("COINBASE => Unsupported order type: %s", order_type)
    #         raise ValueError(f"Unsupported order type: {order_type}")

    # def cancel_all_orders(self) -> Optional[List[dict]]:
    #     """
    #     Cancel all open orders.
    #     """
    #     logger.info("COINBASE => Cancelling all open orders.")
    #     orders = self.get_orders()
    #     responses = []
    #     for o in orders:
    #         try:
    #             resp = self.client.cancel_order(order_id=o.get("id"))
    #             responses.append(resp.to_dict() if hasattr(resp, "to_dict") else resp)
    #         except Exception as e:
    #             logger.error(
    #                 "COINBASE => Failed to cancel order %s: %s", o.get("id"), e
    #             )
    #     return responses

    # def get_positions(self) -> Optional[List]:
    #     """
    #     Coinbase does not support positions via REST; stub for interface compatibility.
    #     """
    #     logger.info("COINBASE => get_positions not supported.")
    #     return []

    # def close_all_positions(self, cancel_orders: bool = True) -> Optional[List]:
    #     """
    #     Coinbase does not support positions via REST; stub for interface compatibility.
    #     """
    #     logger.info("COINBASE => close_all_positions not supported.")
    #     if cancel_orders:
    #         self.cancel_all_orders()
    #     return []

    # def get_ticker_data(
    #     self,
    #     ticker: str,
    #     interval: str,
    #     days: str,
    # ):
    #     """
    #     Coinbase Advanced-Py SDK does not provide historical REST data; stub for interface compatibility.
    #     """
    #     logger.error("COINBASE => get_ticker_data not supported via REST SDK.")
    #     return None
