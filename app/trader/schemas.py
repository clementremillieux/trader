"""Portfolio schemas."""

from pydantic import BaseModel


class PortfolioValue(BaseModel):
    """
    Portfolio value in USD.
    """

    total_value: float

    buying_power: float


class Position(BaseModel):
    """
    Position in the portfolio.
    """

    symbol: str

    qty: float

    price: float
