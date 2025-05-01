"""Analysis schemas."""

from enum import Enum

from pydantic import BaseModel


class AnalysisState(str, Enum):
    """
    Enum for analysis states.
    """

    SELL = "sell"

    BUY = "buy"

    HOLD = "hold"


class AnalysisOutput(BaseModel):
    """
    Analysis output schema.
    """

    state: AnalysisState
