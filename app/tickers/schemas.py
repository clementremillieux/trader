from typing import List, Optional

from datetime import datetime as dt

import pandas as pd

from pydantic import BaseModel, ConfigDict


class Topic(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    topic: str

    relevance_score: float


class TickerSentiment(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    ticker: str

    relevance_score: float

    ticker_sentiment_score: float

    ticker_sentiment_label: str


class NewsItem(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    title: str

    url: str

    time_published: str

    content: Optional[str] = None

    authors: Optional[List[str]]

    summary: Optional[str]

    banner_image: Optional[str]

    source: str

    category_within_source: Optional[str]

    source_domain: str

    topics: Optional[List[Topic]]

    overall_sentiment_score: float

    overall_sentiment_label: str

    ticker_sentiment: Optional[List[TickerSentiment]]


class News(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    articles: List[NewsItem]


class PriceItem(BaseModel):
    """PriceItem model."""

    date: dt

    open: float

    high: float

    low: float

    close: float

    volume: float


class Stock(BaseModel):
    """Stock model."""

    stock: List[PriceItem]


class TickerData(BaseModel):
    """TickerData model."""

    name: str

    sql_db_id: Optional[str] = None

    news: Optional[News] = None

    stock: Optional[pd.DataFrame] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DatasetSignal(BaseModel):
    """DatasetSignal model."""

    column_name: str

    is_derivative: bool

    is_normalize: bool

    is_financial: bool
