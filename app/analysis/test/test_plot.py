"""Test the analysis module."""

from app.analysis.analysis import Analysis

from app.tickers.schemas import DatasetSignal


async def test_analysis():
    """Test the analysis module."""

    ticker = "KGEI"

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

    window_size = 1000

    interval = "1h"

    days = "500"

    momentum_period = 5

    rsi_period = 5

    analysis = Analysis(
        model_path="app/model/model_epoch_4_96.pth",
        signals=signals,
        num_historical_features=12,
        encoder_length=window_size,
        hidden_size=1024,
        dropout=0.5,
        lstm_layers=4,
        n_heads=8,
        num_attention_layers=4,
        patch_size=int(window_size / 8),
    )

    await analysis.plot_last_windows(
        ticker=ticker,
        window_size=window_size,
        interval=interval,
        days=days,
        momentum_period=momentum_period,
        rsi_period=rsi_period,
        nb_windows=200,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_analysis())
