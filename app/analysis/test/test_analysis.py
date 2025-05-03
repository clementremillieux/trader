"""Test the analysis module."""

from app.analysis.analysis import Analysis

from app.tickers.schemas import DatasetSignal


async def test_analysis():
    """Test the analysis module."""

    tickers = [
        "BTCUSDT",
    ]

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

    momentum_period = 5

    rsi_period = 5

    analysis = Analysis(
        model_path="app/model/model_epoch_crypto_44.pth",
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

    for ticker in tickers:
        res = await analysis.analyze(
            ticker=ticker,
            window_size=window_size,
            interval=interval,
            momentum_period=momentum_period,
            rsi_period=rsi_period,
            nb_windows=10,
            nb_2=8,
            nb_last_2=2,
            distance_0=2,
            distance_1=4,
            name="test",
        )

        print(f"Ticker: {ticker}")

        print(f"\t- {res}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_analysis())
