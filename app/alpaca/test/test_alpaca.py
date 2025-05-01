"""Test alapaca module."""

from app.alpaca.alpaca_handler import AlpacaAccountClient

AlpacaAccountClient(
    api_key="PKPAPAB1JWAA5PLJSFZ4",
    secret_key="HTMo7Lj68OpsXsiTkKIn6JV4FYlWtNtqEbd22vlv",
    paper=True,
).get_account()
