"""Test coinbase module."""

from app.coinbase_handler.coinbase_handler import CoinbaseAccountClient

if __name__ == "__main__":
    client = CoinbaseAccountClient(
        api_key="organizations/08325433-29cd-4d1f-9670-f8604b4ffe81/apiKeys/b009b3d1-b758-4e2c-9dfb-a8300618cf04",
        api_secret="-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEINuirLIYuFaLqD5GRObC8B9esrNWUToeKnK0CsdnW38woAoGCCqGSM49\nAwEHoUQDQgAENHrycoH12CpAoPKagrY0y9mHxuLgy/P3U6EQs1Ay8QQeOwXRjMqm\nnB1yzgW3u6X2SC7mPyXk+hOds4LD/tcWlA==\n-----END EC PRIVATE KEY-----\n",
    )

    client.get_porfolio_value()
