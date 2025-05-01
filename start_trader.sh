#!/usr/bin/env bash


cd /home/clement/trader 

# 4. Lancez le module sur le Python du venv
exec /home/clement/.local/bin/poetry run python3 -m app.trader.trader \
     >> /home/clement/trader/trader.out.log \
     2>> /home/clement/trader/trader.err.log
