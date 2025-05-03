#!/usr/bin/env bash


cd /home/clement/trader_cryto 

# 4. Lancez le module sur le Python du venv
exec /home/clement/.local/bin/poetry run python3 -m app.trader.trader \
     >> /home/clement/trader_cryto/trader.out.log \
     2>> /home/clement/trader_cryto/trader.err.log
