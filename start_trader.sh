#!/usr/bin/env bash

if [ -f "$HOME/.zprofile" ]; then
  source "$HOME/.zprofile"
fi

cd /home/clement/trader

exec poetry run python3 -m app.trader.trader \
     >> /home/clement/trader/trader.out.log \
     2>> /home/clement/trader/trader.err.log
