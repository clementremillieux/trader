#!/usr/bin/env bash

if [ -f "$HOME/.zprofile" ]; then
  source "$HOME/.zprofile"
fi

cd /home/clement/trader

exec poetry run python3 -m app.trader.trader \
     >> /Users/remillieux/Documents/trader_auto/trader.out.log \
     2>> /Users/remillieux/Documents/trader_auto/trader.err.log
