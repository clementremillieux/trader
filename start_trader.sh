#!/usr/bin/env bash

# 1. Assurez‐vous que ~/.local/bin (et pyenv si vous l’utilisez) est bien dans le PATH
export PATH="$HOME/.local/bin:$HOME/.pyenv/shims:$PATH"

# 2. Allez dans le dossier du projet
cd /home/clement/trader || exit 1

# 3. Récupérez le chemin du venv Poetry
VENV_PATH=$(/home/clement/.local/bin/poetry env info -p)

# 4. Lancez le module sur le Python du venv
exec "$VENV_PATH/bin/python" -m app.trader.trader \
     >> /home/clement/trader/trader.out.log \
     2>> /home/clement/trader/trader.err.log
