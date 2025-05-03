sudo apt update

sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev

git remote --set-url git@github.com:clementremillieux/trader.git

ssh-keygen -t ed25519 -C "clement.remillieux@gmail.com" -y

eval "$(ssh-agent -s)"

ssh-add ~/.ssh/id_ed25519

cat ~/.ssh/id_ed25519.pub 

curl -sSL https://install.python-poetry.org | python3 -

export PATH="$HOME/.local/bin:$PATH"

curl -fsSL https://pyenv.run | bash

pyenv install 3.10.1

pyenv global 3.10.1

poetry config virtualenvs.in-project true

poetry env use 3.10.1

poetry install

poetry run pip install torch

sudo cp trader.service /etc/systemd/system/trader.service

sudo systemctl daemon-reload

sudo systemctl enable trader.service

sudo systemctl start trader.service

sudo systemctl status trader.service

journalctl -u trader.service -f