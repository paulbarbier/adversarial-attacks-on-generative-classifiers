#!/bin/sh

sudo apt update && sudo apt install -y tmux

cd code

python3 -m pip install --upgrade pip
python3 -m pip install -r tpu_requirements.txt
python3 -m pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

