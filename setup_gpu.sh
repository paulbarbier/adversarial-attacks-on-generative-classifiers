#!/bin/sh

sudo apt update && sudo apt install -y tmux

cd code
python3 -m pip install --upgrade pip
python3 -m pip install -r gpu_requirements.txt
python3 -m pip install --upgrade jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
