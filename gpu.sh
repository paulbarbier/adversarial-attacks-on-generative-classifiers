#!/bin/sh

cd code
python -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r gpu_requirements.txt

python -m train --config=configs/fashion-mnist-dfz.py
python -m train --config=configs/fashion-mnist-gfz.py