#!/bin/sh

python -m train \
    --config=configs/fashion-mnist-gfz.py \
    --config.checkpoint_name="gfz-50-epochs-fashion-mnist" \
    --config.num_epochs=50