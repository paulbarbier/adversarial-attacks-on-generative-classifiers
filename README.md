# Main repo for the Generative Classifiers PGM Project

## Setup requirements

Please follow the following instructions to install the required packages:

```bash
cd code;
python -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Or alternatively, run the `setup.sh` script.

## Training instructions

you can train the model using the following command:

```bash
cd code; python -m train --config=configs/fashion-mnist-gfz.py --config.checkpoint_name="checkpoint-name" --config.num_epochs=1
```

the params checkpoints after each epoch will be stored under `checkpoints`. there is a `load_checkpoint` function to load them back.