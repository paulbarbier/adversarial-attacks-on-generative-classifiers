import optax
from ml_collections import ConfigDict

def get_optimiser(config: ConfigDict):
    if config.optimiser == "adam":
        return optax.adam(config.learning_rate)
    else:
        raise NotImplementedError(config.optimiser)