from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax.nn import one_hot
from ml_collections import ConfigDict, config_dict
from torch.utils import data
import orbax.checkpoint as ocp

def plot_image(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()

def get_data_config(dataset: data.Dataset) -> ConfigDict:
    config = config_dict.ConfigDict()

    config.dataset_image_shape = np.array(dataset.data[0]).shape
    config.image_shape = config.dataset_image_shape + (1,)
    config.n_classes = len(np.unique(dataset.targets))
    config.n_images = len(dataset.targets)
    return config

def prepare_test_dataset(dataset: data.Dataset, dataset_config: ConfigDict):
    config_image_shape = dataset_config.image_shape
    if type(config_image_shape) is list:
        config_image_shape = tuple(config_image_shape)
    image_shape = (-1,) + config_image_shape
    images = jnp.array(dataset.data, dtype=jnp.float32).reshape(image_shape)/255.0
    labels = one_hot(jnp.array(dataset.targets, dtype=jnp.float32), dataset_config.n_classes)
    return images, labels

def load_checkpoint(path: Path) -> object:
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(path) 