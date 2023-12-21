from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from jax.nn import one_hot
from ml_collections import ConfigDict, config_dict
from torch.utils import data
import orbax.checkpoint as ocp

import models.ClassifierGFZ as ClassifierGFZ
import models.ClassifierDFZ as ClassifierDFZ

import attacks.DeepFool as DeepFool
import attacks.DeepFoolPaper as DeepFoolPaper
import attacks.FastGradientSign as FastGradientSign
import attacks.DeepFoolBetter as DeepFoolBetter

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

def prepare_test_dataset(dataset: data.Dataset, dataset_config: ConfigDict, dtype):
    image_shape = (-1,) + dataset_config.image_shape
    images = jnp.array(dataset.data, dtype=dtype).reshape(image_shape)/255.0
    labels = one_hot(jnp.array(dataset.targets, dtype=dtype), dataset_config.n_classes)
    return images, labels

def load_checkpoint(path: Path) -> object:
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(path) 

def get_dtype(dtype_option: str):
    if dtype_option == "float32":
        dtype = jnp.float32
    elif dtype_option == "bfloat16":
        dtype = jnp.bfloat16
    else:
        raise NotImplementedError(dtype_option)
    return dtype

def get_classifier(config: ConfigDict):
    if config.model_name == "GFZ":
        classifier = ClassifierGFZ
    elif config.model_name == "DFZ":
        classifier = ClassifierDFZ
    else:
        raise NotImplementedError(config.model_name)
    return classifier

def get_attack_model(config: ConfigDict):
    if config.attack_name == "deepfool":
        attack_model = DeepFool
    elif config.attack_name == "deepfoolpaper":
        attack_model = DeepFoolPaper
    elif config.attack_name == "deepfoolbetter":
        attack_model = DeepFoolBetter
    elif config.attack_name == "fastgradientsign":
        attack_model = FastGradientSign
    else:
        raise NotImplementedError(config.attack_name)
    return attack_model

@jax.vmap
def perturbation_norm(truth, predicted):
    return jnp.linalg.norm(truth - predicted) / jnp.linalg.norm(truth)