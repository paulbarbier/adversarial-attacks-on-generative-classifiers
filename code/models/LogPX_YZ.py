from flax import struct
from flax import linen as nn
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform
import numpy as np

from typing import Tuple

@struct.dataclass
class PX_YZConfiguration:
    n_classes: int
    d_latent: int
    d_hidden: int
    dropout_rate: float

    n_channels: int = 64
    input_kernel: Tuple[int, int] = (4, 4)
    kernel_size: Tuple[int, int] = (5, 5)
    strides: Tuple[int, int] = (2, 2) 


class LogPX_YZ(nn.Module):
    config: PX_YZConfiguration

    @nn.compact
    def __call__(self, X, y, z, train: bool = False): # X: (height, width), y: (n_classes,), z: (d_latent,) -> 0
        config = self.config
        inputs = jnp.concatenate([y, z], 0)
        inputs = nn.Dense(
            features=config.d_hidden, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(inputs)
        inputs = nn.Dropout(rate=config.dropout_rate, deterministic=not train)(inputs)
        inputs = nn.relu(inputs)
        inputs = nn.Dense(
            features=np.prod(config.input_kernel) * config.n_channels, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(inputs)
        inputs = nn.Dropout(rate=config.dropout_rate, deterministic=not train)(inputs)
        inputs = nn.relu(inputs)
        
        inputs = inputs.reshape(config.input_kernel + (config.n_channels,))
        inputs = nn.ConvTranspose(
            features=config.n_channels,
            kernel_size=config.kernel_size,
            strides=config.strides,
            padding=(2, 2),
            kernel_init=glorot_uniform(),
        )(inputs)
        inputs = nn.relu(inputs)

        inputs = nn.ConvTranspose(
            features=config.n_channels,
            kernel_size=config.kernel_size,
            strides=config.strides,
            padding=((2, 3), (2, 3)),
            kernel_init=glorot_uniform(),
        )(inputs)
        inputs = nn.relu(inputs)

        inputs = nn.ConvTranspose(
            features=1,
            kernel_size=config.kernel_size,
            strides=config.strides,
            padding=((2, 3), (2, 3)),
            kernel_init=glorot_uniform(),
        )(inputs)
        generated_X = nn.sigmoid(inputs)

        logit = -jnp.sum((generated_X - X)**2)
        return logit