from flax import struct
from flax import linen as nn
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform, glorot_normal

from models.utils import transform, log_gaussian

from typing import Tuple

@struct.dataclass
class QZ_XYConfiguration:
    n_classes: int
    d_latent: int
    d_hidden: int
    dropout_rate: float

    n_convolutions: int = 3
    n_channels: int = 64
    kernel_size: Tuple[int, int] = (5, 5)
    strides: Tuple[int, int] = (2, 2)
    
class LogQZ_XY(nn.Module):
    config: QZ_XYConfiguration 

    @nn.compact
    def __call__(self, X, y, epsilon, train: bool = False): # X: (height, width), y: (n_classes,), prior: (d_latent,) -> (d_latent,), 0
        config = self.config
        for _ in range(config.n_convolutions):
            X = nn.Conv(
                features=config.n_channels, 
                kernel_size=config.kernel_size, 
                strides=config.strides, 
                kernel_init=glorot_normal(),
            )(X)
            X = nn.relu(X)

        X_flatten = X.reshape(-1)
        X_flatten = nn.Dense(
            features=config.d_hidden, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(X_flatten)
        X_flatten = nn.Dropout(rate=config.dropout_rate, deterministic=not train)(X_flatten)
        X_flatten = nn.relu(X_flatten)
        
        output = jnp.concatenate((X_flatten, y), axis=0)
        output = nn.Dense(
            features=config.d_hidden, 
            use_bias=False,
            kernel_init=glorot_uniform(), 
        )(output)
        output = nn.Dropout(rate=config.dropout_rate, deterministic=not train)(output)
        output = nn.relu(output)

        output = nn.Dense(
            features=2*config.d_latent, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(output)
        # end of model
        mu, log_sigma = jnp.split(output, 2)
        
        z = transform(epsilon, mu, log_sigma)
        logit = log_gaussian(z, mu, log_sigma)
        return z, logit