import jax
from flax import struct
from flax import linen as nn
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform, glorot_normal

from models.utils import softmax_cross_entropy_with_logits, transform, log_gaussian

from typing import Tuple

@struct.dataclass
class PY_XZConfiguration:
    n_classes: int
    d_latent: int
    d_hidden: int
    dropout_rate: float

    n_convolutions: int = 3
    n_channels: int = 64
    kernel_size: Tuple[int, int] = (5, 5)
    strides: Tuple[int, int] = (2, 2)

class LogPY_XZ(nn.Module):
    config: PY_XZConfiguration

    @nn.compact
    def __call__(self, X, y, z, train: bool = False): # X: (height, width), y: (n_classes,), prior: (d_latent,) -> (d_latent,), 0
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
        
        output = jnp.concatenate((X_flatten, z), axis=0)
        output = nn.Dense(
            features=config.d_hidden, 
            use_bias=False,
            kernel_init=glorot_uniform(), 
        )(output)
        output = nn.Dropout(rate=config.dropout_rate, deterministic=not train)(output)
        output = nn.relu(output)

        logits = nn.Dense(
            features=config.n_classes, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(output)
        # end of model
        return -softmax_cross_entropy_with_logits(logits, jax.lax.stop_gradient(y))