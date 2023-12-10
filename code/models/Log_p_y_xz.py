import jax
from flax import linen as nn
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform, glorot_normal

from models.utils import softmax_cross_entropy_with_logits, transform, log_gaussian

class Log_p_y_xz(nn.Module):
    n_classes: int
    d_latent: int
    d_hidden: int
    dropout_rate: float

    n_convolutions = 3
    n_channels = 64
    kernel_size = (5, 5)
    strides = (2, 2)

    @nn.compact
    def __call__(self, X, y, z, train: bool = False): # X: (height, width), y: (n_classes,), prior: (d_latent,) -> (d_latent,), 0
        for _ in range(self.n_convolutions):
            X = nn.Conv(
                features=self.n_channels, 
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                kernel_init=glorot_normal(),
            )(X)
            X = nn.relu(X)

        X_flatten = X.reshape(-1)
        X_flatten = nn.Dense(
            features=self.d_hidden, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(X_flatten)
        X_flatten = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(X_flatten)
        X_flatten = nn.relu(X_flatten)
        
        output = jnp.concatenate((X_flatten, z), axis=0)
        output = nn.Dense(
            features=self.d_hidden, 
            use_bias=False,
            kernel_init=glorot_uniform(), 
        )(output)
        output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(output)
        output = nn.relu(output)

        logits = nn.Dense(
            features=self.n_classes, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(output)
        # end of model
        return -softmax_cross_entropy_with_logits(logits, jax.lax.stop_gradient(y))