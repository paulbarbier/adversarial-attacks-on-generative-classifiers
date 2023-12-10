from flax import linen as nn
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform
import numpy as np

class Log_p_x_z(nn.Module):
    n_classes: int
    d_latent: int
    d_hidden: int
    dropout_rate: float

    n_channels = 64
    input_kernel = (4, 4)
    kernel_size = (5, 5)
    strides = (2, 2)

    @nn.compact
    def __call__(self, X, z, train: bool = False): # X: (height, width), z: (d_latent,) -> 0
        inputs = z
        inputs = nn.Dense(
            features=self.d_hidden, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(inputs)
        inputs = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(inputs)
        inputs = nn.relu(inputs)
        inputs = nn.Dense(
            features=np.prod(self.input_kernel) * self.n_channels, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(inputs)
        inputs = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(inputs)
        inputs = nn.relu(inputs)
        
        inputs = inputs.reshape(self.input_kernel + (self.n_channels,))
        inputs = nn.ConvTranspose(
            features=self.n_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=(2, 2),
            kernel_init=glorot_uniform(),
        )(inputs)
        inputs = nn.relu(inputs)

        inputs = nn.ConvTranspose(
            features=self.n_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=((2, 3), (2, 3)),
            kernel_init=glorot_uniform(),
        )(inputs)
        inputs = nn.relu(inputs)

        inputs = nn.ConvTranspose(
            features=1,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=((2, 3), (2, 3)),
            kernel_init=glorot_uniform(),
        )(inputs)
        generated_X = nn.sigmoid(inputs)

        logit = -jnp.sum((generated_X - X)**2)
        return logit