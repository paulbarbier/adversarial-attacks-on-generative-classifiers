from flax import linen as nn
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform
import numpy as np

class Log_p_x_yz(nn.Module):
    d_hidden = 500
    d_epsilon = 64
    n_classes = 10
    n_channels = 64
    input_kernel = (4, 4)
    kernel_size = (5, 5)
    strides = (2, 2)
    

    @nn.compact
    def __call__(self, X, y, z): # X: (height, width), y: (n_classes,), z: (d_epsilon,) -> 0
        inputs = jnp.concatenate([y, z], 0)
        inputs = nn.Dense(
            features=self.d_hidden, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(inputs)
        inputs = nn.relu(inputs)
        inputs = nn.Dense(
            features=np.prod(self.input_kernel) * self.n_channels, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(inputs)
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