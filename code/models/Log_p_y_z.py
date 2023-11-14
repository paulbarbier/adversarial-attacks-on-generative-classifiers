from flax import linen as nn
from jax.nn.initializers import glorot_uniform
from objax.functional.loss import cross_entropy_logits

class Log_p_y_z(nn.Module):
    d_hidden: int
    n_classes: int
    d_latent: int
    dropout_rate: float

    @nn.compact
    def __call__(self, y, z, train: bool = False): # y: (n_classes,), z: (d_latent,) -> 0
        logits = nn.Dense(
            features=self.d_hidden, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(z)
        logits = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(logits)
        logits = nn.relu(logits)
        logits = nn.Dense(
            features=self.n_classes, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(logits)
        return -cross_entropy_logits(logits, y)