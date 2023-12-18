from flax import struct
from flax import linen as nn
from jax.nn.initializers import glorot_uniform

from models.utils import softmax_cross_entropy_with_logits

@struct.dataclass
class PY_ZConfiguration:
    d_hidden: int
    n_classes: int
    d_latent: int
    dropout_rate: float

class LogPY_Z(nn.Module):
    config: PY_ZConfiguration

    @nn.compact
    def __call__(self, y, z, train: bool = False): # y: (n_classes,), z: (d_latent,) -> 0
        config = self.config
        logits = nn.Dense(
            features=config.d_hidden, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(z)
        logits = nn.Dropout(rate=config.dropout_rate, deterministic=not train)(logits)
        logits = nn.relu(logits)
        logits = nn.Dense(
            features=config.n_classes, 
            use_bias=True,
            kernel_init=glorot_uniform(), 
        )(logits)
        return -softmax_cross_entropy_with_logits(logits, y)