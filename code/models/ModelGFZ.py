import jax
from flax import linen as nn
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import random
from ml_collections import ConfigDict
from jax.scipy.special import logsumexp

from .utils import sample_p

from .Log_q_z_xy import Log_q_z_xy
from .Log_p_x_yz import Log_p_x_yz
from .Log_p_y_z import Log_p_y_z

class ModelGFZ(nn.Module):
    d_epsilon = 64
    n_classes = 10

    @nn.compact
    def __call__(self, X, y, epsilon): # X: (height, width), y: (n_classes,), epsilon: (d_epsilon,) -> 1, 1
        z, logit_q_z_xy = Log_q_z_xy()(X, y, epsilon)
        logit_p_x_yz = Log_p_x_yz()(X, y, z)
        logit_p_y_z = Log_p_y_z()(y, z)
        return z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z
    
def init_model(key, config: ConfigDict, dataset_config: ConfigDict) -> ModelGFZ:
    model =  ModelGFZ()

    X = jnp.ones(dataset_config.image_shape, dtype=jnp.float32)
    y = jnp.ones(dataset_config.n_classes, dtype=jnp.float32)
    epsilon = jnp.ones(config.d_epsilon, dtype=jnp.float32)

    key, subkey = random.split(key)

    params = model.init(subkey, X, y, epsilon)["params"]

    return key, model, params

@partial(jax.jit, static_argnames=['loss_single'])
def update_step(state, X_batch, y_batch, epsilon, loss_single):
  def batch_loss(params):
    def loss_fn(X, y, epsilon):
      z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z = state.apply_fn(
        {'params': params},
        X, y, epsilon
      )
      return loss_single(z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z)

    loss = jax.vmap(
      loss_fn, 
      in_axes=(0, 0, 0), 
    )(X_batch, y_batch, epsilon)
    return jnp.mean(loss)

  loss, grads = jax.value_and_grad(
    batch_loss,
  )(state.params)

  new_state = state.apply_gradients(
      grads=grads
  )
  return new_state, loss

def make_predictions(key, model, params, X, log_likelyhood, K=10): # X: (batch_size, image_size, image_size, 1)
    batch_size = X.shape[0]
    key, epsilon = sample_p(key, (batch_size, model.n_classes * K, model.d_epsilon))
    y = nn.one_hot(jnp.repeat(jnp.arange(model.n_classes), K), model.n_classes, dtype=jnp.float32)

    @jax.jit
    def make_single_prediction(x, epsilon):
        z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z = jax.vmap(
            partial(model.apply, {'params': params}),
            in_axes=(None, 0, 0)
        )(x, y, epsilon)

        ll = log_likelyhood(z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z).reshape(model.n_classes, K)
        p_bayes = nn.softmax(logsumexp(ll, axis=1) - np.log(K))
        y_prediction = jnp.argmax(p_bayes)
        return y_prediction
    
    y_predictions = jax.vmap(make_single_prediction)(X, epsilon)
    return key, y_predictions