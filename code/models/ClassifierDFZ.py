import jax
from jax.typing import DTypeLike
from flax import struct
from flax import linen as nn
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import random
from ml_collections import ConfigDict
from jax.scipy.special import logsumexp
from flax.training.train_state import TrainState

from models.utils import log_gaussian, sample_gaussian

from models.LogQZ_XY import LogQZ_XY, QZ_XYConfiguration
from models.LogPX_Z import LogPX_Z, PX_ZConfiguration
from models.LogPY_XZ import LogPY_XZ, PY_XZConfiguration

from typing import Tuple

@struct.dataclass
class DFZConfiguration:
    image_shape: Tuple[int, int, int]
    n_classes: int
    d_latent: int
    K: int

    qz_xy: QZ_XYConfiguration
    px_z: PX_ZConfiguration
    py_xz: PY_XZConfiguration

def create_model_config(config: ConfigDict) -> DFZConfiguration:
    model_config = DFZConfiguration(
        image_shape=(config.image_height, config.image_width, config.image_channels),
        n_classes=config.n_classes,
        d_latent=config.model.d_latent,
        K=config.model.K,

        qz_xy=QZ_XYConfiguration(
            n_classes=config.n_classes,
            d_latent=config.model.d_latent,
            d_hidden=config.model.d_hidden,
            dropout_rate=config.model.dropout_rate,
        ),

        px_z=PX_ZConfiguration(
            n_classes=config.n_classes,
            d_latent=config.model.d_latent,
            d_hidden=config.model.d_hidden,
            dropout_rate=config.model.dropout_rate,
        ),
        
        py_xz=PY_XZConfiguration(
            n_classes=config.n_classes,
            d_latent=config.model.d_latent,
            d_hidden=config.model.d_hidden,
            dropout_rate=config.model.dropout_rate,
        ),
    )
    return model_config

# Base class that bundles all the sub-modules of the classifier
class ClassifierDFZ(nn.Module):
    config: DFZConfiguration

    @nn.compact
    def __call__(self, X, y, epsilon, train: bool = False): # X: (height, width), y: (n_classes,), epsilon: (d_latent,) -> 1, 1
        config = self.config
        z, logit_q_z_xy = LogQZ_XY(config.qz_xy)(X, y, epsilon, train)
        logit_p_x_z = LogPX_Z(config.px_z)(X, z, train)
        logit_p_y_xz = LogPY_XZ(config.py_xz)(X, y, z, train)
        return z, logit_q_z_xy, logit_p_x_z, logit_p_y_xz

# create an instance of Classifier and init the params
def init_params(key, config: DFZConfiguration, dtype: DTypeLike = jnp.float32):
    X = jnp.ones(config.image_shape, dtype=dtype)
    y = jnp.ones(config.n_classes, dtype=dtype)
    epsilon = jnp.ones(config.d_latent, dtype=dtype)

    key, subkey = random.split(key)
    params = ClassifierDFZ(config).init(subkey, X, y, epsilon)["params"]

    return key, params

def create_training_state(config: DFZConfiguration, init_params, optimiser) -> TrainState:
    return TrainState.create(
        apply_fn=partial(ClassifierDFZ(config).apply, train=True),
        params=init_params,
        tx=optimiser,
    )

# perform a single training step:
# * compute the batch_loss
# * compute grad
# * update parameter according to defined optimiser
@partial(jax.jit, static_argnames=['loss_single'])
def training_step(state: TrainState, X_batch, y_batch, epsilon, loss_single, dropout_key):
  # fold in a new dropout key
  dropout_key = jax.random.fold_in(dropout_key, state.step)

  def batch_loss(params):
    def loss_fn(X, y, epsilon):
      z, logit_q_z_xy, logit_p_x_z, logit_p_y_xz = state.apply_fn(
        {'params': params},
        X, y, epsilon,
        rngs={"dropout": dropout_key},
      )
      return loss_single(z, logit_q_z_xy, logit_p_x_z, logit_p_y_xz)

    loss = jax.vmap(loss_fn)(X_batch, y_batch, epsilon)
    return jnp.mean(loss)

  loss, grads = jax.value_and_grad(
    batch_loss,
  )(state.params)

  new_state = state.apply_gradients(
      grads=grads
  )
  return new_state, loss

# make batch prediction of X using the ll function and the sampling parameter K
def make_predictions(key, config: DFZConfiguration, params, X, log_likelihood, K = None): # X: (batch_size, image_size, image_size, 1)
    if K is None:
        K = config.K
    batch_size = X.shape[0]
    key, epsilon = sample_gaussian(key, (batch_size, config.n_classes * K, config.d_latent))
    y = nn.one_hot(jnp.repeat(jnp.arange(config.n_classes), K), config.n_classes, dtype=X.dtype)

    @jax.jit
    def make_single_prediction(x, epsilon):
        z, logit_q_z_xy, logit_p_x_z, logit_p_y_xz = jax.vmap(
            partial(ClassifierDFZ(config).apply, {'params': params}, train=False),
            in_axes=(None, 0, 0)
        )(x, y, epsilon)

        ll = log_likelihood(
            z, logit_q_z_xy, logit_p_x_z, logit_p_y_xz
        ).reshape(config.n_classes, K)

        p_bayes = nn.softmax(logsumexp(ll, axis=1) - np.log(K))
        y_prediction = jnp.argmax(p_bayes)
        return y_prediction
    
    y_predictions = jax.vmap(make_single_prediction)(X, epsilon)
    return key, y_predictions

# ll stands for log-likelihood
def loss_A_single(z, logit_q_z_xy, logit_p_x_z, logit_p_y_xz):
    logit_prior_z = log_gaussian(z)
    ll = logit_p_x_z + logit_p_y_xz + logit_prior_z - logit_q_z_xy
    return -ll
# define batch loss
loss_A = jax.vmap(loss_A_single)

def compute_batch_loss(key, config: DFZConfiguration, params, X_batch, y_batch, loss_fn):
    key, epsilon = sample_gaussian(key, (X_batch.shape[0], config.d_latent))
    z, logit_q_z_xy, logit_p_x_z, logit_p_y_xz = jax.vmap(
        partial(ClassifierDFZ(config).apply, {'params': params}, train=False),
        in_axes=(0, 0, 0)
    )(X_batch, y_batch, epsilon)
    loss_value = jnp.mean(
        loss_fn(z, logit_q_z_xy, logit_p_x_z, logit_p_y_xz)
    )
    return key, loss_value

def compute_batch_accuracy(key, model, params, X_batch, y_batch, log_likelihood, K = None):
    key, y_predictions = make_predictions(key, model, params, X_batch, log_likelihood, K)
    labels = jnp.argmax(y_batch, axis=1)
    accuracy = 100.0 * jnp.mean(y_predictions == labels)
    return key, accuracy

def log_likelihood_A(z, logit_q_z_xy, logit_p_x_z, logit_p_y_xz):
        return -loss_A(z, logit_q_z_xy, logit_p_x_z, logit_p_y_xz)