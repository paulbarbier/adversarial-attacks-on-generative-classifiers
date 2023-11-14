import jax
from flax import linen as nn
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import random
from ml_collections import ConfigDict
from jax.scipy.special import logsumexp
from flax.training.train_state import TrainState

from models.utils import log_gaussian, sample_gaussian

from models.Log_q_z_xy import Log_q_z_xy
from models.Log_p_x_yz import Log_p_x_yz
from models.Log_p_y_z import Log_p_y_z


# Base class that bundles all the sub-modules of the classifier
class ClassifierGFZ(nn.Module):
    n_classes: int
    d_latent: int
    d_hidden: int
    K: int

    @nn.compact
    def __call__(self, X, y, epsilon): # X: (height, width), y: (n_classes,), epsilon: (d_latent,) -> 1, 1
        z, logit_q_z_xy = Log_q_z_xy(
            n_classes=self.n_classes,
            d_latent=self.d_latent,
            d_hidden=self.d_hidden,
        )(X, y, epsilon)
        logit_p_x_yz = Log_p_x_yz(
            n_classes=self.n_classes,
            d_latent=self.d_latent,
            d_hidden=self.d_hidden,
        )(X, y, z)
        logit_p_y_z = Log_p_y_z(
            n_classes=self.n_classes,
            d_latent=self.d_latent,
            d_hidden=self.d_hidden,
        )(y, z)
        return z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z

# create an instance of Classifier and init the parameters
def create_and_init(key, config: ConfigDict, dataset_config: ConfigDict) -> ClassifierGFZ:
    model = ClassifierGFZ(
       **config.model,
       n_classes=dataset_config.n_classes,
    )

    X = jnp.ones(dataset_config.image_shape, dtype=jnp.float32)
    y = jnp.ones(dataset_config.n_classes, dtype=jnp.float32)
    
    epsilon = jnp.ones(config.model.d_latent, dtype=jnp.float32)

    key, subkey = random.split(key)
    params = model.init(subkey, X, y, epsilon)["params"]

    return key, model, params

# perform a single training step:
# * compute the batch_loss
# * compute grad
# * update parameter according to defined optimiser
@partial(jax.jit, static_argnames=['loss_single'])
def training_step(state: TrainState, X_batch, y_batch, epsilon, loss_single):
  def batch_loss(params):
    def loss_fn(X, y, epsilon):
      z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z = state.apply_fn(
        {'params': params},
        X, y, epsilon
      )
      return loss_single(z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z)

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
def make_predictions(key, model, params, X, log_likelyhood, K = None): # X: (batch_size, image_size, image_size, 1)
    if K is None:
        K = model.K
    batch_size = X.shape[0]
    key, epsilon = sample_gaussian(key, (batch_size, model.n_classes * K, model.d_latent))
    y = nn.one_hot(jnp.repeat(jnp.arange(model.n_classes), K), model.n_classes, dtype=jnp.float32)

    @jax.jit
    def make_single_prediction(x, epsilon):
        z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z = jax.vmap(
            partial(model.apply, {'params': params}),
            in_axes=(None, 0, 0)
        )(x, y, epsilon)

        ll = log_likelyhood(
            z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z
        ).reshape(model.n_classes, K)

        p_bayes = nn.softmax(logsumexp(ll, axis=1) - np.log(K))
        y_prediction = jnp.argmax(p_bayes)
        return y_prediction
    
    y_predictions = jax.vmap(make_single_prediction)(X, epsilon)
    return key, y_predictions

# ll stands for log-likelyhood
def loss_A_single(z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z):
    logit_prior_z = log_gaussian(z)
    ll = logit_p_x_yz + logit_p_y_z + logit_prior_z - logit_q_z_xy
    return -ll
# define batch loss
loss_A = jax.vmap(loss_A_single)

def compute_batch_loss(key, model, params, X_batch, y_batch, loss_fn):
    key, epsilon = sample_gaussian(key, (X_batch.shape[0], model.d_latent))
    z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z = jax.vmap(
        model.apply, 
        in_axes=(None, 0, 0, 0)
    )({"params": params}, X_batch, y_batch, epsilon)
    loss_value = jnp.mean(
        loss_fn(z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z)
    )
    return key, loss_value

def compute_batch_accuracy(key, model, params, X_batch, y_batch, log_likelyhood, K = None):
    key, y_predictions = make_predictions(key, model, params, X_batch, log_likelyhood, K)
    labels = jnp.argmax(y_batch, axis=1)
    accuracy = 100.0 * jnp.mean(y_predictions == labels)
    return key, accuracy

def log_likelyhood_A(z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z):
        return -loss_A(z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z)