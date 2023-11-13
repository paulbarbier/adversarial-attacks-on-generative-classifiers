from .ModelGFZ import make_predictions
from models.utils import log_gaussian, sample_p
from jax import vmap
import jax.numpy as jnp

# ll stands for log-likelyhood
def loss_A_single(z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z):
    logit_p_z = log_gaussian(z)
    ll = logit_p_x_yz + logit_p_y_z + (logit_p_z - logit_q_z_xy)
    return -ll

loss_A = vmap(loss_A_single)

def compute_batch_loss(key, model, params, X_batch, y_batch, loss):
    key, epsilon = sample_p(key, (X_batch.shape[0], model.d_epsilon))
    z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z = vmap(
        model.apply, 
        in_axes=(None, 0, 0, 0)
    )({"params": params}, X_batch, y_batch, epsilon)
    loss_value = jnp.mean(
        loss(z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z)
    )
    return key, loss_value

def compute_batch_accuracy(key, model, params, X_batch, y_batch, log_likelyhood, K=10):
    key, y_predictions = make_predictions(key, model, params, X_batch, log_likelyhood, K)
    labels = jnp.argmax(y_batch, axis=1)
    accuracy = 100.0 * jnp.mean(y_predictions == labels)
    return key, accuracy