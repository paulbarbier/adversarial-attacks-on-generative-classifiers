from functools import partial
import jax
from jax import random
from jax.scipy.special import logsumexp
from flax import linen as nn
import jax.numpy as jnp
import numpy as np
from models.utils import sample_gaussian


def corrupt_batch(key, model, attack_config, X, y_true):
    classifier, model_config, params, log_likelihood, loss_single, K = model
    dtype = X.dtype

    key, labels = classifier.make_predictions(
        key, model_config, params, log_likelihood, X
    )
    y = nn.one_hot(labels, model_config.n_classes, dtype=dtype)

    @jax.jit 
    def compute_single_loss(x, y, epsilon):
        logits = classifier.classifier(model_config).apply(
            {'params': params}, 
            x, y, epsilon, 
            train=False
        )
        loss = loss_single(*logits)
        return loss

    @jax.jit
    @jax.vmap
    def pertubate(x, y, epsilon):
        grads = jax.grad(compute_single_loss)(x, y, epsilon)
        pertubated = x + attack_config.eta * jnp.sign(grads)
        pertubated = jnp.clip(pertubated, 0.0, 1.0)
        return pertubated

    epsilon_shape = (X.shape[0], model_config.d_latent)
    key, epsilon = sample_gaussian(key, epsilon_shape) 

    X_corrupted = pertubate(X, y, epsilon)
    
    return key, X_corrupted