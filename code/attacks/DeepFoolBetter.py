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

    y_probe = nn.one_hot(jnp.repeat(jnp.arange(model_config.n_classes, dtype=dtype), K), model_config.n_classes)

    @jax.jit 
    def compute_single_ll(x, epsilon):
        logits = jax.vmap(
            partial(classifier.classifier(model_config).apply, {'params': params}, train=False),
            in_axes=(None, 0, 0)
        )(x, y_probe, epsilon)

        ll = log_likelihood(*logits).reshape(model_config.n_classes, K)
        ll = logsumexp(ll, axis=1) - np.log(K)
        return ll
    
    @jax.jit
    @jax.vmap
    def pertubation_step(x, epsilon):
        ll = compute_single_ll(x, epsilon)
        y = jnp.argmax(ll)
        grads = jax.jacrev(
            compute_single_ll, argnums=0
        )(x, epsilon)
        w = grads - grads[None, y, ...]
        f = ll - ll[None, y]
        w_norm = jnp.linalg.norm(w.squeeze(), axis=(1, 2))
        perturbation = jnp.abs(f) / w_norm
        perturbation = perturbation.at[y].set(jnp.inf)
        idx = jnp.argmin(perturbation)
        r = perturbation[idx] * w[idx] / w_norm[idx]
        return attack_config.learning_rate * r
    
    # compute predictions to prevent label leakage
    key, labels = classifier.make_predictions(
        key, model_config, params, log_likelihood, X
    )
    
    @jax.jit
    def perturbate(indices, X, epsilon):
        perturbation = pertubation_step(X, epsilon[0])
        X = jnp.where(indices[:, None, None, None], X + perturbation, 0.0)

        y_corrupted = classifier.make_deterministic_predictions(
            model_config, params, log_likelihood, X, y_probe, epsilon[1]
        )

        indices = indices * (y_corrupted == labels)
        return X, indices

    X_corrupted = jnp.copy(X)
    epsilon_shape = (model_config.n_classes * K, model_config.d_latent)

    target_indices = jnp.ones_like(labels, dtype=bool)  
    iteration = 0
    while jnp.any(target_indices) and iteration < attack_config.max_iter:
        key, epsilon = sample_gaussian(key, (2, X.shape[0]) + epsilon_shape, dtype)
        X_corrupted, target_indices = perturbate(target_indices, X_corrupted, epsilon)
        iteration += 1
    
    return key, X_corrupted