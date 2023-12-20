from functools import partial
import jax
from jax.scipy.special import logsumexp
from flax import linen as nn
import jax.numpy as jnp
import numpy as np
from models.utils import sample_gaussian


def corrupt_batch(key, model, attack_config, X, y_true):
    classifier, model_config, params, log_likelihood, K = model
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
        return r

    X_corrupted = jnp.copy(X)
    perturbation = jnp.zeros_like(X_corrupted)
    epsilon_shape = (model_config.n_classes * K, model_config.d_latent)
    
    key, y_corrupted = classifier.make_predictions(
        key, model_config, params, log_likelihood, X_corrupted
    )
    target_indices = y_corrupted == y_true 
    
    iteration = 0
    while jnp.any(target_indices) and iteration < attack_config.max_iter:
        X_targeted = X_corrupted[target_indices]
        key, epsilon = sample_gaussian(key, (X_targeted.shape[0],) + epsilon_shape, dtype)
        perturbation = pertubation_step(X_targeted, epsilon)
        X_corrupted = X_corrupted.at[target_indices].set(
            X_targeted + attack_config.learning_rate * perturbation
        )
        
        key, y_corrupted = classifier.make_predictions(
            key, model_config, params, log_likelihood, X_corrupted
        )
        target_indices = y_corrupted == y_true
        iteration += 1
    
    return key, X_corrupted