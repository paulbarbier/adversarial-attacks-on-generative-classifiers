from functools import partial
import jax
from jax.scipy.special import logsumexp
from flax import linen as nn
import jax.numpy as jnp
import numpy as np
from models.utils import sample_gaussian


def corrupt_batch(key, model, attack_config, X, y_true):
    classifier, model_config, params, log_likelihood, K = model
    
    batch_size = X.shape[0]
    key, epsilon = sample_gaussian(key, (batch_size, model_config.n_classes * K, model_config.d_latent))
    y_probe = nn.one_hot(jnp.repeat(jnp.arange(model_config.n_classes), K), model_config.n_classes, dtype=X.dtype)

    @jax.jit
    def compute_single_ll(x, epsilon):
        outputs = jax.vmap(
            partial(classifier.classifier(model_config).apply, {'params': params}, train=False),
            in_axes=(None, 0, 0)
        )(x, y_probe, epsilon)

        ll = log_likelihood(*outputs).reshape(model_config.n_classes, K)
        ll = logsumexp(ll, axis=1) - np.log(K)
        return ll
    
    @jax.jit
    @jax.vmap
    def pertubation_step(x, epsilon, label):
        log_likelihoods = compute_single_ll(x, epsilon)
        gradients = jax.jacrev(
            compute_single_ll, argnums=0
        )(x, epsilon)
        w = gradients - gradients[None,label,...]
        f = log_likelihoods - log_likelihoods[None,label]
        perturbation = jnp.abs(f) / jnp.linalg.norm(w.squeeze(), axis=(1, 2))
        perturbation = perturbation.at[label].set(jnp.inf)
        idx = jnp.argmin(perturbation)
        r_i = perturbation[idx] * w[idx] / jnp.linalg.norm(w[idx])
        return attack_config.learning_rate * r_i
    
    make_predictions = lambda X: classifier.make_deterministic_predictions(
        model_config, params, log_likelihood, X, y_probe, epsilon, K=K
    )
    X_corrupted = X.copy()
    y_corrupted = make_predictions(X_corrupted)
    target_indices = y_corrupted == y_true

    iter = 0
    while jnp.sum(target_indices) > 0 and iter < attack_config.max_iter:
        perturbation = pertubation_step(X_corrupted, epsilon, y_true)
        X_corrupted += jnp.where(target_indices[:,None,None,None], perturbation, 0.0)
        y_corrupted = make_predictions(X_corrupted)
        target_indices = y_corrupted == y_true
        iter += 1
    
    return key, X_corrupted