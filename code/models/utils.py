import jax.numpy as jnp
import numpy as np
import jax.random as random

def log_gaussian(x, mu=0.0, logsigma=0.0):
    delta = ((x - mu) / jnp.exp(logsigma))**2
    logits = -(0.5*np.log(2*np.pi) + logsigma + 0.5*delta)
    return jnp.sum(logits)

def transform(epsilon, mu, log_sigma):
    sigma = jnp.exp(log_sigma)
    z = mu + sigma * epsilon
    return z

def sample_gaussian(key, shape: tuple) -> np.ndarray:
    key, sample_key = random.split(key)
    epsilon = random.normal(sample_key, shape, dtype=jnp.float32)
    return key, epsilon