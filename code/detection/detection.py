from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.nn as nn


def compute_logits_stats(key, train_dl, model_data):
    classifier, model_config, params, log_likelihood_fn, loss_single_fn, K = model_data

    logits = []
    with tqdm(train_dl, unit="batch") as ttrain:
        ttrain.set_description(f"Prepare threshold detection")
        for train_step, (X_batch, y_batch) in enumerate(ttrain):
            key, logits_batch = classifier.compute_logits(
                key, model_config, params, log_likelihood_fn, X_batch, K
            )
            logits.append(logits_batch)
    logits = jnp.concatenate(logits, axis=0)
    predictions = jnp.argmax(logits, axis=1)
    logits = jnp.max(logits, axis=1)

    mean = jnp.empty(model_config.n_classes, logits.dtype)
    std = jnp.empty(model_config.n_classes, logits.dtype)

    for label in range(model_config.n_classes):
        indices = predictions == label
        mean = mean.at[label].set(jnp.mean(logits, where=indices))
        std = std.at[label].set(jnp.std(logits, where=indices))

    return key, mean, std

def compute_thresholds(key, train_dl, model_data, detection_config):
    key, mean, std = compute_logits_stats(key, train_dl, model_data)

    print(f"mean: \n{mean}\n std: \n{std}")

    alpha = jnp.linspace(-detection_config.alpha, detection_config.alpha, detection_config.num_thresholds)
    threshold = mean[None, :] - alpha[:, None] * std[None, :]
    return key, threshold

#@jax.jit
def detect_attack(thresholds, logits, y_true = None):
    def single_detection(threshold, logits):
        label = jnp.argmax(logits)
        return logits[label] < threshold[label]
    
    detect_batch_attack = jax.vmap(single_detection, in_axes=(None, 0))
    detections = jax.vmap(
        detect_batch_attack,
        in_axes=(0, None)
    )(thresholds, logits)

    if y_true is None:
        return jnp.mean(detections, axis=1)
    else:
        indices = y_true == jnp.argmax(logits, axis=1)
        return jnp.mean(detections, where=indices, axis=1)
