from jax import random
from ml_collections import ConfigDict, config_flags
from tqdm import tqdm

from dataset_utils import get_dataloader, get_dataset
from utils import get_attack_model, get_classifier, get_dtype, load_checkpoint, perturbation_norm
from jax import random
import jax.numpy as jnp
import numpy as np
from jax.nn import one_hot
import orbax.checkpoint as ocp

from absl import app
from absl import flags
from pathlib import Path

def attack(flags):
  checkpoint = load_checkpoint(Path.cwd() / Path(flags.checkpoint))

  config = ConfigDict(checkpoint["config"])

  if flags.dtype == "":
    dtype = get_dtype(config.dtype)
  else:
    dtype = get_dtype(flags.dtype)

  K = flags.K
  if K < 0:
    K = config.model.K

  classifier = get_classifier(config) 

  eval_ds = get_dataset(config.dataset, train=False)
  eval_dl = get_dataloader(eval_ds, config.test_batch_size, dtype)

  model_config = classifier.create_model_config(config)
  log_likelihood_fn = classifier.log_likelihood_A
  params = checkpoint["params"]

  model_data = (classifier, model_config, params, log_likelihood_fn, K)

  attack_key = random.PRNGKey(config.attack_seed)
  attack_key, test_key = random.split(attack_key)

  #sink for the metric values
  metrics = {
    "labels": [],
    "predictions": [],
    "corrupted_images": [],
    "pertubation_norms": [],
    "attack_success_rate": [],
  }

  attack_model = get_attack_model(flags.config)

  with tqdm(eval_dl, unit="batch") as tattack:
    tattack.set_description(f"Attack {flags.type}")
    for attack_step, (X_batch, y_batch) in enumerate(tattack):
      if flags.debug and attack_step == 1:
        break

      attack_key, X_corrupted_batch = attack_model.corrupt_batch(
        attack_key, 
        model_data,
        flags.config, # attack config
        X_batch, 
        y_batch
      )

      # compute batch prediction
      test_key, y_pred_batch = classifier.make_predictions(
        test_key, model_config, params, X_batch, log_likelihood_fn
      )

      metrics["corrupted_images"].append(X_corrupted_batch)
      metrics["labels"].append(y_batch)
      metrics["predictions"].append(y_pred_batch)

      corrupted_indices = y_pred_batch != y_batch
      X_corrupted_success = jnp.take(X_corrupted_batch, corrupted_indices, axis=0)
      X_batch_success = jnp.take(X_batch, corrupted_indices, axis=0)

      metrics["pertubation_norms"].append(
        perturbation_norm(X_corrupted_success, X_batch_success)
      )
      metrics["attack_success_rate"].append(
        jnp.mean(corrupted_indices)
      )
      tattack.set_postfix(
        success_rate=metrics["attack_success_rate"][-1],
        mean_perturbation_norm=jnp.mean(metrics["pertubation_norms"][-1]),
      )

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'path to attack configuration file.',
    lock_config=True,
)

flags.DEFINE_string("checkpoint", "", "Checkpoint relative path.")
flags.DEFINE_string("dtype", "", "dtype")
flags.DEFINE_string("type", "deepfool", "Attack type (deepfool, ...)")
flags.DEFINE_bool("debug", False, "debug flag")
flags.DEFINE_integer("K", -1, "number of samples to use for importance sampling")

def main(argv):
  attack(FLAGS)

if __name__ == '__main__':
  app.run(main)