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

from detection import detection

from absl import app
from absl import flags
from pathlib import Path

CHECKPOINT_DIR = Path.cwd() / Path("checkpoints")

def attack_and_detect(flags):
  checkpoint = load_checkpoint(Path.cwd() / Path(flags.checkpoint))

  config = ConfigDict(checkpoint["config"])

  attack_config = flags.attack_config
  detection_config = flags.detection_config

  if flags.dtype == "":
    dtype = get_dtype(config.dtype)
  else:
    dtype = get_dtype(flags.dtype)

  K = flags.K
  if K < 0:
    K = config.model.K

  classifier = get_classifier(config) 

  train_ds = get_dataset(config.dataset, train=True, dtype=dtype)
  train_dl = get_dataloader(train_ds, config.train_batch_size, shuffle=False)
  eval_ds = get_dataset(config.dataset, train=False, dtype=dtype)
  eval_dl = get_dataloader(eval_ds, attack_config.attack_batch_size, shuffle=False)

  model_config = classifier.create_model_config(config)
  log_likelihood_fn = classifier.log_likelihood_A
  loss_single_fn = classifier.loss_A_single
  params = checkpoint["params"]

  model_data = (classifier, model_config, params, log_likelihood_fn, loss_single_fn, K)

  attack_key = random.PRNGKey(config.attack_seed)
  attack_key, test_key = random.split(attack_key)

  #sink for the metric values
  metrics = {
    "false_positive_rates": [],
    "true_positive_rates": [],
    "detection_rate_at_5_pc": [],
    "pertubation_norms": [],
    "attack_success_rate": [],
  }

  attack_model = get_attack_model(flags.attack_config)

  ## detection preparation
  attack_key, thresholds = detection.compute_thresholds(attack_key, train_dl, model_data, detection_config)
  metrics["thresholds"] = thresholds

  with tqdm(eval_dl, unit="batch") as tattack:
    tattack.set_description(f"Attack {flags.attack_config.attack_name} with {flags.detection_config.name} detection")
    for attack_step, (X_batch, y_batch) in enumerate(tattack):
      if flags.debug and attack_step == 2:
        break

      test_key, original_logits = classifier.compute_logits(
        test_key, model_config, params, log_likelihood_fn, X_batch
      )

      false_positives_batch = detection.detect_attack(
        thresholds, original_logits, y_batch
      )
      metrics["false_positive_rates"].append(false_positives_batch)

      attack_key, X_corrupted_batch = attack_model.corrupt_batch(
        attack_key, 
        model_data,
        attack_config,
        X_batch, 
        y_batch
      )

      # compute attacked batch prediction
      test_key, attacked_logits = classifier.compute_logits(
        test_key, model_config, params, log_likelihood_fn, X_corrupted_batch
      )

      true_positives_batch = detection.detect_attack(
        thresholds, attacked_logits
      )
      metrics["true_positive_rates"].append(true_positives_batch)

      metrics["detection_rate_at_5_pc"].append(
        jnp.max(true_positives_batch[false_positives_batch < 0.05])
      )
      
      corrupted_indices = jnp.argmax(attacked_logits, axis=1) != y_batch
      X_corrupted_success = jnp.take(X_corrupted_batch, corrupted_indices, axis=0)
      X_batch_success = jnp.take(X_batch, corrupted_indices, axis=0)

      metrics["pertubation_norms"].append(
        jnp.mean(perturbation_norm(X_corrupted_success, X_batch_success))
      )
      metrics["attack_success_rate"].append(
        jnp.mean(corrupted_indices)
      )
      tattack.set_postfix(
        detection_rate_at_5_pc=metrics["detection_rate_at_5_pc"][-1],
        attack_success_rate=np.mean(metrics["attack_success_rate"]),
        perturbation_norm=np.mean(metrics["pertubation_norms"]),
      )

  metrics["true_positive_rates"] = jnp.stack(metrics["true_positive_rates"], axis=1)
  metrics["false_positive_rates"] = jnp.stack(metrics["false_positive_rates"], axis=1)
  metrics["true_positive_rates"] = jnp.mean(metrics["true_positive_rates"], axis=1)
  metrics["false_positive_rates"] = jnp.mean(metrics["false_positive_rates"], axis=1) 

  metrics["pertubation_norms"] = np.mean(metrics["pertubation_norms"])
  metrics["attack_success_rate"] = np.mean(metrics["attack_success_rate"])

  keys = ["pertubation_norms", "attack_success_rate", "detection_rate_at_5_pc"]
  print("Detection results:")
  for key in keys:
    print(f"{key}: {metrics[key]}")

  checkpointer = ocp.PyTreeCheckpointer()
  checkpoint_name = flags.checkpoint_name
  if checkpoint_name == "":
    checkpoint_name = f"{config.checkpoint_name}-attack-{attack_config.checkpoint_name}-{detection_config.name}"

  checkpointer.save(
    CHECKPOINT_DIR / checkpoint_name, 
    {
      "metrics": metrics,
      "attack_config": attack_config.to_dict(),
      "detection_config": detection_config.to_dict(),
    }
  )

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'attack_config',
    None,
    'path to attack configuration file.',
    lock_config=True,
)

config_flags.DEFINE_config_file(
    'detection_config',
    "configs/logit_detection.py",
    'path to detection configuration file.',
    lock_config=True,
)

flags.DEFINE_string("checkpoint", "", "Checkpoint relative path.")
flags.DEFINE_string("checkpoint_name", "", "folder under checkpoints to save the experiment")
flags.DEFINE_string("dtype", "", "dtype")
flags.DEFINE_bool("debug", False, "debug flag")
flags.DEFINE_integer("K", -1, "number of samples to use for importance sampling")

def main(argv):
  attack_and_detect(FLAGS)

if __name__ == '__main__':
  app.run(main)