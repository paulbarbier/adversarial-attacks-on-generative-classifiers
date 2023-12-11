import orbax.checkpoint as ocp
from jax import random
from ml_collections import ConfigDict, config_flags

from models.utils import sample_gaussian
from dataset_utils import get_dataset
from utils import prepare_test_dataset
from jax import random
from jax.nn import one_hot
import orbax.checkpoint as ocp

from absl import app
from absl import flags
from pathlib import Path

import models.ClassifierGFZ as ClassifierGFZ
import models.ClassifierDFZ as ClassifierDFZ

def evaluate(checkpoint_path: str):
  path = Path.cwd() / Path(f"checkpoints") / Path(checkpoint_path)
  checkpoint = ocp.PyTreeCheckpointer().restore(path, item=None)

  config = ConfigDict(checkpoint["config"])
  dataset_config = ConfigDict(checkpoint["dataset_config"])

  if config.model_name == "GFZ":
    classifier = ClassifierGFZ
  elif config.model_name == "DFZ":
    classifier = ClassifierDFZ
  else:
    raise NotImplementedError(config.model_name)

  _, test_ds = get_dataset(config.dataset)
  test_images, test_labels = prepare_test_dataset(
    test_ds, dataset_config
  )

  trained_params = checkpoint["params"]

  log_likelyhood_fn = classifier.log_likelyhood_A

  test_key = random.PRNGKey(config.seed)

  test_key, model, _ = classifier.create_and_init(
    test_key, config, dataset_config
  )

  test_key, test_accuracy_value = classifier.compute_batch_accuracy(
    test_key, 
    model, 
    trained_params, 
    test_images[:100], 
    test_labels[:100], 
    log_likelyhood_fn,
  )
  print("\nAccuracy:", test_accuracy_value)


FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", "", "Checkpoint relative path.")

def main(argv):
  evaluate(FLAGS.checkpoint)

if __name__ == '__main__':
  app.run(main)