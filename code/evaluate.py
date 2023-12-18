import orbax.checkpoint as ocp
from jax import random
from ml_collections import ConfigDict

from models.utils import sample_gaussian
from dataset_utils import get_dataset
from utils import get_classifier, get_data_config, get_dtype, load_checkpoint, prepare_test_dataset
from jax import random
import jax.numpy as jnp
from jax.nn import one_hot
import orbax.checkpoint as ocp

from absl import app
from absl import flags
from pathlib import Path

def evaluate(flags):
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

  _, test_ds = get_dataset(config.dataset)
  dataset_config = get_data_config(test_ds)
  test_images, test_labels = prepare_test_dataset(
    test_ds, dataset_config, dtype
  )

  model_config = classifier.create_model_config(config, dataset_config)
  log_likelihood_fn = classifier.log_likelihood_A

  eval_key = random.PRNGKey(config.eval_seed)

  n_test = int(test_images.shape[0]*flags.ratio)
  eval_key, test_accuracy_value = classifier.compute_batch_accuracy(
    eval_key, 
    model_config, 
    checkpoint["params"], 
    test_images[:n_test], 
    test_labels[:n_test], 
    log_likelihood_fn,
    K=K,
  )
  print(f"\nAccuracy: {test_accuracy_value}% on {n_test} samples.")


FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", "", "Checkpoint relative path.")
flags.DEFINE_float("ratio", 1.0, "ratio of the test set to use for the evaluation")
flags.DEFINE_string("dtype", "", "dtype")
flags.DEFINE_integer("K", -1, "number of samples to use for importance sampling")

def main(argv):
  evaluate(FLAGS)

if __name__ == '__main__':
  app.run(main)