from jax import random
from ml_collections import ConfigDict
from tqdm import tqdm

from dataset_utils import get_dataloader, get_dataset
from utils import get_classifier, get_dtype, load_checkpoint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
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

  eval_ds = get_dataset(config.dataset, train=False)
  eval_dl = get_dataloader(eval_ds, config.test_batch_size, dtype)

  model_config = classifier.create_model_config(config)
  log_likelihood_fn = classifier.log_likelihood_A
  loss_fn = classifier.loss_A
  params = checkpoint["params"]

  eval_key = random.PRNGKey(config.eval_seed)

  #sink for the metric values
  metrics = {
    "loss": [],
    "accuracy": [],
    "f1_score_micro": [],
    "f1_score_macro": [],
    "precision_micro": [],
    "precision_macro": [],
    "recall_micro": [],
    "recall_macro": [],
    "confusion_matrix": [],
  }

  y_true, y_predictions, eval_losses = [], [], []
  with tqdm(eval_dl, unit="batch") as teval:
    teval.set_description(f"Evaluation")
    for eval_step, (X_batch, y_batch) in enumerate(teval):
      if flags.debug and eval_step == 2:
        break
      y_batch_one_hot = one_hot(y_batch, config.n_classes)

      # compute batch prediction
      eval_key, y_pred_batch = classifier.make_predictions(
        eval_key, model_config, params, X_batch, log_likelihood_fn
      )
      y_predictions.append(y_pred_batch)
      y_true.append(y_batch)

      # compute batch loss
      eval_key, batch_loss = classifier.compute_batch_loss(
        eval_key, model_config, params, X_batch, y_batch_one_hot, loss_fn
      )
      eval_losses.append(batch_loss)

  y_true = jnp.concatenate(y_true)
  y_predictions = jnp.concatenate(y_predictions)
  eval_losses = jnp.array(eval_losses)

  metrics["accuracy"].append(accuracy_score(y_true, y_predictions))
  metrics["precision_micro"].append(precision_score(y_true, y_predictions, average="micro"))
  metrics["precision_macro"].append(precision_score(y_true, y_predictions, average="macro"))
  metrics["f1_score_micro"].append(f1_score(y_true, y_predictions, average="micro"))
  metrics["f1_score_macro"].append(f1_score(y_true, y_predictions, average="macro"))
  metrics["recall_micro"].append(recall_score(y_true, y_predictions, average="micro"))
  metrics["recall_macro"].append(recall_score(y_true, y_predictions, average="macro"))
  metrics["confusion_matrix"].append(confusion_matrix(y_true, y_predictions))
  metrics["loss"].append(jnp.mean(eval_losses))

  metric_keys = ["loss", "accuracy", "precision_micro", "precision_macro", "recall_micro", "recall_macro", "f1_score_micro", "f1_score_macro"]
  print("\n".join(f"{key}: {metrics[key][-1]:.3f}" for key in metric_keys))

  print("Confusion matrix:")
  print(metrics["confusion_matrix"][-1])


FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", "", "Checkpoint relative path.")
flags.DEFINE_string("dtype", "", "dtype")
flags.DEFINE_bool("debug", False, "debug flag")
flags.DEFINE_integer("K", -1, "number of samples to use for importance sampling")

def main(argv):
  evaluate(FLAGS)

if __name__ == '__main__':
  app.run(main)