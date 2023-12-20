
from functools import partial
from ml_collections import ConfigDict, config_flags

from models.utils import sample_gaussian
from dataset_utils import get_dataset, get_dataloader, split_dataset
from utils import get_classifier, get_data_config, get_dtype, prepare_test_dataset
from optimiser import get_optimiser
from jax import random
import jax.numpy as jnp
from jax.nn import one_hot
from flax.training import train_state
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import orbax.checkpoint as ocp

from absl import app
from absl import flags
from pathlib import Path

CHECKPOINT_DIR = Path.cwd() / Path("checkpoints")

def train_and_evaluate(flags):
    config = flags.config

    dtype = get_dtype(config.dtype)

    if config.checkpoint:
        checkpointer = ocp.PyTreeCheckpointer()

    ds = get_dataset(config.dataset, train=True)
    train_ds, test_ds = split_dataset(ds, [config.n_train, config.n_test], config.train_seed)

    train_dl = get_dataloader(train_ds, config.train_batch_size, dtype)
    test_dl = get_dataloader(test_ds, config.test_batch_size, dtype)
    
    key = random.PRNGKey(config.train_seed)

    classifier = get_classifier(config)
    model_config = classifier.create_model_config(config)
    key, init_params = classifier.init_params(key, model_config, dtype)

    optimiser = get_optimiser(config)

    training_state = classifier.create_training_state(model_config, init_params, optimiser) 
    del init_params # access params only through training_state

    # split training/test keys
    key, training_key, dropout_key, test_key = random.split(key, 4)

    #sink for the metric values
    metrics = {
        "training_steps": [],
        "test_steps": [],
        "training_loss": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_f1_score_micro": [],
        "test_f1_score_macro": [],
        "test_precision_micro": [],
        "test_precision_macro": [],
        "test_recall_micro": [],
        "test_recall_macro": [],
        "test_confusion_matrix": [],
    }
    
    epsilon_shape = (config.train_batch_size, config.model.d_latent)
    log_likelihood_fn = classifier.log_likelihood_A
    loss_fn = classifier.loss_A
    loss_single_fn = classifier.loss_A_single

    # training loop
    for epoch in range(1, config.num_epochs+1):
        with tqdm(train_dl, unit="batch") as train_epoch:
            train_epoch.set_description(f"Train epoch #{epoch}")
            for train_step, (X_batch, y_batch) in enumerate(train_epoch):
                if flags.debug and train_step == 100:
                    break
                # one-hot encoding
                y_batch_one_hot = one_hot(y_batch, config.n_classes)
                
                # sample the prior from gaussian distribution
                training_key, epsilon = sample_gaussian(training_key, epsilon_shape)

                # training step
                training_state, loss_value = classifier.training_step(
                    training_state, X_batch, y_batch_one_hot, epsilon, loss_single_fn, dropout_key
                )
                
                # log the training loss
                metrics["training_loss"].append(loss_value)
                metrics["training_steps"].append(epoch*config.num_epochs + train_step+1)

                train_epoch.set_postfix(training_loss=loss_value)
            
            y_true, y_predictions, test_losses = [], [], []
            with tqdm(test_dl, unit="batch") as test_epoch:
                test_epoch.set_description(f"Test epoch #{epoch}")
                for test_step, (X_batch, y_batch) in enumerate(test_epoch):
                    if flags.debug and test_step == 2:
                        break
                    y_batch_one_hot = one_hot(y_batch, config.n_classes)

                    # compute batch prediction
                    test_key, y_pred_batch = classifier.make_predictions(
                        test_key, model_config, training_state.params, X_batch, log_likelihood_fn
                    )
                    y_predictions.append(y_pred_batch)
                    y_true.append(y_batch)

                    # compute batch loss
                    test_key, batch_loss = classifier.compute_batch_loss(
                        test_key, model_config, training_state.params, X_batch, y_batch_one_hot, loss_fn
                    )
                    test_losses.append(batch_loss)

            y_true = jnp.concatenate(y_true)
            y_predictions = jnp.concatenate(y_predictions)
            test_losses = jnp.array(test_losses)

            metrics["test_steps"].append((epoch-1)*config.num_epochs + train_step+1)
            metrics["test_accuracy"].append(accuracy_score(y_true, y_predictions))
            metrics["test_precision_micro"].append(precision_score(y_true, y_predictions, average="micro"))
            metrics["test_precision_macro"].append(precision_score(y_true, y_predictions, average="macro"))
            metrics["test_f1_score_micro"].append(f1_score(y_true, y_predictions, average="micro"))
            metrics["test_f1_score_macro"].append(f1_score(y_true, y_predictions, average="macro"))
            metrics["test_recall_micro"].append(recall_score(y_true, y_predictions, average="micro"))
            metrics["test_recall_macro"].append(recall_score(y_true, y_predictions, average="macro"))
            metrics["test_confusion_matrix"].append(confusion_matrix(y_true, y_predictions))
            metrics["test_loss"].append(jnp.mean(test_losses))

            metric_keys = ["test_loss", "test_accuracy", "test_precision_micro", "test_recall_micro", "test_f1_score_micro"]
            print(", ".join(f"{key}: {metrics[key][-1]:.3f}" for key in metric_keys))

            if config.checkpoint:
                checkpointer.save(
                    CHECKPOINT_DIR / f"{config.checkpoint_name}-{epoch}",
                    {
                        "config": config,
                        "model_config": model_config,
                        "params": training_state.params,
                        "metrics": metrics, 
                    }
                )

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'path to configuration file.',
    lock_config=True,
)

flags.DEFINE_bool("debug", False, "debug flag")

def main(argv):
  train_and_evaluate(FLAGS)

if __name__ == '__main__':
  app.run(main)
