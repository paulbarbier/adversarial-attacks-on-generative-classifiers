
from functools import partial
from ml_collections import ConfigDict, config_flags

from models.utils import sample_gaussian
from dataset_utils import get_dataset, get_dataloader
from utils import get_classifier, get_data_config, get_dtype, prepare_test_dataset
from optimiser import get_optimiser
from jax import random
from jax.nn import one_hot
from flax.training import train_state
from tqdm import tqdm
import orbax.checkpoint as ocp

import models.ClassifierGFZ as ClassifierGFZ
import models.ClassifierDFZ as ClassifierDFZ

from absl import app
from absl import flags
from pathlib import Path

CHECKPOINT_DIR = Path.cwd() / Path("checkpoints")


def train_and_evaluate(config: ConfigDict):
    dtype = get_dtype(config.dtype)

    checkpointer = None
    if config.checkpoint:
        checkpointer = ocp.PyTreeCheckpointer()

    train_ds, test_ds = get_dataset(config.dataset)
    train_dl = get_dataloader(train_ds, config.batch_size, dtype)

    dataset_config = get_data_config(train_ds)

    test_images, test_labels = prepare_test_dataset(
       test_ds, dataset_config, dtype
    )
    
    key = random.PRNGKey(config.train_seed)

    classifier = get_classifier(config)
    model_config = classifier.create_model_config(config, dataset_config)
    key, init_params = classifier.init_params(key, model_config, dtype)

    optimiser = get_optimiser(config)

    training_state = classifier.create_training_state(model_config, init_params, optimiser) 
    del init_params # access params only through training_state

    # split training/test keys
    key, training_key, dropout_key, test_key = random.split(key, 4)

    #sinks for the metric values
    training_steps, test_steps = [], []
    training_loss_values, test_loss_values, test_accuracy_values = [], [], []
    loss_value, test_loss_value, test_accuracy_value = None, None, None

    epsilon_shape = (config.batch_size, config.model.d_latent)

    log_likelihood_fn = classifier.log_likelihood_A
    loss_fn = classifier.loss_A
    loss_single_fn = classifier.loss_A_single

    # training loop
    for epoch in range(1, config.num_epochs+1):
        with tqdm(train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for train_step, (X_batch, y_batch) in enumerate(tepoch):
                # one-hot encoding
                y_batch_one_hot = one_hot(y_batch, dataset_config.n_classes)
                
                # sample the prior from gaussian distribution
                training_key, epsilon = sample_gaussian(training_key, epsilon_shape)

                # training step
                training_state, loss_value = classifier.training_step(
                    training_state, X_batch, y_batch_one_hot, epsilon, loss_single_fn, dropout_key
                )
                
                # log the training loss
                training_loss_values.append(loss_value)
                training_steps.append(epoch*config.num_epochs + train_step+1)

                tepoch.set_postfix(training_loss=loss_value, test_loss=test_loss_value, test_accuracy=test_accuracy_value)

            # compute test loss and accuracy
            test_key, test_loss_value = classifier.compute_batch_loss(
                test_key, model_config, training_state.params, test_images, test_labels, loss_fn,
            )

            test_key, test_accuracy_value = classifier.compute_batch_accuracy(
                test_key, 
                model_config, 
                training_state.params, 
                test_images[:100], 
                test_labels[:100], 
                log_likelihood_fn,
            )

            test_loss_values.append(test_loss_value)
            test_accuracy_values.append(test_accuracy_value)
            test_steps.append(epoch*config.num_epochs + train_step+1)

            tepoch.set_postfix(
               training_loss=loss_value, test_loss=test_loss_value, accuracy=test_accuracy_value
            )
            if config.checkpoint:
                checkpointer.save(
                    CHECKPOINT_DIR / f"{config.checkpoint_name}-{epoch}",
                    {
                        "model": model_config,
                        "params": training_state.params,
                        "config": config.to_dict(),
                        "dataset_config": dataset_config.to_dict(),
                        "training_steps": training_steps,
                        "test_steps": test_steps,
                        "training_loss_values": training_loss_values,
                        "test_loss_values": test_loss_values,
                        "test_accuracy_values": test_accuracy_values, 
                    }
                )

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'path to configuration file.',
    lock_config=True,
)

def main(argv):
  train_and_evaluate(FLAGS.config)

if __name__ == '__main__':
  app.run(main)