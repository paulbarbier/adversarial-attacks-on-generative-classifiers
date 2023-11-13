
from ml_collections import ConfigDict, config_flags

from models.utils import sample_p
from models.loss import compute_batch_accuracy, compute_batch_loss, loss_A_single, loss_A
from datasets import get_dataset, get_dataloader
from utils import get_data_config, prepare_test_dataset
from optimiser import get_optimiser
from models.ModelGFZ import init_model, update_step
from jax import random
import jax.nn as nn
from flax.training import train_state
from tqdm import tqdm
import orbax.checkpoint as ocp

from absl import app
from absl import flags
from pathlib import Path

CHECKPOINT_DIR = Path.cwd() / Path("checkpoints")


def train_and_evaluate(config: ConfigDict):
    checkpointer = ocp.PyTreeCheckpointer()

    train_ds, test_ds = get_dataset(config.dataset)
    train_dl = get_dataloader(train_ds, config.batch_size)

    dataset_config = get_data_config(train_ds)

    test_images, test_labels = prepare_test_dataset(test_ds, dataset_config)
    
    key = random.PRNGKey(config.seed)

    key, model, init_params = init_model(key, config, dataset_config)

    optimiser = get_optimiser(config)

    training_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=init_params,
        tx=optimiser,
    )

    # split training/test keys
    key, training_key, test_key = random.split(key, 3)

    #sinks for the loss values
    training_steps, test_steps = [], []
    training_loss_values, test_loss_values, test_accuracy_values = [], [], []
    loss_value, test_loss_value, test_accuracy_value = None, None, None

    epsilon_shape = (config.batch_size, config.d_epsilon)
    def log_likelyhood(z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z):
        return -loss_A(z, logit_q_z_xy, logit_p_x_yz, logit_p_y_z)
    loss = loss_A

    # training loop
    for epoch in range(1, config.num_epochs+1):
        with tqdm(train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for train_step, (X_batch, y_batch) in enumerate(tepoch):
                # one-hot encoding
                y_batch_one_hot = nn.one_hot(y_batch, dataset_config.n_classes)
                
                # sample from noise distribution
                training_key, epsilon = sample_p(training_key, epsilon_shape)

                # training step
                training_state, loss_value = update_step(
                    training_state, X_batch, y_batch_one_hot, epsilon, loss_A_single
                )
                
                # log the training loss: here it's just the batch loss, need to be fixed.
                training_loss_values.append(loss_value)
                training_steps.append(epoch*config.num_epochs + train_step+1)

                tepoch.set_postfix(training_loss=loss_value, test_loss=test_loss_value, test_accuracy=test_accuracy_value)

            # compute test loss and accuracy
            test_key, test_loss_value = compute_batch_loss(
                test_key, model, training_state.params, test_images, test_labels, loss
            )

            test_key, test_accuracy_value = compute_batch_accuracy(
                test_key, model, training_state.params, test_images[:100], test_labels[:100], log_likelyhood, K=config.K
            )
            test_loss_values.append(test_loss_value)
            test_accuracy_values.append(test_accuracy_value)
            test_steps.append(epoch*config.num_epochs + train_step+1)

            tepoch.set_postfix(training_loss=loss_value, test_loss=test_loss_value, accuracy=test_accuracy_value)

            checkpointer.save(
               CHECKPOINT_DIR / f"{config.checkpoint_name}-{epoch}",
               {
                  "model": model,
                  "params": training_state.params,
                  "config": config.to_dict(),
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
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)


def main(argv):
  train_and_evaluate(FLAGS.config)

if __name__ == '__main__':
  app.run(main)