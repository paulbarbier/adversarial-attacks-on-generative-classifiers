import orbax.checkpoint as ocp
from pathlib import Path
from ml_collections import config_dict
from tqdm import tqdm

def get_config_dfz():
  config = config_dict.ConfigDict()
  # params for the run
  config.checkpoint_name = "dfz-50-epochs-a"
  config.checkpoint = True
  config.train_seed = 123
  config.eval_seed = 456
  config.attack_seed = 789

  # params about the dataset
  config.dataset = "fashion-mnist"
  config.n_train = 50000
  config.n_test = 10000
  config.n_classes = 10
  config.image_width = 28
  config.image_height = 28
  config.image_channels = 1

  # training params
  config.dtype = "float32"
  config.train_batch_size = 100 
  config.test_batch_size = 100
  config.optimiser = "adam"
  config.learning_rate = 1e-4
  config.num_epochs = 50

  # high-level model params
  config.model_name = "DFZ"
  config.model = config_dict.ConfigDict()
  config.model.d_latent = 64
  config.model.d_hidden = 500
  config.model.K = 10
  config.model.dropout_rate = 0.5

  return config

def get_config_gfz():
  config = config_dict.ConfigDict()
  # params for the run
  config.checkpoint_name = "gfz-50-epochs-a"
  config.checkpoint = True
  config.train_seed = 123
  config.eval_seed = 456
  config.attack_seed = 789

  # params about the dataset
  config.dataset = "fashion-mnist"
  config.n_train = 50000
  config.n_test = 10000
  config.n_classes = 10
  config.image_width = 28
  config.image_height = 28
  config.image_channels = 1

  # training params
  config.dtype = "float32"
  config.train_batch_size = 100 
  config.test_batch_size = 100
  config.optimiser = "adam"
  config.learning_rate = 1e-4
  config.num_epochs = 50

  # high-level model params
  config.model_name = "GFZ"
  config.model = config_dict.ConfigDict()
  config.model.d_latent = 64
  config.model.d_hidden = 500
  config.model.K = 10
  config.model.dropout_rate = 0.5

  return config

checkpoints_path = [
    f"gfz-50-epochs-a-{n}" for n in range(1, 50)
]

config = get_config_gfz().to_dict()

checkpointer = ocp.PyTreeCheckpointer()
for path in tqdm(checkpoints_path):
    checkpoint = checkpointer.restore(Path.cwd() / Path("checkpoints") / Path(path)) 
    checkpoint["config"] = config

    checkpointer.save(Path.cwd() / Path("checkpoints/modified") / Path(path), checkpoint)