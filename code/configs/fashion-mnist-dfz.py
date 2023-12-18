from ml_collections import config_dict

def get_config():
  config = config_dict.ConfigDict()
  # params for the run
  config.checkpoint_name = "dfz-2-epochs-first-try"
  config.checkpoint = True
  config.train_seed = 123
  config.eval_seed = 456
  config.attack_seed = 789

  # params about the dataset
  config.dataset = "fashion-mnist"

  # training params
  config.dtype = "float32"
  config.batch_size = 50
  config.optimiser = "adam"
  config.learning_rate = 1e-4
  config.num_epochs = 2

  # high-level model params
  config.model_name = "DFZ"
  config.model = config_dict.ConfigDict()
  config.model.d_latent = 64
  config.model.d_hidden = 500
  config.model.K = 10
  config.model.dropout_rate = 0.2

  return config