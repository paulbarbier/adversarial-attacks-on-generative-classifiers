from ml_collections import config_dict

def get_config():
  config = config_dict.ConfigDict()
  config.seed = 1
  config.batch_size = 50
  config.optimiser = "adam"
  config.learning_rate = 1e-4
  config.d_epsilon = 64
  config.dataset = "fashion-mnist"
  config.num_epochs = 20
  config.K = 10
  config.checkpoint_name = "dummy_checkpoint"
  return config