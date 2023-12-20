from ml_collections import config_dict

def get_config():
  config = config_dict.ConfigDict()

  config.attack_name = "fastgradientsign"
  config.checkpoint_name = "fastgradientsign-1"
  config.eta = 0.3
  config.attack_batch_size = 100

  return config