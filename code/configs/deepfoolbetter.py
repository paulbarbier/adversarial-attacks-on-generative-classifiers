from ml_collections import config_dict

def get_config():
  config = config_dict.ConfigDict()

  config.attack_name = "deepfoolbetter"
  config.checkpoint_name = "deepfoolbetter-1"
  config.max_iter = 50
  config.learning_rate = 1.0
  config.p = 2
  config.attack_batch_size = 80

  return config
