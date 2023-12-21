from ml_collections import config_dict

def get_config():
  config = config_dict.ConfigDict()

  config.attack_name = "deepfoolpaper"
  config.checkpoint_name = "deepfoolpaper-1"
  config.max_iter = 10
  config.learning_rate = 1.0
  config.p = 2
  config.attack_batch_size = 30

  return config
