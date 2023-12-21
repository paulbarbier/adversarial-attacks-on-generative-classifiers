from ml_collections import config_dict

def get_config():
  config = config_dict.ConfigDict()

  config.name = "logit"
  config.alpha = 2.0
  config.num_thresholds = 100

  return config