from ml_collections import config_dict

def get_config():
  config = config_dict.ConfigDict()

  config.attack_name = "fastgradientsign"
  config.eta = 0.3

  return config