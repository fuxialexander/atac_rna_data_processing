# load the yml file into a config object
from box import Box
import yaml

def load_config(conf):
    with open(f"{conf}.yaml", "r") as f:
        config_data = yaml.safe_load(f)
        config = Box(config_data)
    return config

