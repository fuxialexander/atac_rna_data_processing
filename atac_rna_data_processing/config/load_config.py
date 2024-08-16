# load the yml file into a config object
import yaml


class Config:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                value = Config(**value)
            self.__dict__[key] = value
    def __getitem__(self, key):
        return self.__dict__.get(key, None)
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    def __getattr__(self, item):
        if item not in self.__dict__:
            self.__dict__[item] = Config()
        return self.__dict__[item]
    def __repr__(self):
        return f"Config({self.__dict__})"
    def to_dict(self):
        return {key: value.to_dict() if isinstance(value, Config) else value for key, value in self.__dict__.items()}


def load_config(conf):
    with open(f"{conf}.yaml", "r") as f:
        config_data = yaml.safe_load(f)
        config = Config(config_data)
    return config
