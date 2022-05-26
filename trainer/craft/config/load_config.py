import os
import yaml
from functools import reduce

CONFIG_PATH = os.path.dirname(__file__)

def load_yaml(config_name):

    with open(os.path.join(CONFIG_PATH, config_name)+ '.yaml') as file:
        config = yaml.safe_load(file)

    return config

class DotDict(dict):
    def __getattr__(self, k):
        try:
            v = self[k]
        except:
            return super().__getattr__(k)
        if isinstance(v, dict):
            return DotDict(v)
        return v

    def __getitem__(self, k):
        if isinstance(k, str) and '.' in k:
            k = k.split('.')
        if isinstance(k, (list, tuple)):
            return reduce(lambda d, kk: d[kk], k, self)
        return super().__getitem__(k)

    def get(self, k, default=None):
        if isinstance(k, str) and '.' in k:
            try:
                return self[k]
            except KeyError:
                return default
        return super().get(k, default=default)