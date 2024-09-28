import os
import json
from collections import namedtuple

CONFIG_PATH = 'config.json'

with open(f'{os.path.dirname(__file__)}/{CONFIG_PATH}') as f:
    config = json.load(f)
config = namedtuple('config', config.keys())(**config)