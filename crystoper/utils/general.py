from .. import config
from pathlib import Path
from os import makedirs

def make_parent_dirs(filepath):
    makedirs(str(Path(filepath).parent), exist_ok=True)

def vprint(*args, **kwargs):
    if config.verbose:
        print(*args, **kwargs)
    