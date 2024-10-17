from .. import config


def vprint(*args, **kwargs):
    if config.verbose:
        print(*args, **kwargs)
    