"""
Package to validate the cogwheel code by doing inference on injections.
"""
import importlib


def load_config(config_filename):
    """
    Import desired config module.
    See cogwheel/validation/example/config.py for an example.

    Parameters
    ----------
    config_filename: PathLike
        Full path to a file containing configuration parameters for the
        injections.

    Return
    ------
    module
    """
    spec = importlib.util.spec_from_file_location('config', config_filename)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config
