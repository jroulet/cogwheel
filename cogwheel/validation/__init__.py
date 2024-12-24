"""Validate the cogwheel code by doing inference on injections."""
import importlib
import sys

from cogwheel import utils


def load_config(config_filename):
    """
    Import desired config module.
    See cogwheel/validation/example/config.py for an example.

    Parameters
    ----------
    config_filename : PathLike
        Full path to a file containing configuration parameters for the
        injections.

    Returns
    -------
    module
    """
    spec = importlib.util.spec_from_file_location('config', config_filename)
    config = importlib.util.module_from_spec(spec)
    with utils.temporarily_change_attributes(sys, dont_write_bytecode=True):
        spec.loader.exec_module(config)

    return config
