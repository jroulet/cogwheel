# Note: The __version__ will not get updated automatically if cogwheel
# was installed in editable mode (i.e. as ``pip install -e .``). In that
# case, you should reinstall it every time you need the version to get
# updated. Code below from
# https://github.com/pypa/setuptools_scm/issues/143#issuecomment-672878863
try:
    from ._version import __version__
except ImportError:
    # Possibly running in the git repository
    try:
        import setuptools_scm
        __version__ = setuptools_scm.get_version()
        del setuptools_scm
    except Exception:
        import logging
        logging.warning(f'Could not determine {__name__} package version.')
        __version__ = None
        del logging
