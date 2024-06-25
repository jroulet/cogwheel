# Download latest version from pypi and create cogwheel-pe/meta.yaml
grayskull pypi cogwheel-pe --maintainers jroulet

# The default meta.yaml has a couple problems, fix them:
python scripts/finalize_meta_yaml.py

# Make .tar.bz2 file to upload to conda-forge
conda build cogwheel_pe
