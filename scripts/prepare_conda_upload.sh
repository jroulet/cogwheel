# Download latest version from pypi and create cogwheel-pe/meta.yaml
scripts_dir=$(dirname "$0")
grayskull pypi cogwheel-pe --maintainers jroulet --strict-conda-forge --output "${scripts_dir}/.."

# The default meta.yaml has a couple problems, fix them:
python "${scripts_dir}/finalize_meta_yaml.py"
