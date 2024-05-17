from setuptools import setup, find_packages

setup(
    name='cogwheel',
    version='v1.0.3',
    packages=find_packages(),
    package_data={'cogwheel': ['cogwheel/data/events_metadata.csv',
                               'cogwheel/data/*.npz',
                               'cogwheel/data/example_asds/*.npz']},
    include_package_data=True,
)
