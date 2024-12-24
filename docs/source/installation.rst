Installation
============

Via ``conda``
-------------

The easiest way of installing ``cogwheel`` is using ``conda``. Inside a virtual environment, do:

.. code-block:: sh

    # Create and activate conda environment
    conda create --name <environment_name>
    conda activate <environment_name>

    # Install cogwheel
    conda install -c conda-forge cogwheel-pe

(replace ``<environment_name>`` by a name of your choice).

From source
-----------

If you would like to modify the source code or try the latest development version, you should instead install from source. Note that you still have to install ``lalsimulation`` in the environment, which ``pip`` won't download automatically.

.. code-block:: sh

    # Create environment, install lalsimulation
    conda create --name <environment_name> python-lalsimulation --channel conda-forge
    conda activate <environment_name>

    # Download source and install
    git clone git@github.com:jroulet/cogwheel.git
    cd cogwheel
    pip install -e .
