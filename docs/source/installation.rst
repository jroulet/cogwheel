Installation
============

Via ``conda``
-------------

The easiest way of installing ``cogwheel`` is using ``conda``:

.. code-block:: sh


    # Create and activate conda environment
    conda create --name <environment_name>
    conda activate <environment_name>

    # Install cogwheel
    conda install -c conda-forge cogwheel-pe

(replace ``<environment_name>`` by a name of your choice).

From source
-----------

If you would like to modify the source code or try the latest development version, you should instead install from source. In a virtual environment, do:

.. code-block:: sh

    # Clone repository
    git clone git@github.com:jroulet/cogwheel.git
    cd cogwheel

    # Create and activate virtual environment
    conda env create -f environment.yaml
    conda activate cogwheel-env

    # Install cogwheel
    pip install -e .
