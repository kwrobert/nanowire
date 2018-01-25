.. toctree::
    :maxdepth: 1

Useage
======

There are two main ways to use this library. The first way is more conducive to
an interactive workflow. The second is more conducive to running large
parameter sweeps and optimizations. 

Iteractive Useage
-----------------

Oftentimes, its really nice to be able to use a library interactively to
perform quick tests, sanity checks, and to get a sense for how the API of the
library works. To accomodate this kind of workflow, this library provides some
useful objects with a concise, simple API. One starts by loading up a
configuration file into a convenient dict-like object using the 
:py:class:nanowire.optics.utils.config.Config class. One feeds this class into
the constructor of a :py:class:nanowire.optics.simulate.Simulator object to run
a simulation and generate some data.

To analyze the data interactively, you have two options. then either write your
own code that utilizes an instance of ``Simulator`` directly, or feed a
``Simulator`` instance to the constructor of a 
:py:class:nanowire.optics.postprocess.Simulation. The ``Simulation`` object
contains a lot of useful methods for analyzing simulation data, and has very
similiar data storage and access mechanism.

Batch Useage
------------

To run large job campaigns and postprocess all the results, two scripts are
provided: the ``run_optics`` script and the ``process_optics`` script. If you
installed this library using ``pip`` and your ``$PATH`` environment variable is
configured to include ``$HOME/.local/bin``, then these scripts should be
accessible directly from your terminal. 

One uses these scripts by properly specifying the desired parameter sweeps and
optimizations in the input YAML file, as well as the desired postprocessing
functions to call. To run all the jobs specified in your config file, execute::

    run_optics path/to/input.yml

Once the jobs have completed, postprocess all the results with::

    process_optics path/to/input.yml
                                              
And thats it! 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

