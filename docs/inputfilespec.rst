.. toctree::
    :maxdepth: 1

YAML Input File
===============

A YAML input file consists of the following top level sections (in no
particular order)

1. General
2. Solver
3. Simulation
4. Materials
5. Layers
6. Postprocessing 

For starters, here is a complete input YAML file, you could copy and paste this
into some text file on your machine and be ready to roll. The comments in here
well describe what all the various options do. 

.. literalinclude:: example.yml

Types of Parameters
-------------------

There are 3 "types" of parameters to choose from:

* "Fixed" parameters
* "Variable" parameters
* "Optimized" parameters

Fixed Parameters
----------------

Fixed parameters are just that, some configurable system parameter that has a
single fixed value. Specify them with

.. code-block:: yaml

    param_name:
        type: 'fixed'
        value: $your_value

Variable Parameters
-------------------

Variable parameters are parameters you would like to sweep through (i.e
frequency, or maybe incident angle). There are two ways to specify this:

.. code-block:: yaml

    frequency:
        type: 'variable'
        start: START
        end: END
        step: 20
        itertype: 'numsteps'

This says start at START and sweep up to END (inclusive of END) in 20 equally
spaced steps. The ``itertype: 'numsteps'`` param tells the ``sim_wrapper.py`` script to
interpret the 'step' parameter as the number of steps to take. This is the
equivalent of numpy/matlab's  ``linspace``

The other way to specify a variable parameter is:

.. code-block:: yaml

    frequency:
        type: 'variable'
        start: START
        end: END
        step: 20
        itertype: 'stepsize'

This says sweep from START to END (inclusive) in steps of size 20. 

You can specify multiple variable parameters. The ``sim_wrapper`` script will
automatically generate every possible unique parameter combination and run a
simulation for each one it generates. 

Optimized Parameters
--------------------

Optimized parameters are parameters you wish to optimize. The code uses the
Nelder Mead simplex optimizer from scipy for optimizations. Right now the setup
of the optimizer is a bit immature. It basically assumes you want to optimize
the photocurrent density of the device (i.e spectrum-wide absorbtion) and also
assumes that your only variable parameter is frequency. I hope to enhance this
at some point to make the objective function configurable. My current use case
it to optimize the geometry to maximize total absorption. Specify optimized
parameters with:

.. code-block:: yaml

    param_name:
        type: 'optimized'
        value: $your_value

You can specify multiple optimized parameters. The optimizer will attempt to
optimize all of them to maximize absorption. The optimizer is NOT a global
optimizer, so be wary of falling into local minima. Also, know that as you
increase the number of parameters you optimize over the optimizer will take
longer to converge.  

Parameter Substition
--------------------

Within the YAML file, it is possible to refer to other parameters in the config
by inserting a specifically formatted string. This string will be replaced with
the value of the parameter to which it refers. Best illustrated with an
example:

.. code-block:: yaml

    max_depth: '%(Layers.Air.params.thickness.value)s'

The thing inside the ``()`` is a dot-separated path to some location in the
config file. In this case, the value of the thickness of the air layer. The
``%()s`` is what tells the ``Config`` object to run a substitution. Always enclose
you substitions in quotes or things will break.

Parameter Evaluation
--------------------

Its possible to evaluate arbitrary python expressions by enclosing any
parameter in back ticks. There is not validation or sanitation of this
expression, so it is horribly insecure but no external, malicious entities will
be running this code or injecting anything so its not really a problem (yet). 
Anyway, illustrated with an example

.. code-block:: yaml

    max_depth: '`1+2+3`'

This is silly and just says "add these numbers and set max_depth to the
result". We could have just entered 6. However, this gets really handy when
combined with parameter substitution. Example: 

.. code-block:: yaml

    max_depth: '`(%(Layers.Air.params.thickness.value)s+%(Layers.ITO.params.thickness.value)s+%(Layers.NW_AlShell.params.thickness.value)s+3)*2`'

This says:

#. Go get the values of all these layer thicknesses (whatever they may be, you
   could be sweeping through these thicknesses or optimizing them and then you won't
   know what the particular value will be ahead of time)
#. Sum them all
#. Add 3 for some reason
#. Multiply the value of the sum by two (notice the extra outer parentheses)
#. Set max depth to the result 

This is super handy when for example you want your shell to be 30 nm thick (i.e
shell_radius = core_radius + 30 nm) 

Environment Variables
---------------------

You can reference environment variables in the config using the familiar
$VARIABLE bash syntax. This can be useful for specifying the location of your
$HOME directory, or some other variables you may have set in your config
file. For example, when referencing some input files you could do:

.. code-block:: yaml

    Materials:
        ITO: '$HOME/software/nanowire/NK/008_ITO_nk_Hz.txt'
        Cyclotene: '$HOME/software/nanowire/NK/007_Cyclotrene_nk_Hz.txt'
        GaAs: '$HOME/software/nanowire/NK/006_GaAs_nk_Walker_modified_Hz.txt'
        AlInP: '$HOME/software/nanowire/NK/009_AlInP_nk_Hz.txt'

This is super handy when you work on multiple servers, perhaps with different
home directories. You no longer have to change your config files when moving
them across servers, you can just use the proper environment variables. 

Postprocessing
--------------

The postprocessing section of the config file is pretty nifty. The structure
of this section is used to configure which postprocessing functions get called,
and what arguments they receive. 

The headings immediately below the top level ``Postprocessing`` heading each
correspond to objects in ``optics/postprocess.py``. Each of these objects
serve a specific purpose, detailed below

* Cruncher: Computes quantities that are specific to an individual simulation
* Global_Cruncher: Computes global quantities for a group of simulations
* Plotter: Plots things that are specific to an individual simulation
* Global_Plotter: Plots global quantities for a group of simulations

Underneath each of these 1st level object headings there exists a configuration
section that determines which methods of the object will get called, and with
which arguments they will be called. The level of headings beneath each object
correspond to the name of a method defined within that object. The string here
is very important, as it corresponds *exactly* to the name of a method of the
parent object in ``postprocess.py``. Any typos mean the method won't get called
properly.  Beneath the method name, the ``compute`` flag just determines
whether or not the function will get called so you can quickly turn things off
and on. The ``args`` flag determines what arguments get passed to the method.
This must be entered as a list for python's argument expansion operators to
work. You can specify a single list, meaning only a single set of arguments
will be passed to the function. You can also specify a list of lists, meaning
the function will get called multiple times with multiple sets of arguments.
The order of the arguments is super important, and you need to make sure you
check the function signature of the method in ``postprocess.py`` to get the
order of the arguments correct. Accidentally swapping arguments can cause weird
things to happen. 

Overall, this system is a little bit delicate but really nicely allows you to
rapidly and completely automatically compute and plot a bunch of interesting
things without having to modify any code. 

NOTE: I might be changing the design of the postprocessing code really soon. It
currently doesn't allow me to run postprocessing functions in parallel, which
slows things down a lot. I'm basically using each of these objects as a big
dictionary for methods, so I might literally switch to a "dispatcher" approach
where I store module-level functions in a big dictionary instead of inside
objects. This will be a long term fix that will ocurr on a separate branch so
don't worry about it for now. 

