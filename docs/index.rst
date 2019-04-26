Welcome to the pySOT documentation!
===================================

This is the documentation for the Surrogate Optimization Toolbox (pySOT) for
global deterministic optimization problems. pySOT is hosted on GitHub:
`https://github.com/dme65/pySOT <https://github.com/dme65/pySOT>`_.

The main purpose of the toolbox is for optimization of computationally
expensive black-box objective functions with continuous and/or integer
variables. 
All variables are assumed to have bound constraints in some form where none
of the bounds are infinity. The tighter the bounds, the more efficient are
the algorithms since it reduces the search region and increases the quality
of the constructed surrogate. 
This toolbox may not be very efficient for problems with computationally cheap
function evaluations. Surrogate models are intended to be used when function
evaluations take from several minutes to several hours or more.


For easier understanding of the algorithms in this toolbox, it is recommended
and helpful to read these papers. If you have any questions, or you encounter
any bugs, please feel free to either submit a bug report on GitHub (recommended)
or to contact me at the email address: dme65@cornell.edu. Keep an eye on the
GitHub repository for updates and changes to both the toolbox and the documentation.

The toolbox is based on the following published papers: [1_], [2_], [3_] [4_]

.. toctree::
   :maxdepth: 4
   :caption: User Documentation

   quickstart
   surrogate_optimization
   options
   poap
   logging
   source_code
   changes
   license
   contributors


.. [1] Rommel G Regis and Christine A Shoemaker.
    A stochastic radial basis function method for the global optimization of expensive functions.
    INFORMS Journal on Computing, 19(4): 497–509, 2007.

.. [2] Rommel G Regis and Christine A Shoemaker.
    Parallel stochastic global optimization using radial basis functions.
    INFORMS Journal on Computing, 21(3):411–426, 2009.

.. [3] Rommel G Regis and Christine A Shoemaker.
    Combining radial basis function surrogates and dynamic coordinate search in high-dimensional expensive black-box optimization.
    Engineering Optimization, 45(5): 529–555, 2013.

.. [4] Tipaluck Krityakierne, Taimoor Akhtar and Christine A. Shoemaker.
    SOP: parallel surrogate global optimization with Pareto center selection for computationally expensive single objective problems.
    Journal of Global Optimization, 66(3): 417–437, 2016.
