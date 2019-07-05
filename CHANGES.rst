v.0.2.3,  2019-07-04
--------------------

- Adding SOP (Contributed by drkupi)
- Re-enabling restarts: Start a fresh run when we stop making progress

v.0.2.2,  2019-02-12
--------------------

- Experimental designs can now map and round to domains
- Support for generating multiple experimental designs and picking the best

v.0.2.1,  2019-01-26
--------------------

- Removing numpy.asmatrix calls, since this is now deprecated

v.0.2.0,  2018-12-06
--------------------

- Most of the pySOT codebase has been rewritten
- We support asynchronous function evaluations
- The strategy has been merged with the adaptive sampling
- The penalty method strategy has been removed, but may be added back later
- A CheckpointController has been added that enables resuming terminated runs
- Python 2 support has been dropped, we now support Python 3.4 and later
- Expected improvement (EI) and lower confidence bound (LCB) have been added

v.0.1.36, 2017-07-20
--------------------

- The GUI is now built in PyQt5 instead of PySide

v.0.1.35, 2017-04-29
--------------------

- Added support for termination based on elapsed time
- Added the Hartman6 test problem

v.0.1.34, 2017-03-28
--------------------

- Added support for adding points with known (and unknown) function values to the experimental design

v.0.1.33, 2016-12-27
--------------------

- Fixed a bug in MARS that resulted in using a lot of zero points for fitting
- Added a GP regression object based on scikit-learn 0.18.1
- Updated tests and documentation

v.0.1.32, 2016-12-07
--------------------

- Switched to make py-earth, matlab_wrapper, and subprocess32 optional dependencies to resolve pip installation issues

v.0.1.31, 2016-11-23
--------------------

- Added Python 3 support
- Removed Sphinx dependency
- Added six dependency to get py-earth to work for Python 3

v.0.1.30, 2016-11-18
--------------------

- Moved all of the official pySOT documentation over to Sphinx
- Five pySOT tutorials were added to the documentation
- The documentation is now hosted on Read the Docs (https://pysot.readthedocs.io)
- Removed pyKriging in order to remove the matplotlib and inspyred dependencies. A new Kriging module will be added in the next version.
- Added the MARS installation to the setup.py since it can now be installed via scikit-learn
- Updated the Sphinx documentation to include all of the source files
- The License, Changes, Contributors, and README files are not in .rst
- Renamed sampling_methods.py to adaptive_sampling.py
- Moved the kernels and tails to separate Python files
- Added a Gitter for pySOT

v0.1.29, 2016-10-20
-------------------

-  Correcting an error in the pypi upload

v0.1.28, 2016-10-20
-------------------

- Making the GUI work with the new RBF design

v0.1.27, 2016-10-18
-------------------

- Removed dimensionality argument for the RBF to match the other surrogates

v0.1.26, 2016-10-14
-------------------

- Signficant changes in the RBFInterpolant. Users need to update their code
- Added RBF regression surfaces
- Added version information in the module. pySOT.__version__ gives the version of the current pySOT installation
- The Gutmann strategy has been temporarily removed due to the RBF redesign, but will be added back soon
- Check out test_rbf.py to see how to use the new RBF

v0.1.25, 2016-09-14
-------------------

- Fixed a bug in DYCORS when the subset has length 1

v0.1.24, 2016-08-04
-------------------

- Changed to setup.py to use rst format for pypi

v0.1.23, 2016-07-28
-------------------

- Updates to support the new MPIController in POAP
- pySOT now sends copies of key variables in case they are changed by the method

v0.1.23, 2016-07-28
-------------------

- Updates to support the new MPIController in POAP
- pySOT now sends copies of key variables in case they are changed by the method

v0.1.22, 2016-06-27
-------------------

- Added two tests for the MPI controller in POAP
- Removed the accidental matplotlib dependency
- Fixed some printouts in the tests

v0.1.21, 2016-06-23
-------------------

- Added an option for supplying weights to the candidate point methods
- Cleaned up some of the tests by appending attributes to the workers
- Extended the MATLAB example to parallel
- Added a help function for doing a progress plot

v0.1.20, 2016-06-18
-------------------

- Added some basic input checking (evaluations, dimensionality, etc)
- Added an example with a MATLAB engine in case the optimization problems is in MATLAB
- Fixed a bug in the polynomial regression
- Moved the merit function out of sampling_methods.py

v0.1.19, 2016-01-30
-------------------

- Too much regularization was added to the RBF surface when the volume of the domain was large. This has been fixed.

v0.1.18, 2016-01-24
-------------------

- Significant restructuring of the code base
- make_points now takes an argument that specifies the number of new points to be generated
- Added Box-Behnken and 2-factorial to the experimental designs
- Simplified the penalty method strategy by moving evals and derivs into a surrogate wrapper

v0.1.17, 2016-01-13
-------------------

- Added the possibility to input the penalty for the penalty method in the GUI
- Added the possibility of making a performance plot using matplotlib that adds new points dynamically as evaluations are finished
- Switched from subprocess to subprocess32

v0.1.16, 2016-01-06
-------------------

- Added a projection strategy

v0.1.15, 2015-09-23
-------------------

- Added an example test_subprocess_files that shows how to use pySOT in case the objective function needs to read the input from a textfile

v0.1.14, 2015-09-22
-------------------

- Updated the Tutorial to reflect the changes for the last few months
- Simplified the object creation from strings in the GUI by importing directly from the namespace.

v0.1.13, 2015-09-03
-------------------

- Allowed to still import the rest of pySOT when PySide is not found. In this case, the GUI will be unavailable.

v0.1.12, 2015-07-23
-------------------

- The capping can now take in a general transformation that is used to transform the function values. Default is median capping.
- The Genetic Algorithm now defaults to initialize the population using a symmetric latin hypercube design
- DYCORS uses the remaining evaluation budget to change the probabilities after a restart instead of using the total budget

v0.1.11, 2015-07-22
-------------------

- Fixed a bug in the capped response surface
- pySOT now internally works on the unit hypercube
- The distance can be passed to the RBF after being computed when generating candidate points so it’s not computed twice anymore
- Fixed some bugs in the candidate functions
- GA and Multi-Search gradient perturb the best solution in the case when the best solution is a previously evaluated point
- Added an additional test for the multi-search strategy

v0.1.10, 2015-07-14
-------------------

- README.md not uploaded to pypi which caused pip install to fail

v0.1.9, 2015-07-13
------------------

- Fixed a bug in the merit function and several bugs in the DYCORS strategy
- Added a DDS candidate based strategy for searching on the surrogate

v0.1.8, 2015-07-01
------------------

- Multi Start Gradient method that uses the L-BFGS-B algorithm to search on the surroagate

v0.1.7, 2015-06-30
------------------

- Fixed some parameters (and bugs) to improve the DYCORS results. Using DYCORS together with the genetic algorithm is recommended.
- Added polynomial regression (not yet in the GUI)
- Changed so that candidate points are generated using truncated normal distribution to avoid projections onto the boundary
- Removed some accidental scikit dependencies in the ensemble surrogate

v0.1.6, 2015-06-28
------------------

- GUI inactivates all buttons but the stop button while running
- Bug fixes

v0.1.5, 2015-06-28
------------------

- GUI now has support for multiple search strategies and ensemble surrogates
- Reallocation bug in the ensemble surrogates fixed
- Genetic algorithm added to search on the surrogate

v0.1.4, 2015-06-26
------------------

- GUI now has improved error handling
- Strategies informs the user if they get constraints when not expecting constraints (and the other way) before the run starts

v0.1.3, 2015-06-26
------------------

- Experimental (but not documented) GUI added. You need PySide to use it.
- Changes in testproblems.py to allow external objective functions that implement ProcessWorkerThread
- Added GUI test examples in documentation (Ackley.py, Keane.py, SphereExt.py)

v0.1.2, 2015-06-24
------------------

- Changed to using the logging module for all the logging in order to conform to the changes in POAP 0.1.9
- The quiet and stream arguments in the strategies were removed and the tests updated accordingly
- Turned sleeping of in the subprocess test, to avoid platform dependency issues

v0.1.1, 2015-06-21
------------------

- surrogate_optimizer removed, so the user now has to create his own controller
- constraint_method.py is gone, and the constraint handling is handled in specific strategies instead
- There are now two strategies, SyncStrategyNoConstraints and SyncStrategyPenalty
- The search strategies now take a method for providing surrogate predictions rather than keeping a copy of the response surface
- It is now possible for the user to provide additional points to be added to the initial design, in case a 'good starting point' is known.
- Ensemble surrogates have been added to the toolbox
- The strategies takes an additional option 'quiet' so that all of the printing can be avoided if the user wants
- There is also an option 'stream' in case the printing should be redirected somewhere else, for example to a text file. Default is printing to stdout.
- Several examples added to pySOT.test

v0.1.0, 2015-06-03
------------------

- Initial release
