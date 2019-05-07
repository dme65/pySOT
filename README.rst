|Travis| |codecov| |ReadTheDocs| |Downloads| |DOI|

pySOT: Python Surrogate Optimization Toolbox
--------------------------------------------

The Python Surrogate Optimization Toolbox (pySOT) is an asynchronous parallel
optimization toolbox for computationally expensive global optimization problems.
pySOT is built on top of the Plumbing for Optimization with Asynchronous Parallelism (POAP),
which is an event-driven framework for building and combining asynchronous optimization
strategies. POAP has support for both threads and MPI.

pySOT implements many popular surrogate optimization algorithms such as the
Stochastic RBF (SRBF) and DYCORS methods by Regis and Shoemaker, and the SOP
method by Krityakierne et. al. We also support Expected Improvement (EI) and
Lower Confidence Bounds (LCB), which are popular in Bayesian optimization. All
optimization algorithms can be used in serial, synchronous parallel, and
asynchronous parallel and we support both continuous and integer variables.

The toolbox is hosted on GitHub: https://github.com/dme65/pySOT

Documentation: http://pysot.readthedocs.io/

Installation
------------

Installation instructions are available at: http://pysot.readthedocs.io/en/latest/quickstart.html

Examples
--------

Several pySOT examples and notebooks can be found at:

https://github.com/dme65/pySOT/tree/master/pySOT/examples

https://github.com/dme65/pySOT/tree/master/pySOT/notebooks


News
----

pySOT 0.2.0 has finally been released!

FAQ
---

| Q: Can I use pySOT with MPI?
| A: Yes. You need to install mpi4py in order to use the MPIController in POAP.
|
| Q: I used pySOT for my research and want to cite it
| A: There is currently no published paper on pySOT so we recommend
  citing pySOT like this: *D. Eriksson, D. Bindel, and C. Shoemaker.
  Surrogate Optimization Toolbox (pySOT). github.com/dme65/pySOT, 2015*
|
| Q: Is there support for Python 2?
| A: Python 2 support was removed in version 0.2.0
|
| Q: I can't find the MARS interpolant
| A: You need to install py-earth in order to use MARS. More information is
  available here: https://github.com/scikit-learn-contrib/py-earth
|

.. |Travis| image:: https://travis-ci.org/dme65/pySOT.svg?branch=master
   :target: https://travis-ci.org/dme65/pySOT
.. |ReadTheDocs| image:: https://readthedocs.org/projects/pysot/badge/?version=latest
    :target: http://pysot.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |DOI| image:: https://zenodo.org/badge/36836292.svg
   :target: https://zenodo.org/badge/latestdoi/36836292
.. |codecov| image:: https://codecov.io/gh/dme65/pySOT/branch/dme/graph/badge.svg
   :target: https://codecov.io/gh/dme65/pySOT
.. |Downloads| image:: https://pepy.tech/badge/pysot
   :target: https://pepy.tech/project/pySOT
