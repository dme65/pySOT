|Travis| |codecov| |ReadTheDocs| |Downloads| |DOI|

pySOT: Surrogate Optimization Toolbox
-------------------------------------

pySOT is an asynchronous parallel optimization toolbox for global
deterministic optimization problems. The main purpose of the toolbox is
optmizing computationally expensive black-box objective
functions with continuous and/or integer variables given a limited number of
function evaluations. pySOT supports synchronous and asynchronous parallel
function evaluations, both using threads and MPI. This functionality is provided 
by the event-driven POAP (https://github.com/dbindel/POAP) framework.

The toolbox is hosted on GitHub: https://github.com/dme65/pySOT

Documentation: http://pysot.readthedocs.io/


Installation
------------

Installation instructions are available at: http://pysot.readthedocs.io/en/latest/quickstart.html

Examples
--------

Several pySOT examples can be found at:
https://github.com/dme65/pySOT/tree/master/pySOT/test

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