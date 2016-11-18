|Travis| |ReadTheDocs| |Gitter| |DOI|

pySOT: Surrogate Optimization Toolbox
-------------------------------------

pySOT is an asynchronous parallel optimization toolbox for global
deterministic optimization problems. The main purpose of the toolbox is
for optimization of computationally expensive black-box objective
functions with continuous and/or integer variables where the number of
evaluations is limited. If there are several processors available it may
make sense to evaluate the objective function using either asynchronous
or synchronous parallel. pySOT uses the event-driven framework for
asynchronous optimization strategies POAP
(https://github.com/dbindel/POAP) to provide this functionality.

The toolbox is hosted on GitHub: https://github.com/dme65/pySOT

Documentation: http://pysot.readthedocs.io/

pySOT has been downloaded 16,001 times from 2015-June-4 to
2016-October-15

Installation
------------

Make sure you have Python 2.7.x and pip installed. The easiest way to
install pySOT is using:

.. code:: bash

    pip install pySOT

Examples
--------

Several pySOT examples can be found at:
https://github.com/dme65/pySOT/tree/master/pySOT/test

News
----

A two-hour short course on how to use pySOT was given at the CMWR 2016
conference in Toronto. The slides and Python notebooks can be downloaded
from: https://people.cam.cornell.edu/~dme65/talks.html

Check out the new C++ implementation of pySOT:
https://github.com/dme65/SOT

Check out the new pySOT documentation: http://pysot.readthedocs.io/

FAQ
---

| Q: I can't find the GUI
| A: You need to install PySide
|
| Q: I can't find the MARS interpolant
| A: You need to install py-earth (https://github.com/jcrudy/py-earth)
|
| Q: I used pySOT for my research and want to cite it
| A: There is currently no published paper on pySOT so we recommend
  citing pySOT like this: *D. Eriksson, D. Bindel, and C. Shoemaker.
  Surrogate Optimization Toolbox (pySOT). github.com/dme65/pySOT, 2015*
|
| Q: Is there support for Python 3?
| A: pySOT currently doesn't support Python 3, mainly because of some
  pySOT dependencies lacking Python 3 support.

.. |Travis| image:: https://travis-ci.org/dme65/pySOT.svg?branch=master
   :target: https://travis-ci.org/dme65/pySOT
.. |ReadTheDocs| image:: https://readthedocs.org/projects/pysot/badge/?version=latest
    :target: http://pysot.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |Gitter| image:: https://badges.gitter.im/dme65/pySOT.svg
   :target: https://gitter.im/pySOT/Lobby
.. |DOI| image:: https://zenodo.org/badge/36836292.svg
   :target: https://zenodo.org/badge/latestdoi/36836292
