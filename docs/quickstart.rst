Quickstart
==========



Dependencies
------------

Before starting you will need Python 2.7.x and pip. You must also have numpy and scipy
installed and we recommend installing Anaconda for Python 2.7:
`https://www.continuum.io/downloads <https://www.continuum.io/downloads>`_

If you want to use the GUI you need to install PySide. This can be done with pip:

.. code-block:: bash

   pip install PySide

Installation
------------

There are currently two ways to install pySOT:

1. **(Recommended)** The easiest way to install pySOT is through pip in which case
   the following command should suffice:

   .. code-block:: bash

      pip install pySOT

2. The other option is cloning the repository and installing.

|  2.1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/dme65/pySOT

|  2.2. Navigate to the repository using:

   .. code-block:: bash

      cd pySOT

|  2.3. Install pySOT (you may need to use sudo for UNIX):

   .. code-block:: bash

      python setup.py install

Several examples problems are available at ./pySOT/test or in the pySOT.test module