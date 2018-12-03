Quickstart
==========

.. _quickstart-label:


Dependencies
------------

Before starting you will need Python 3.4 or newer. You need to have numpy, scipy, and pip
installed and we recommend installing Anaconda/Miniconda for your desired Python version.

There are a couple of optional components of pySOT that needs to be installed manually:

1. **py-earth**: Implementation of MARS. Can be installed using:

   .. code-block:: bash

      pip install six http://github.com/scikit-learn-contrib/py-earth/tarball/master

   or

   .. code-block:: bash

      git clone git://github.com/scikit-learn-contrib/py-earth.git
      cd py-earth
      pip install six
      python setup.py install

2. **mpi4py**: This module is necessary in order to use pySOT with MPI. Can be installed through pip:

   .. code-block:: bash

      pip install mpi4py

   or through conda (Anaconda/Miniconda) where it can be channeled with your favorite MPI implementation
   such as mpich:

   .. code-block:: bash

      conda install --channel mpi4py mpich mpi4py

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

Several examples are available in ./pySOT/examples and ./pySOT/notebooks