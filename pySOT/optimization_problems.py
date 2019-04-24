"""
.. module:: optimization_problems
  :synopsis: Optimization test problems for multi-modal and
             box-constrained global optimization

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,
                 David Bindel <bindel@cornell.edu>

:Module: optimization_problems
:Author: David Eriksson <dme65@cornell.edu>,
        David Bindel <bindel@cornell.edu>
"""

import numpy as np
import abc
from abc import abstractmethod


class OptimizationProblem(object):
    """Base class for optimization problems."""
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.dim = None
        self.lb = None
        self.ub = None
        self.int_var = None
        self.cont_var = None

    def __check_input__(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')

    @abstractmethod
    def eval(self, record):  # pragma: no cover
        pass

# ========================= 2-dimensional =======================
