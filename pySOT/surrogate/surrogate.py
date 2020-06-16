"""
.. module:: surrogate
   :synopsis: Surrogate models

.. moduleauthor:: David Eriksson <dme65@cornell.edu>

:Module: surrogate
:Author: David Eriksson <dme65@cornell.edu>

"""

from abc import ABC, abstractmethod

import numpy as np


class Surrogate(ABC):
    """Base class for a surrogate model.

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    """

    def __init__(self):  # pragma: no cover
        self.dim = None
        self.num_pts = None
        self.X = None
        self.fX = None
        self.updated = None

    def reset(self):
        """Reset the surrogate."""
        self.num_pts = 0
        self.X = np.empty([0, self.dim])
        self.fX = np.empty([0, 1])
        self.updated = False

    def add_points(self, xx, fx):
        """Add new function evaluations.

        This method SHOULD NOT trigger a new fit, it just updates X
        and fX but leaves the original surrogate object intact

        :param xx: Points to add
        :type xx: numpy.ndarray
        :param fx: The function values of the point to add
        :type fx: numpy.array or float
        """
        xx = np.atleast_2d(xx)
        if isinstance(fx, float):
            fx = np.array([fx])
        if fx.ndim == 0:
            fx = np.expand_dims(fx, axis=0)
        if fx.ndim == 1:
            fx = np.expand_dims(fx, axis=1)
        assert xx.shape[0] == fx.shape[0] and xx.shape[1] == self.dim
        newpts = xx.shape[0]
        self.X = np.vstack((self.X, xx))
        self.fX = np.vstack((self.fX, fx))
        self.num_pts += newpts
        self.updated = False

    @abstractmethod
    def predict(self, xx):  # pragma: no cover
        """Evaluate surroagte at points xx.

        :param xx: xx must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Surrogate predictions, of size num_pts x 1
        :rtype: numpy.ndarray
        """
        return

    @abstractmethod
    def predict_deriv(self, xx):  # pragma: no cover
        """Evaluate derivative of interpolant at points xx.

        :param xx: xx must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Surrogate derivative predictions, of size num_pts x dim
        :rtype: numpy.ndarray
        """
        return
