import warnings

import numpy as np

from ..utils import to_unit_box
from .surrogate import Surrogate


class MARSInterpolant(Surrogate):
    """Compute and evaluate a MARS interpolant

    MARS builds a model of the form

    .. math::

        \\hat{f}(x) = \\sum_{i=1}^{k} c_i B_i(x).

    The model is a weighted sum of basis functions :math:`B_i(x)`. Each basis
    function :math:`B_i(x)` takes one of the following three forms:

    1. a constant 1.
    2. a hinge function of the form :math:`\\max(0, x - const)` or \
       :math:`\\max(0, const - x)`. MARS automatically selects variables \
       and values of those variables for knots of the hinge functions.
    3. a product of two or more hinge functions. These basis functions c \
       an model interaction between two or more variables.

    :param dim: Number of dimensions
    :type dim: int
    :param lb: Lower variable bounds
    :type lb: numpy.array
    :param ub: Upper variable bounds
    :type ub: numpy.array
    :param output_transformation: Transformation applied to values before fitting
    :type output_transformation: Callable

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar output_transformation: Transformation to apply to function values before fitting
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar model: Earth object
    """

    def __init__(self, dim, lb, ub, output_transformation=None):
        super().__init__(dim=dim, lb=lb, ub=ub, output_transformation=output_transformation)

        try:
            from pyearth import Earth

            self.model = Earth()
        except ImportError as err:
            print("Failed to import pyearth")
            raise err

    def _fit(self):
        """Compute new coefficients if the MARS interpolant is not updated."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Surpress deprecation warnings
            if self.updated is False:
                fX = self.output_transformation(self.fX.copy())
                self.model.fit(self._X, fX)
                self.updated = True

    def predict(self, xx):
        """Evaluate the MARS interpolant at the points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = to_unit_box(np.atleast_2d(xx), self.lb, self.ub)
        return np.expand_dims(self.model.predict(xx), axis=1)

    def predict_deriv(self, xx):
        """Evaluate the derivative of the MARS interpolant at points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.array

        :return: Derivative of the RBF interpolant at xx
        :rtype: numpy.array
        """
        self._fit()
        xx = to_unit_box(np.atleast_2d(xx), self.lb, self.ub)
        dfx = self.model.predict_deriv(xx, variables=None)
        return dfx[0] / (self.ub - self.lb)
