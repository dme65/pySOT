import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from ..utils import to_unit_box
from .surrogate import Surrogate


class PolyRegressor(Surrogate):
    """Multi-variate polynomial regression with cross-terms

    :param dim: Number of dimensions
    :type dim: int
    :param lb: Lower variable bounds
    :type lb: numpy.array
    :param ub: Upper variable bounds
    :type ub: numpy.array
    :param output_transformation: Transformation applied to values before fitting
    :type output_transformation: Callable
    :param degree: Polynomial degree
    :type degree: int

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar output_transformation: Transformation to apply to function values before fitting
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar model: scikit-learn pipeline for polynomial regression
    """

    def __init__(self, dim, lb, ub, output_transformation=None, degree=2):
        super().__init__(dim=dim, lb=lb, ub=ub, output_transformation=output_transformation)
        self.model = make_pipeline(PolynomialFeatures(degree), Ridge())

    def _fit(self):
        """Update the polynomial regression model."""
        if not self.updated:
            fX = self.output_transformation(self.fX.copy())
            self.model.fit(self._X, fX)
            self.updated = True

    def predict(self, xx):
        """Evaluate the polynomial regressor at the points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = to_unit_box(np.atleast_2d(xx), self.lb, self.ub)
        return self.model.predict(xx)

    def predict_deriv(self, xx):
        """TODO: Not implemented"""
        raise NotImplementedError
