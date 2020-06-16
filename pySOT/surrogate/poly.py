import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from .surrogate import Surrogate


class PolyRegressor(Surrogate):
    """Multi-variate polynomial regression with cross-terms

    :param dim: Number of dimensions
    :type dim: int
    :param degree: Polynomial degree
    :type degree: int

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar model: scikit-learn pipeline for polynomial regression
    """

    def __init__(self, dim, degree=2):
        self.num_pts = 0
        self.X = np.empty([0, dim])
        self.fX = np.empty([0, 1])
        self.dim = dim
        self.updated = False
        self.model = make_pipeline(PolynomialFeatures(degree), Ridge())

    def _fit(self):
        """Update the polynomial regression model."""
        if not self.updated:
            self.model.fit(self.X, self.fX)
            self.updated = True

    def predict(self, xx):
        """Evaluate the polynomial regressor at the points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = np.atleast_2d(xx)
        return self.model.predict(xx)

    def predict_deriv(self, xx):
        """TODO: Not implemented"""
        raise NotImplementedError
