import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from ..utils import to_unit_box
from .surrogate import Surrogate


class GPRegressor(Surrogate):
    """Gaussian process (GP) regressor.

    Wrapper around the GPRegressor in scikit-learn.

    :param dim: Number of dimensions
    :type dim: int
    :param gp: GPRegressor model
    :type gp: object
    :param n_restarts_optimizer: Number of restarts in hyperparam fitting
    :type n_restarts_optimizer: int

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar model: GPRegressor object
    """

    def __init__(self, dim, gp=None, n_restarts_optimizer=5):
        self.num_pts = 0
        self.dim = dim
        self.X = np.empty([0, dim])  # pylint: disable=invalid-name
        self.fX = np.empty([0, 1])
        self.updated = False

        if gp is None:  # Use the SE kernel
            kernel = ConstantKernel(1, (0.01, 100)) * RBF(
                length_scale=0.5, length_scale_bounds=(0.05, 2.0)
            ) + WhiteKernel(1e-4, (1e-6, 1e-2))
            self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
        else:
            self.model = gp
            if not isinstance(gp, GaussianProcessRegressor):
                raise TypeError("gp is not of type GaussianProcessRegressor")

    def _fit(self):
        """Compute new coefficients if the GP is not updated."""
        if not self.updated:
            self._lb, self._ub = self.X.min(axis=0), self.X.max(axis=0)
            X = to_unit_box(self.X.copy(), self._lb, self._ub)
            self._mu, self._sigma = np.mean(self.fX), max([np.std(self.fX), 1e-6])
            fX = (self.fX - self._mu) / self._sigma
            self.model.fit(X, fX)
            self.updated = True

    def predict(self, xx):
        """Evaluate the GP regressor at the points xx.

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = to_unit_box(np.atleast_2d(xx), self._lb, self._ub)
        return self._mu + self._sigma * self.model.predict(xx)

    def predict_std(self, xx):
        """Predict standard deviation at points xx.

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Predicted standard deviation, of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = to_unit_box(np.atleast_2d(xx), self._lb, self._ub)
        _, std = self.model.predict(xx, return_std=True)
        return self._sigma * np.expand_dims(std, axis=1)

    def predict_deriv(self, xx):
        """TODO: Not implemented"""
        raise NotImplementedError
