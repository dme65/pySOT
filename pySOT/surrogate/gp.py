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
    :param lb: Lower variable bounds
    :type lb: numpy.array
    :param ub: Upper variable bounds
    :type ub: numpy.array
    :param output_transformation: Transformation applied to values before fitting
    :type output_transformation: Callable
    :param gp: GPRegressor model
    :type gp: object
    :param n_restarts_optimizer: Number of restarts in hyperparam fitting
    :type n_restarts_optimizer: int

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar output_transformation: Transformation to apply to function values before fitting
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar model: GPRegressor object
    """

    def __init__(self, dim, lb, ub, output_transformation=None, gp=None, n_restarts_optimizer=5):
        super().__init__(dim=dim, lb=lb, ub=ub, output_transformation=output_transformation)

        if gp is None:  # Use the SE kernel
            kernel = ConstantKernel(1, (0.01, 100)) * RBF(
                length_scale=0.5 * np.ones(self.dim,), length_scale_bounds=(0.05, 2.0)
            ) + WhiteKernel(1e-4, (1e-6, 1e-2))
            self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
        else:
            self.model = gp
            if not isinstance(gp, GaussianProcessRegressor):
                raise TypeError("gp is not of type GaussianProcessRegressor")

    def _fit(self):
        """Compute new coefficients if the GP is not updated."""
        if not self.updated:
            fX = self.output_transformation(self.fX.copy())
            self._mu, self._sigma = np.mean(fX), max([np.std(fX), 1e-6])
            fX = (fX - self._mu) / self._sigma
            self.model.fit(self._X, fX)
            self.updated = True

    def predict(self, xx):
        """Evaluate the GP regressor at the points xx.

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = to_unit_box(np.atleast_2d(xx), self.lb, self.ub)
        return self._mu + self._sigma * self.model.predict(xx)

    def predict_std(self, xx):
        """Predict standard deviation at points xx.

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Predicted standard deviation, of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = to_unit_box(np.atleast_2d(xx), self.lb, self.ub)
        _, std = self.model.predict(xx, return_std=True)
        return self._sigma * np.expand_dims(std, axis=1)

    def predict_deriv(self, xx):
        """TODO: Not implemented"""
        raise NotImplementedError
