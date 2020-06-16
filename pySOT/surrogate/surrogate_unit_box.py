import numpy as np

from ..utils import to_unit_box
from .surrogate import Surrogate


class SurrogateUnitBox(Surrogate):
    """Unit box adapter for surrogate models.

    This adapter takes an existing surrogate model and replaces it
    by a modified version where the domain is rescaled to the unit
    hypercube. This is useful for surrogate models that are sensitive to
    scaling, such as RBFs.

    :param model: Original surrogate model (must implement Surrogate)
    :type model: object
    :param lb: Lower variable bounds, of size 1 x dim
    :type lb: numpy.array
    :param ub: Upper variable bounds, of size 1 x dim
    :type ub: numpy.array

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar model: scikit-learn pipeline for polynomial regression
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    """

    def __init__(self, model, lb, ub):
        self.num_pts = 0
        self.X = np.empty([0, model.dim])
        self.fX = np.empty([0, 1])
        self.dim = model.dim
        self.updated = False

        assert isinstance(model, Surrogate)
        self.model = model
        self.lb = lb
        self.ub = ub

    def reset(self):
        """Reset the surrogate model."""
        super().reset()
        self.model.reset()

    def add_points(self, xx, fx):
        """Add new function evaluations.

        This method SHOULD NOT trigger a new fit, it just updates X and
        fX but leaves the original surrogate object intact

        :param xx: Points to add
        :type xx: numpy.ndarray
        :param fx: The function values of the point to add
        :type fx: numpy.array or float
        """
        super().add_points(xx, fx)
        self.model.add_points(to_unit_box(xx, self.lb, self.ub), fx)

    def predict(self, xx):
        """Evaluate the surrogate model at the points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        return self.model.predict(to_unit_box(xx, self.lb, self.ub))

    def predict_std(self, x):
        """Predict standard deviation at points xx.

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Predicted standard deviation, of size num_pts x 1
        :rtype: numpy.ndarray
        """
        return self.model.predict_std(to_unit_box(x, self.lb, self.ub))

    def predict_deriv(self, x):
        """Evaluate the derivative of the surrogate model at points xx

        Remember the chain rule:
            f'(x) = (d/dx) g((x-a)/(b-a)) = g'((x-a)/(b-a)) * 1/(b-a)

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.array

        :return: Derivative of the RBF interpolant at xx
        :rtype: numpy.array
        """
        return self.model.predict_deriv(to_unit_box(x, self.lb, self.ub)) / (self.ub - self.lb)
