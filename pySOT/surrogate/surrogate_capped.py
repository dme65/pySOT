import numpy as np

from .surrogate import Surrogate


class SurrogateCapped(Surrogate):
    """Wrapper for tranformation of function values.

    This adapter takes an existing surrogate model and replaces it
    with a modified version where the function values are replaced
    according to some transformation. A common transformation
    is replacing all values above the median by the median
    to reduce the influence of large function values.

    :param model: Original surrogate model (must implement Surrogate)
    :type model: object
    :param transformation: Function that transforms the function values
    :type transformation: function

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar model: scikit-learn pipeline for polynomial regression
    :ivar transformation: Transformation function
    """

    def __init__(self, model, transformation=None):
        self.num_pts = 0
        self.X = np.empty([0, model.dim])
        self.fX = np.empty([0, 1])
        self.dim = model.dim
        self.updated = False

        self.transformation = transformation
        if self.transformation is None:

            def median_transformation(fvalues):
                medf = np.median(fvalues)
                fvalues[fvalues > medf] = medf
                return fvalues

            self.transformation = median_transformation

        assert isinstance(model, Surrogate)
        self.model = model

    def reset(self):
        """Reset the surrogate."""
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
        self.model.add_points(xx, fx)
        # Apply transformation
        self.model.fX = self.transformation(np.copy(self.fX))

    def predict(self, xx):
        """Evaluate the surrogate model at the points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        return self.model.predict(xx)

    def predict_std(self, xx):
        """Predict standard deviation at points xx.

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Predicted standard deviation, of size num_pts x 1
        :rtype: numpy.ndarray
        """
        return self.model.predict_std(xx)

    def predict_deriv(self, xx):
        """Evaluate the derivative of the surrogate model at points xx.

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.array

        :return: Derivative of the RBF interpolant at xx
        :rtype: numpy.array
        """
        return self.model.predict_deriv(xx)
