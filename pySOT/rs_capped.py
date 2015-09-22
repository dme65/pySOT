"""
.. module:: rs_capped
   :synopsis: Median-capped interpolation
.. moduleauthor:: David Bindel <bindel@cornell.edu>

:Module: rs_capped
:Author: David Bindel <bindel@cornell.edu>
"""

import numpy as np


class RSCapped(object):
    """Cap adapter for RBF response surface.

    This adapter takes an existing response surface and replaces it
    with a modified version in which any function values above the
    median are replaced by the median value.

    :ivar model: Original response surface
    :ivar fvalues: Function values
    """

    def __init__(self, model, transformation=None):
        """Initialize the response surface adapter

        :param model: Original response surface object
        :param transformation: Function value transformation
        """
        self.needs_update = False
        self.transformation = transformation
        if self.transformation is None:
            def transformation(fvalues):
                medf = np.median(fvalues)
                fvalues[fvalues > medf] = medf
                return fvalues
            self.transformation = transformation
        self.model = model
        self.fvalues = np.zeros((100, 1))
        self.nump = 0

    @property
    def x(self):
        return self.get_x()

    @property
    def fx(self):
        return self.get_fx()

    def reset(self):
        """Reset the capped response surface
        """
        self.model.reset()
        self.fvalues[:] = 0
        self.nump = 0

    def add_point(self, xx, fx):
        """Add a new function evaluation

        :param xx: Point to add
        :param fx: The function value of the point to add
        """
        if self.nump >= self.fvalues.shape[0]:
            self.fvalues.resize(2*self.fvalues.shape[0], 1)
        self.fvalues[self.nump] = fx
        self.nump += 1
        self.needs_update = True
        self.model.add_point(xx, fx)

    def get_x(self):
        """Get the list of data points

        :return: List of data points
        """
        return self.model.get_x()

    def get_fx(self):
        """Get the list of function values for the data points.

        :return: List of function values
        """
        return self.model.get_fx()

    def eval(self, xx, d=None):
        """Evaluate the capped rbf interpolant at the point xx

        :param xx: Point where to evaluate
        :return: Value of the capped rbf interpolant at x
        """
        self._apply_cap()
        return self.model.eval(xx, d)

    def evals(self, xx, d=None):
        """Evaluate the capped rbf interpolant at the points xx

        :param xx: Points where to evaluate
        :return: Values of the capped rbf interpolant at x
        """
        self._apply_cap()
        return self.model.evals(xx, d)

    def deriv(self, xx, d=None):
        """Evaluate the derivative of the rbf interpolant at x

        :param x: Data point
        :return: Derivative of the rbf interpolant at x
        """
        self._apply_cap()
        return self.model.deriv(xx, d)

    def _apply_cap(self):
        """ Apply the cap to the function values.
        """
        fvalues = np.copy(self.fvalues[0:self.nump])
        self.model.transform_fx(self.transformation(fvalues))
