"""
.. module:: rs_wrappers
   :synopsis: Response surface wrappers
.. moduleauthor:: David Bindel <bindel@cornell.edu>

:Module: rs_wrappers
:Author: David Bindel <bindel@cornell.edu>
"""

import numpy as np
from pySOT.utils import from_unit_box, to_unit_box


class RSCapped(object):
    """Cap adapter for response surfaces.

    This adapter takes an existing response surface and replaces it
    with a modified version in which the function values are replaced
    according to some transformation. A very common transformation
    is to replace all values above the median by the median in order
    to reduce the influence of large function values.

    :param model: Original response surface object
    :type model: Object
    :param transformation: Function value transformation object. Median capping
        is used if no object (or None) is provided
    :type transformation: Object

    :ivar transformation: Object used to transform the function values.
    :ivar model: original response surface object
    :ivar fvalues: Function values
    :ivar nump: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar updated: True if the surface is updated
    """

    def __init__(self, model, transformation=None):

        self.transformation = transformation
        if self.transformation is None:
            def transformation(fvalues):
                medf = np.median(fvalues)
                fvalues[fvalues > medf] = medf
                return fvalues
            self.transformation = transformation
        self.model = model
        self.fvalues = np.zeros((model.maxp, 1))
        self.nump = 0
        self.maxp = model.maxp
        self.updated = True

    def reset(self):
        """Reset the capped response surface"""

        self.model.reset()
        self.fvalues[:] = 0
        self.nump = 0

    def add_point(self, xx, fx):
        """Add a new function evaluation

        :param xx: Point to add
        :type xx: numpy.array
        :param fx: The function value of the point to add
        :type fx: float
        """

        if self.nump >= self.fvalues.shape[0]:
            self.fvalues.resize(2*self.fvalues.shape[0], 1)
        self.fvalues[self.nump] = fx
        self.nump += 1
        self.updated = False
        self.model.add_point(xx, fx)

    def get_x(self):
        """Get the list of data points

        :return: List of data points
        :rtype: numpy.array
        """

        return self.model.get_x()

    def get_fx(self):
        """Get the list of function values for the data points.

        :return: List of function values
        :rtype: numpy.array
        """

        return self.model.get_fx()

    def eval(self, x, ds=None):
        """Evaluate the capped interpolant at the point x

        :param x: Point where to evaluate
        :type x: numpy.array
        :return: Value of the RBF interpolant at x
        :rtype: float
        """

        self._apply_transformation()
        return self.model.eval(x, ds)

    def evals(self, x, ds=None):
        """Evaluate the capped interpolant at the points x

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Distances between the centers and the points x, of size npts x ncenters
        :type ds: numpy.array
        :return: Values of the capped interpolant at x, of length npts
        :rtype: numpy.array
        """

        self._apply_transformation()
        return self.model.evals(x, ds)

    def deriv(self, x, ds=None):
        """Evaluate the derivative of the capped interpolant at a point x

        :param x: Point for which we want to compute the RBF gradient
        :type x: numpy.array
        :param ds: Distances between the centers and the point x
        :type ds: numpy.array
        :return: Derivative of the capped interpolant at x
        :rtype: numpy.array
        """

        self._apply_transformation()
        return self.model.deriv(x, ds)

    def _apply_transformation(self):
        """ Apply the cap to the function values."""

        fvalues = np.copy(self.fvalues[0:self.nump])
        self.model.transform_fx(self.transformation(fvalues))


class RSPenalty(object):
    """Penalty adapter for response surfaces.

    This adapter can be used for approximating an objective function plus
    a penalty function. The response surface is fitted only to the objective
    function and the penalty is added on after.

    :param model: Original response surface object
    :type model: Object
    :param evals: Object that takes the response surface and the points and adds up
        the response surface value and the penalty function value
    :type evals: Object
    :param devals: Object that takes the response surface and the points and adds up
        the response surface derivative and the penalty function derivative
    :type devals: Object

    :ivar eval_method: Object that takes the response surface and the points and adds up
        the response surface value and the penalty function value
    :ivar deval_method: Object that takes the response surface and the points and adds up
        the response surface derivative and the penalty function derivative
    :ivar model: original response surface object
    :ivar fvalues: Function values
    :ivar nump: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar updated: True if the surface is updated
    """

    def __init__(self, model, evals, derivs):

        self.model = model
        self.fvalues = np.zeros((model.maxp, 1))
        self.nump = 0
        self.maxp = model.maxp
        self.eval_method = evals
        self.deriv_method = derivs
        self.updated = True

    def reset(self):
        """Reset the capped response surface"""

        self.model.reset()
        self.fvalues[:] = 0
        self.nump = 0

    def add_point(self, xx, fx):
        """Add a new function evaluation

        :param xx: Point to add
        :type xx: numpy.array
        :param fx: The function value of the point to add
        :type fx: float
        """

        if self.nump >= self.fvalues.shape[0]:
            self.fvalues.resize(2*self.fvalues.shape[0], 1)
        self.fvalues[self.nump] = fx
        self.nump += 1
        self.updated = False
        self.model.add_point(xx, fx)

    def get_x(self):
        """Get the list of data points

        :return: List of data points
        :rtype: numpy.array
        """

        return self.model.get_x()

    def get_fx(self):
        """Get the list of function values for the data points.

        :return: List of function values
        :rtype: numpy.array
        """

        return self.eval_method(self.model, self.model.get_x())[0, 0]

    def eval(self, x, ds=None):
        """Evaluate the penalty adapter interpolant at the point xx

        :param x: Point where to evaluate
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Value of the interpolant at x
        :rtype: float
        """

        return self.eval_method(self.model, np.atleast_2d(x)).ravel()

    def evals(self, x, ds=None):
        """Evaluate the penalty adapter at the points xx

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the interpolant at x, of length npts
        :rtype: numpy.array
        """

        return self.eval_method(self.model, x)

    def deriv(self, x, ds=None):
        """Evaluate the derivative of the penalty adapter at x

        :param x: Point for which we want to compute the gradient
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Derivative of the interpolant at x
        :rtype: numpy.array
        """

        return self.deriv_method(self.model, x)


class RSUnitbox(object):
    """Unit box adapter for response surfaces

    This adapter takes an existing response surface and replaces it
    with a modified version where the domain is rescaled to the unit
    box. This is useful for response surfaces that are sensitive to
    scaling, such as radial basis functions.

    :param model: Original response surface object
    :type model: Object
    :param data: Optimization problem object
    :type data: Object

    :ivar data: Optimization problem object
    :ivar model: original response surface object
    :ivar fvalues: Function values
    :ivar nump: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar updated: True if the surface is updated
    """

    def __init__(self, model, data):

        self.model = model
        self.fvalues = np.zeros((model.maxp, 1))
        self.nump = 0
        self.maxp = model.maxp
        self.data = data
        self.updated = True

    def reset(self):
        """Reset the capped response surface"""

        self.model.reset()
        self.fvalues[:] = 0
        self.nump = 0

    def add_point(self, xx, fx):
        """Add a new function evaluation

        :param xx: Point to add
        :type xx: numpy.array
        :param fx: The function value of the point to add
        :type fx: float
        """

        if self.nump >= self.fvalues.shape[0]:
            self.fvalues.resize(2*self.fvalues.shape[0], 1)
        self.fvalues[self.nump] = fx
        self.nump += 1
        self.updated = False
        self.model.add_point(to_unit_box(xx, self.data), fx)

    def get_x(self):
        """Get the list of data points

        :return: List of data points
        :rtype: numpy.array
        """

        return from_unit_box(self.model.get_x(), self.data)

    def get_fx(self):
        """Get the list of function values for the data points.

        :return: List of function values
        :rtype: numpy.array
        """

        return self.model.get_fx()

    def eval(self, x, ds=None):
        """Evaluate the response surface at the point xx

        :param x: Point where to evaluate
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Value of the interpolant at x
        :rtype: float
        """

        return self.model.eval(to_unit_box(x, self.data), ds)

    def evals(self, x, ds=None):
        """Evaluate the capped rbf interpolant at the points xx

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the MARS interpolant at x, of length npts
        :rtype: numpy.array
        """

        return self.model.evals(to_unit_box(x, self.data), ds)

    def deriv(self, x, ds=None):
        """Evaluate the derivative of the rbf interpolant at x

        :param x: Point for which we want to compute the MARS gradient
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Derivative of the MARS interpolant at x
        :rtype: numpy.array
        """

        return self.model.deriv(to_unit_box(x, self.data), ds)
