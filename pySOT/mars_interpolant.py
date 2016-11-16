"""
.. module:: mars_interpolant
   :synopsis: MARS model interpolation

.. moduleauthor:: Yi Shen <ys623@cornell.edu>

:Module: mars_interpolant
:Author: Yi Shen <ys623@cornell.edu>

"""

import numpy as np
import numpy.linalg as la
from pyearth import Earth


class MARSInterpolant(Earth):
    """Compute and evaluate a MARS interpolant

    MARS builds a model of the form

    .. math::

        \hat{f}(x) = \sum_{i=1}^{k} c_i B_i(x).

    The model is a weighted sum of basis functions :math:`B_i(x)`. Each basis
    function :math:`B_i(x)` takes one of the following three forms:

    1. a constant 1.
    2. a hinge function of the form :math:`\max(0, x - const)` or \
       :math:`\max(0, const - x)`. MARS automatically selects variables \
       and values of those variables for knots of the hinge functions.
    3. a product of two or more hinge functions. These basis functions c \
       an model interaction between two or more variables.

    :param maxp: Initial capacity
    :type maxp: int

    :ivar nump: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar x: Interpolation points
    :ivar fx: Function evaluations of interpolation points
    :ivar dim: Number of dimensions
    :ivar model: MARS interpolaion model
    """

    def __init__(self, maxp=100):
        self.nump = 0
        self.maxp = maxp
        self.x = None     # pylint: disable=invalid-name
        self.fx = None
        self.dim = None
        self.model = Earth()
        self.updated = False

    def reset(self):
        """Reset the interpolation."""

        self.nump = 0
        self.x = None
        self.fx = None
        self.updated = False

    def _alloc(self, dim):
        """Allocate storage for x, fx, rhs, and A.

        :param dim: Number of dimensions
        :type dim: int
        """

        maxp = self.maxp
        self.dim = dim
        self.x = np.zeros((maxp, dim))
        self.fx = np.zeros((maxp, 1))

    def _realloc(self, dim, extra=1):
        """Expand allocation to accommodate more points (if needed)

        :param dim: Number of dimensions
        :type dim: int
        :param extra: Number of additional points to accommodate
        :type extra: int
        """

        if self.nump == 0:
            self._alloc(dim)
        elif self.nump+extra > self.maxp:
            self.maxp = max(self.maxp*2, self.maxp+extra)
            self.x.resize((self.maxp, dim))
            self.fx.resize((self.maxp, 1))

    def get_x(self):
        """Get the list of data points

        :return: List of data points
        :rtype: numpy.array
        """

        return self.x[:self.nump, :]

    def get_fx(self):
        """Get the list of function values for the data points.

        :return: List of function values
        :rtype: numpy.array
        """

        return self.fx[:self.nump, :]

    def add_point(self, xx, fx):
        """Add a new function evaluation

        :param xx: Point to add
        :type xx: numpy.array
        :param fx: The function value of the point to add
        :type fx: float
        """

        dim = len(xx)
        self._realloc(dim)
        self.x[self.nump, :] = xx
        self.fx[self.nump, :] = fx
        self.nump += 1
        self.updated = False

    def eval(self, x, ds=None):
        """Evaluate the MARS interpolant at the point xx

        :param x: Point where to evaluate
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Value of the MARS interpolant at x
        :rtype: float
        """

        if self.updated is False:
            self.model.fit(self.x, self.fx)
        self.updated = True

        x = np.expand_dims(x, axis=0)
        fx = self.model.predict(x)
        return fx[0]

    def evals(self, x, ds=None):
        """Evaluate the MARS interpolant at the points x

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the MARS interpolant at x, of length npts
        :rtype: numpy.array
        """

        if self.updated is False:
            self.model.fit(self.x, self.fx)
        self.updated = True

        fx = np.zeros(shape=(x.shape[0], 1))
        fx[:, 0] = self.model.predict(x)
        return fx

    def deriv(self, x, ds=None):
        """Evaluate the derivative of the MARS interpolant at a point x

        :param x: Point for which we want to compute the MARS gradient
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Derivative of the MARS interpolant at x
        :rtype: numpy.array
        """

        if self.updated is False:
            self.model.fit(self.x, self.fx)
        self.updated = True

        x = np.expand_dims(x, axis=0)
        dfx = self.model.predict_deriv(x, variables=None)
        return dfx[0]

# ====================================================================


def _main():
    """Main test routine"""

    def test_f(x):
        """ Test function"""
        fx = x[1]*np.sin(x[0]) + x[0]*np.cos(x[1])
        return fx

    def test_df(x):
        """ Derivative of test function"""
        dfx = np.array([x[1]*np.cos(x[0]) + np.cos(x[1]),
                        np.sin(x[0]) - x[0]*np.sin(x[1])])
        return dfx

    # Set up Earth model
    fhat = MARSInterpolant(20)

    # Set up initial points to train the MARS model
    xs = np.random.rand(15, 2)
    for x in xs:
        fhat.add_point(x, test_f(x))

    x = np.random.rand(10, 2)
    fhx = fhat.evals(x)
    print(" \n------ (fx - fhx)/|fx| ----- ")
    for i in range(10):
        fx = test_f(x[i, :])
        print("Err: %e" % (abs(fx-fhx[i])/abs(fx)))

    print(" \n ------ (fx - fhx)/|fx| , |dfx-dfhx|/|dfx| -----")
    for i in range(10):
        xx = x[i, :]
        fx = test_f(xx)
        dfx = test_df(xx)
        fhx = fhat.eval(xx)
        dfhx = fhat.deriv(xx)
        print("Err (interp): %e : %e" % (abs(fx-fhx)/abs(fx),
                                         la.norm(dfx-dfhx)/la.norm(dfx)))

if __name__ == "__main__":
    _main()
