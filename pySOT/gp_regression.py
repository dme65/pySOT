"""
.. module:: gp_regression
   :synopsis: Gaussian Process regression

.. moduleauthor:: David Eriksson <dme65@cornell.edu>

:Module: gp_regression
:Author: David Eriksson <dme65@cornell.edu>

"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class GPRegression(GaussianProcessRegressor):
    """Compute and evaluate a GP

    Gaussian Process Regression object.

    Depends on scitkit==0.18.1.

    More details:
        http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html

    :param maxp: Initial capacity
    :type maxp: int
    :param gp: GP object (can be None)
    :type gp: GaussianProcessRegressor

    :ivar nump: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar x: Interpolation points
    :ivar fx: Function evaluations of interpolation points
    :ivar gp: Object of type GaussianProcessRegressor
    :ivar dim: Number of dimensions
    :ivar model: MARS interpolation model
    """

    def __init__(self, maxp=100, gp=None):
        self.nump = 0
        self.maxp = maxp
        self.x = None     # pylint: disable=invalid-name
        self.fx = None
        self.dim = None
        if gp is None:
            self.model = GaussianProcessRegressor(n_restarts_optimizer=10)
        else:
            self.model = gp
            if not isinstance(gp, GaussianProcessRegressor):
                raise TypeError("gp is not of type GaussianProcessRegressor")
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
        """Evaluate the GP regression object at the point x

        :param x: Point where to evaluate
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Value of the GP regression obejct at x
        :rtype: float
        """

        if self.updated is False:
            self.model.fit(self.get_x(), self.get_fx())
        self.updated = True

        x = np.expand_dims(x, axis=0)
        fx = self.model.predict(x)
        return fx

    def evals(self, x, ds=None):
        """Evaluate the GP regression object at the points x

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the GP regression object at x, of length npts
        :rtype: numpy.array
        """

        if self.updated is False:
            self.model.fit(self.get_x(), self.get_fx())
        self.updated = True

        fx = self.model.predict(x)
        return fx

    def deriv(self, x, ds=None):
        """Evaluate the GP regression object at a point x

        :param x: Point for which we want to compute the GP regression gradient
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Derivative of the GP regression object at x
        :rtype: numpy.array
        """

        # FIXME, To be implemented
        raise NotImplementedError

# ====================================================================


def _main():
    """Main test routine"""

    def test_f(x):
        """ Test function"""
        fx = x[1]*np.sin(x[0]) + x[0]*np.cos(x[1])
        return fx

    # Set up GP model
    fhat = GPRegression(20)

    # Set up initial points to train the MARS model
    xs = np.random.rand(15, 2)
    for xx in xs:
        fhat.add_point(xx, test_f(xx))

    x = np.random.rand(10, 2)
    fhx = fhat.evals(x)
    print(" \nGP with RBF kernel\n ------ (fx - fhx)/|fx| ----- ")
    for i in range(10):
        fx = test_f(x[i, :])
        print("Err: %e" % (abs(fx-fhx[i])/abs(fx)))

    # Try to pass in a GP object with a different kernel
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    fhat2 = GPRegression(20, gp)

    for xx in xs:
        fhat2.add_point(xx, test_f(xx))

    fhx = fhat2.evals(x)
    print(" \nGP with RBF + white kernel\n ------ (fx - fhx)/|fx| ----- ")
    for i in range(10):
        fx = test_f(x[i, :])
        print("Err: %e" % (abs(fx - fhx[i]) / abs(fx)))

if __name__ == "__main__":
    _main()
