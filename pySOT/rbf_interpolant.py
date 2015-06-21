"""
.. module:: rbf_interpolant
   :synopsis: Basic radial basis function interpolation
.. moduleauthor:: David Bindel <bindel@cornell.edu>

:Module: rbf_interpolant
:Author: David Bindel <bindel@cornell.edu>
"""

import numpy as np
import numpy.linalg as la
import scipy.spatial as scp


class RBFInterpolant(object):
    """Compute and evaluate RBF interpolant.

    Manages an expansion of the form

    .. math::
        f(x) = \\sum_j c_j \\phi(\\|x-x_j\\|) + \\sum_j \\lambda_j p_j(x)

    where the functions :math:`p_j(x)` are low-degree polynomials.
    The fitting equations are

    .. math::

        \\begin{bmatrix} \\eta I & P^T \\\\ P & \\Phi+\\eta I \\end{bmatrix}
        \\begin{bmatrix} \\lambda \\\\ c \\end{bmatrix} =
        \\begin{bmatrix} 0 \\\\ f \\end{bmatrix}

    where :math:`P_{ij} = p_j(x_i)` and
    :math:`\\Phi_{ij}=\\phi(\\|x_i-x_j\\|)`.
    The regularization parameter :math:`\\eta` allows us to avoid problems
    with potential poor conditioning of the system.

    :ivar phi: Kernel function
    :ivar P: Tail functions
    :ivar dphi: Derivative of kernel function
    :ivar dP: Gradient of tail functions
    :ivar eta: Regularization parameter
    :ivar ntail: Number of tail functions
    :ivar nump: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar A: Interpolation system matrix
    :ivar rhs: Right hand side for interpolation system
    :ivar x: Interpolation points
    :ivar fx: Values at interpolation points
    :ivar c: Expansion coefficients
    :ivar dim: Number of dimensions
    :ivar ntail: Number of tail functions
    """

    def __init__(self, phi, P, dphi=None, dP=None, eta=1e-8, maxp=100):
        self.phi = phi
        self.dphi = dphi
        self.P = P        # pylint: disable=invalid-name
        self.dP = dP      # pylint: disable=invalid-name
        self.eta = eta
        self.nump = 0
        self.maxp = maxp
        self.x = None     # pylint: disable=invalid-name
        self.fx = None
        self.rhs = None
        self.A = None     # pylint: disable=invalid-name
        self.c = None     # pylint: disable=invalid-name
        self.dim = None
        self.ntail = None

    def reset(self):
        """Re-set the interpolation."""
        self.nump = 0
        self.x = None
        self.fx = None
        self.rhs = None
        self.A = None
        self.c = None

    def _alloc(self, dim, ntail):
        """Allocate storage for x, fx, rhs, and A.

        :param dim: Number of dimensions
        :param ntail: Number of tail functions
        """
        maxp = self.maxp
        self.dim = dim
        self.ntail = ntail
        self.x = np.zeros((maxp, dim))
        self.fx = np.zeros((maxp, 1))
        self.rhs = np.zeros((maxp+ntail, 1))
        self.A = np.zeros((maxp+ntail, maxp+ntail))
        for i in range(ntail):
            self.A[i, i] = self.eta

    def _realloc(self, dim, ntail, extra=1):
        """Expand allocation to accommodate more points (if needed)

        :param dim: Number of dimensions
        :param ntail: Number of tail functions
        :param extra: Number of additional points to accommodate
        """
        if self.nump == 0:
            self._alloc(dim, ntail)
        elif self.nump+extra > self.maxp:
            self.maxp = max(self.maxp*2, self.maxp+extra)
            self.x.resize((self.maxp, dim))
            self.fx.resize((self.maxp, 1))
            self.rhs.resize((self.maxp+ntail, 1))
            A0 = self.A  # pylint: disable=invalid-name
            self.A = np.zeros((self.maxp+ntail, self.maxp+ntail))
            self.A[:A0.shape[0], :A0.shape[1]] = A0

    def coeffs(self):
        """Compute the expansion coefficients

        :return: Expansion coefficients
        """
        if self.c is None:
            nact = self.ntail + self.nump
            self.c = la.solve(self.A[:nact, :nact], self.rhs[:nact])
        return self.c

    def get_x(self):
        """Get the list of data points

        :return: List of data points
        """
        return self.x[:self.nump, :]

    def get_fx(self):
        """Get the list of function values for the data points.

        :return: List of function values
        """
        return self.fx[:self.nump, :]

    def add_point(self, xx, fx):
        """Add a new function evaluation

        :param xx: Point to add
        :param fx: The function value of the point to add
        """
        px = self.P(xx)
        dim = len(xx)
        ntail = len(px)
        nact = ntail + self.nump
        self._realloc(dim, ntail)
        self.x[self.nump, :] = xx
        self.fx[self.nump, :] = fx
        self.rhs[nact] = fx
        A = self.A  # pylint: disable=invalid-name
        A[nact, :ntail] = px
        A[:ntail, nact] = px
        A[nact, nact] = self.phi(0) + self.eta
        for i in range(self.nump):
            rij = la.norm(xx-self.x[i, :])
            aij = self.phi(rij)
            A[ntail+i, nact] = aij
            A[nact, ntail+i] = aij
        self.nump += 1
        self.c = None

    def transform_fx(self, fx):
        """Replace f with transformed function values for the fitting

        :param fx: Transformed function value
        """
        self.rhs[self.ntail:self.ntail+self.nump] = fx
        self.c = None

    def eval(self, xx):
        """Evaluate the rbf interpolant at the point xx

        :param xx: Point where to evaluate
        :return: Value of the rbf interpolant at x
        """
        px = self.P(xx)
        ntail = self.ntail
        c = self.coeffs()
        fx = 0
        for i in range(ntail):
            fx = fx + px[i]*c[i, 0]
        for i in range(self.nump):
            ri = la.norm(xx-self.x[i, :])
            fx = fx + self.phi(ri)*c[i+ntail, 0]
        return fx

    def evals(self, xx):
        """Evaluate the rbf interpolant at the points xx

        :param xx: Points where to evaluate
        :return: Values of the rbf interpolant at x
        """
        ntail = self.ntail
        c = np.asmatrix(self.coeffs())
        ds = scp.distance.cdist(xx, self.x[:self.nump, :])
        fx = self.phi(ds)*c[ntail:ntail+self.nump] + self.P(xx)*c[:ntail]
        return fx

    def deriv(self, x):
        """Evaluate the derivative of the rbf interpolant at x

        :param x: Data point
        :return: Derivative of the rbf interpolant at x
        """
        dpx = self.dP(x)
        ntail = self.ntail
        c = self.coeffs()
        dfx = np.zeros(self.dim)
        for i in range(ntail):
            dfx += dpx[i, :]*c[i, 0]
        for i in range(self.nump):
            ri = la.norm(x-self.x[i, :])
            if ri == 0:
                dri = x/la.norm(x)
            else:
                dri = (x-self.x[i, :])/ri
            dfx += self.dphi(ri)*dri*c[i+ntail, 0]
        return dfx


def phi_linear(r):
    """Linear RBF interpolant

    :param r: Data point
    :return: Value of the linear rbf interpolant
    """
    return r


def phi_cubic(r):
    """Cubic RBF interpolant

    :param r: Data point
    :return: Value of the cubic rbf interpolant
    """
    return r*r*r


def phi_plate(r):
    """Thin plate RBF interpolant

    :param r: Data point
    :return: Value of the thin plate rbf interpolant
    """
    eps = np.finfo(np.double).tiny
    return r*r * np.log(r+eps)


def dphi_linear(r):
    """Derivative of linear RBF interpolant

    :param r: Data point
    :return: Derivative of the linear rbf interpolant
    """
    return r*0+1


def dphi_cubic(r):
    """Derivative of cubic RBF interpolant

    :param r: Data point
    :return: Derivative of the cubic rbf interpolant
    """
    return 3*r*r


def dphi_plate(r):
    """Derivative of thin plate RBF interpolant

    :param r: Data point
    :return: Derivative of the thin plate rbf interpolant
    """
    eps = np.finfo(np.double).tiny
    return (1+2*np.log(r+eps))*r


def const_tail(x):
    """Constant polynomial tail

    :param x: Data point
    :return: Value the constant tail
    """
    if len(x.shape) == 1:
        px = np.ones(1)
    else:
        px = np.ones((x.shape[0], 1))
    return px


def linear_tail(x):
    """Linear polynomial tail

    :param x: Data point
    :return: Value the linear tail
    """
    if len(x.shape) == 1:
        px = np.zeros(x.shape[0]+1)
        px[0] = 1
        px[1:] = x
    else:
        px = np.ones((x.shape[0], x.shape[1]+1))
        px[:, 1:] = x
    return px


def dconst_tail(x):
    """Derivative of constant polynomial tail

    :param x: Data point
    :return: Derivative of constant tail
    """
    return np.zeros((x.shape[0], 1))


def dlinear_tail(x):
    """Derivative of linear polynomial tail

    :param x: Data point
    :return: Derivative of linear tail
    """
    dpx = np.zeros((x.shape[0]+1, x.shape[0]))
    dpx[1:, :] = np.eye(x.shape[0])
    return dpx

# ====================================================================


def _main():
    """Main test routine"""

    def test_f(x):
        """Test function"""
        fx = x[1]*np.sin(x[0]) + x[0]*np.cos(x[1])
        return fx

    def test_df(x):
        """Derivative of test function"""
        dfx = np.array([x[1]*np.cos(x[0])+np.cos(x[1]),
                        np.sin(x[0])-x[0]*np.sin(x[1])])
        return dfx

    fhat = RBFInterpolant(phi_cubic, linear_tail,
                          dphi_cubic, dlinear_tail, 1e-8, 20)
    xs = np.random.rand(120, 2)
    for i in range(100):
        xx = xs[i, :]
        fx = test_f(xx)
        fhat.add_point(xx, fx)
    fhx = fhat.evals(xs[:5, :])
    print fhx.shape
    for i in range(5):
        fx = test_f(xs[i, :])
        print("Err: %e" % (abs(fx-fhx[i])/abs(fx)))
    for i in range(10):
        xx = xs[100+i, :]
        fx = test_f(xx)
        dfx = test_df(xx)
        fhx = fhat.eval(xx)
        dfhx = fhat.deriv(xx)
        print("Err (interp): %e : %e" % (abs(fx-fhx)/abs(fx),
                                         la.norm(dfx-dfhx)/la.norm(dfx)))

if __name__ == "__main__":
    _main()
