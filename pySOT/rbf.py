"""
.. module:: rbf
   :synopsis: Radial basis function interpolation and regression

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,
                David Bindel <bindel@cornell.edu>

:Module: rbf
:Author: David Eriksson <dme65@cornell.edu>,
        David Bindel <bindel@cornell.edu>

"""

import numpy as np
import scipy.spatial as scpspatial
import scipy.linalg as scplinalg
from pySOT.kernels import *
from pySOT.tails import *


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

    where :math:`P_{ij} = p_j(x_i)` and :math:`\\Phi_{ij}=\\phi(\\|x_i-x_j\\|)`.
    The regularization parameter :math:`\\eta` allows us to avoid problems
    with potential poor conditioning of the system. The regularization parameter
    can either be fixed or estimated via LOOCV. Specify eta='adapt' for estimation.

    :param kernel: RBF kernel object
    :type kernel: Kernel
    :param tail: RBF polynomial tail object
    :type tail: Tail
    :param maxp: Initial point capacity
    :type maxp: int
    :param eta: Regularization parameter
    :type eta: float or 'adapt'

    :ivar kernel: RBF kernel
    :ivar tail: RBF tail
    :ivar eta: Regularization parameter
    :ivar ntail: Number of tail functions
    :ivar nump: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar A: Interpolation system matrix
    :ivar LU: LU-factorization of the RBF system
    :ivar piv: pivot vector for the LU-factorization
    :ivar rhs: Right hand side for interpolation system
    :ivar x: Interpolation points
    :ivar fx: Values at interpolation points
    :ivar c: Expansion coefficients
    :ivar dim: Number of dimensions
    :ivar ntail: Number of tail functions
    :ivar updated: True if the RBF coefficients are up to date
    """

    def __init__(self, kernel=CubicKernel, tail=LinearTail, maxp=500, eta=1e-8):

        if kernel is None or tail is None:
            kernel = CubicKernel
            tail = LinearTail

        self.maxp = maxp
        self.nump = 0
        self.kernel = kernel()
        self.tail = tail()
        self.ntail = None
        self.A = None
        self.LU = None
        self.piv = None
        self.c = None
        self.dim = None
        self.x = None
        self.fx = None
        self.rhs = None
        self.c = None
        self.eta = eta
        self.updated = False

        if eta is not 'adapt' and (eta < 0 or eta >= 1):
            raise ValueError("eta has to be in [0,1) or be the string 'adapt' ")

        if self.kernel.order() - 1 > self.tail.degree():
            raise ValueError("Kernel and tail mismatch")

    def reset(self):
        """Reset the RBF interpolant"""
        self.nump = 0
        self.x = None
        self.fx = None
        self.rhs = None
        self.A = None
        self.LU = None
        self.piv = None
        self.c = None
        self.updated = False

    def _alloc(self, dim, ntail):
        """Allocate storage for x, fx, rhs, and A.

        :param dim: Number of dimensions
        :type dim: int
        :param ntail: Number of tail functions
        :type ntail: int
        """

        maxp = self.maxp
        self.dim = dim
        self.ntail = ntail
        self.x = np.zeros((maxp, dim))
        self.fx = np.zeros((maxp, 1))
        self.rhs = np.zeros((maxp+ntail, 1))
        self.A = np.zeros((maxp+ntail, maxp+ntail))

    def _realloc(self, dim, extra=1):
        """Expand allocation to accommodate more points (if needed)

        :param dim: Number of dimensions
        :type dim: int
        :param extra: Number of additional points to accommodate
        :type extra: int
        """

        self.dim = dim
        self.ntail = self.tail.dim_tail(dim)
        if self.nump == 0:
            self._alloc(dim, self.ntail)
        elif self.nump + extra > self.maxp:
            self.maxp = max(self.maxp*2, self.maxp+extra)
            self.x.resize((self.maxp, dim))
            self.fx.resize((self.maxp, 1))
            self.rhs.resize((self.maxp + self.ntail, 1))
            A0 = self.A  # pylint: disable=invalid-name
            self.A = np.zeros((self.maxp + self.ntail, self.maxp + self.ntail))
            self.A[:A0.shape[0], :A0.shape[1]] = A0

    def coeffs(self):
        """Compute the expansion coefficients

        :return: Expansion coefficients
        :rtype: numpy.array
        """

        if self.c is None:
            nact = self.ntail + self.nump

            if self.eta is 'adapt':
                eta_vec = np.linspace(0, 0.99, 30)
            else:
                eta_vec = np.array([self.eta])

            rms_best = np.inf

            for i in range(len(eta_vec)):
                eta = eta_vec[i]

                Aa = np.copy(self.A[:nact, :nact])
                for j in range(self.nump):
                    Aa[j + self.ntail, j + self.ntail] += eta/(1-eta)*self.nump

                [LU, piv] = scplinalg.lu_factor(Aa)
                c = scplinalg.lu_solve((LU, piv), self.rhs[:nact])

                # Do LOOCV if requested
                if self.eta is 'adapt':
                    I = np.eye(nact)
                    AinvI = scplinalg.lu_solve((LU, piv), I[:, self.ntail:])

                    chat = c - np.multiply(AinvI, np.transpose(
                        c[self.ntail:]/np.transpose(np.atleast_2d(np.diag(AinvI[self.ntail:, :])))))

                    for j in range(self.nump):
                        chat[j + self.ntail, j] = 0

                    f_pred = np.sum(np.transpose(self.A[self.ntail:nact, self.ntail:nact]) * chat[self.ntail:, :], axis=0) + \
                        np.sum(np.transpose(self.A[self.ntail:nact, :self.ntail]) * chat[:self.ntail, :], axis=0)
                    rms_val = np.sqrt(np.sum((self.fx[:self.nump] - np.transpose(np.atleast_2d(f_pred))) ** 2)/self.nump)

                    if rms_val < rms_best:
                        rms_best = rms_val
                        self.eta_best = eta
                        self.c = np.copy(c)
                        self.piv = piv
                        self.LU = LU
                else:
                    self.c = c
                    self.piv = piv
                    self.LU = LU
                    return self.c

        return self.c

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
        self.fx[self.nump] = fx
        self.rhs[self.ntail + self.nump] = fx

        self.nump += 1
        nact = self.nump + self.ntail

        p = self.tail.eval(xx)
        phi = self.kernel.eval(scpspatial.distance.cdist(self.get_x(), np.atleast_2d(xx)))

        #  Create the matrix with the initial points
        self.A[nact-1, 0:self.ntail] = p.ravel()
        self.A[0:self.ntail, nact-1] = p.ravel()
        self.A[nact-1, self.ntail:nact] = phi.ravel()
        self.A[self.ntail:nact, nact-1] = phi.ravel()

        # Coefficients and LU are outdated
        self.LU = None
        self.piv = None
        self.c = None

        self.updated = False

    def transform_fx(self, fx):
        """Replace f with transformed function values for the fitting

        :param fx: Transformed function values
        :type fx: numpy.array
        """
        self.rhs[self.ntail:self.ntail+self.nump] = fx
        self.LU = None
        self.piv = None
        self.c = None

    def eval(self, x, ds=None):
        """Evaluate the RBF interpolant at the point x

        :param x: Point where to evaluate
        :type x: numpy.array
        :return: Value of the RBF interpolant at x
        :rtype: float
        """

        px = self.tail.eval(x)
        ntail = self.ntail
        c = self.coeffs()
        if ds is None:
            ds = scpspatial.distance.cdist(np.atleast_2d(x), self.x[:self.nump, :])
        fx = np.dot(px, c[:ntail]) + np.dot(self.kernel.eval(ds), c[ntail:ntail+self.nump])
        return fx[0][0]

    def evals(self, x, ds=None):
        """Evaluate the RBF interpolant at the points x

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Distances between the centers and the points x, of size npts x ncenters
        :type ds: numpy.array
        :return: Values of the rbf interpolant at x, of length npts
        :rtype: numpy.array
        """

        ntail = self.ntail
        c = np.asmatrix(self.coeffs())
        if ds is None:
            ds = scpspatial.distance.cdist(x, self.x[:self.nump, :])
        fx = self.kernel.eval(ds)*c[ntail:ntail+self.nump] + self.tail.eval(x)*c[:ntail]
        return fx

    def deriv(self, x, ds=None):
        """Evaluate the derivative of the RBF interpolant at a point x

        :param x: Point for which we want to compute the RBF gradient
        :type x: numpy.array
        :param ds: Distances between the centers and the point x
        :type ds: numpy.array
        :return: Derivative of the RBF interpolant at x
        :rtype: numpy.array
        """

        if len(x.shape) == 1:
            x = np.atleast_2d(x)  # Make x 1-by-dim
        ntail = self.ntail
        dpx = self.tail.deriv(x.transpose())
        c = self.coeffs()
        dfx = np.dot(dpx, c[:ntail]).transpose()
        if ds is None:
            ds = scpspatial.distance.cdist(self.x[:self.nump, :], np.atleast_2d(x))
        ds[ds < 1e-10] = 1e-10  # Better safe than sorry
        dsx = - self.x[:self.nump, :]
        dsx += x
        dsx *= (np.multiply(self.kernel.deriv(ds), c[ntail:]) / ds)
        dfx += np.sum(dsx, 0)

        return dfx
