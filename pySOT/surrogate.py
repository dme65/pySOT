"""
.. module:: surrogate
   :synopsis: Surrogate surfaces

.. moduleauthor:: David Eriksson <dme65@cornell.edu>

:Module: surrogate
:Author: David Eriksson <dme65@cornell.edu>

"""

# from pyds import MassFunction
# from copy import copy, deepcopy
# import math
# import numpy.linalg as la
import numpy as np
import scipy.spatial as scpspatial
import scipy.linalg as scplinalg
import abc
import six
from pySOT.utils import reallocate


@six.add_metaclass(abc.ABCMeta)
class Surrogate(object):
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def dim(self):  # pragma: no cover
        """Get dimensionality"""
        pass

    @property
    @abc.abstractmethod
    def npts(self):  # pragma: no cover
        """Get number of points"""
        pass

    @property
    @abc.abstractmethod
    def X(self):  # pragma: no cover
        """Return centers fx of the interpolant"""
        return

    @property
    @abc.abstractmethod
    def fX(self):  # pragma: no cover
        """Return values fx of the interpolant"""
        return

    @abc.abstractmethod
    def reset(self):  # pragma: no cover
        """Reset the object"""
        pass

    @abc.abstractmethod
    def add_points(self, X, fX):  # pragma: no cover
        """Add points xx with values fxx

        xx must be of size npts x dim or (dim, )
        fxx must be of size npts x 1 or (npts, )
        """
        return

    @abc.abstractmethod
    def eval(self, X):  # pragma: no cover
        """Evaluate interpolant at points xx

        xx must be of size npts x dim or (dim, )
        """
        return

    @abc.abstractmethod
    def deriv(self, X):  # pragma: no cover
        """Evaluate derivative of interpolant at points xx

        xx must be of size npts x dim or (dim, )
        """
        return


@six.add_metaclass(abc.ABCMeta)
class Kernel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):  # pragma: no cover
        pass

    @property
    @abc.abstractmethod
    def order(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def eval(self, dists):  # pragma: no cover
        pass

    @abc.abstractmethod
    def deriv(self, dists):  # pragma: no cover
        pass


@six.add_metaclass(abc.ABCMeta)
class Tail(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, dim):  # pragma: no cover
        pass

    @property
    @abc.abstractmethod
    def degree(self):  # pragma: no cover
        pass

    @property
    @abc.abstractmethod
    def dim_tail(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def eval(self, X):  # pragma: no cover
        pass

    @abc.abstractmethod
    def deriv(self, x):  # pragma: no cover
        pass


class CubicKernel(Kernel):
    """Cubic RBF kernel

    This is a basic class for the Cubic RBF kernel: :math:`\\varphi(r) = r^3` which is
    conditionally positive definite of order 2.
    """

    def __init__(self):
        super(CubicKernel, self).__init__()

    @property
    def order(self):
        """returns the order of the Cubic RBF kernel

        :returns: 2
        :rtype: int
        """

        return 2

    def eval(self, dists):
        """evaluates the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\|x_i - x_j \|^3`
        :rtype: numpy.array
        """

        return dists ** 3

    def deriv(self, dists):
        """evaluates the derivative of the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`3 \| x_i - x_j \|^2`
        :rtype: numpy.array
        """

        return 3 * dists ** 2


class TPSKernel(Kernel):
    """Thin-plate spline RBF kernel

    This is a basic class for the TPS RBF kernel: :math:`\\varphi(r) = r^2 \log(r)` which is
    conditionally positive definite of order 2.
    """

    @property
    def order(self):
        """returns the order of the TPS RBF kernel

        :returns: 2
        :rtype: int
        """

        return 2

    def eval(self, dists):
        """evaluates the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\|x_i - x_j \|^2 \log (\|x_i - x_j \|)`
        :rtype: numpy.array
        """

        dists[dists < np.finfo(float).tiny] = np.finfo(float).tiny
        return (dists ** 2) * np.log(dists)

    def deriv(self, dists):
        """evaluates the derivative of the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\|x_i - x_j \|(1 + 2 \log (\|x_i - x_j \|) )`
        :rtype: numpy.array
        """

        dists[dists < np.finfo(float).tiny] = np.finfo(float).tiny
        return dists * (1 + 2 * np.log(dists))


class LinearKernel(Kernel):
    """Linear RBF kernel

     This is a basic class for the Linear RBF kernel: :math:`\\varphi(r) = r` which is
     conditionally positive definite of order 1.
     """

    @property
    def order(self):
        """returns the order of the Linear RBF kernel

        :returns: 1
        :rtype: int
        """

        return 1

    def eval(self, dists):
        """evaluates the Linear kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\|x_i - x_j \|`
        :rtype: numpy.array
        """

        return dists

    def deriv(self, dists):
        """evaluates the derivative of the Linear kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is 1
        :rtype: numpy.array
        """

        return np.ones(dists.shape)


class LinearTail(Tail):
    """Linear polynomial tail

    This is a standard linear polynomial in d-dimension, built from the basis
    :math:`\{1,x_1,x_2,\ldots,x_d\}`.
    """

    def __init__(self, dim):
        super(LinearTail, self).__init__(dim)
        self.dim = dim

    @property
    def degree(self):
        """returns the degree of the linear polynomial tail

        :returns: 1
        :rtype: int
        """

        return 1

    @property
    def dim_tail(self):
        """returns the dimensionality of the linear polynomial space for a given dimension

        :param dim: Number of dimensions of the Cartesian space
        :type dim: int
        :returns: 1 + dim
        :rtype: int
        """

        return 1 + self.dim

    def eval(self, X):
        """evaluates the linear polynomial tail for a set of points

        :param X: Points to evaluate, of size npts x dim
        :type X: numpy.array
        :returns: A numpy.array of size npts x dim_tail(dim)
        :rtype: numpy.array
        """

        X = np.atleast_2d(X)
        if X.shape[1] != self.dim:
            raise ValueError("Input has the wrong number of dimensions")
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def deriv(self, x):
        """evaluates the gradient of the linear polynomial tail for one point

        :param x: Point to evaluate, of length dim
        :type x: numpy.array
        :returns: A numpy.array of size dim x dim_tail(dim)
        :rtype: numpy.array
        """

        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError("Input has the wrong number of dimensions")
        return np.hstack((np.zeros((x.shape[1], 1)), np.eye((x.shape[1]))))


class ConstantTail(Tail):
    """Constant polynomial tail

    This is a standard linear polynomial in d-dimension, built from the basis
    :math:`\{1\}`.
    """

    def __init__(self, dim):
        super(ConstantTail, self).__init__(dim)
        self.dim = dim

    @property
    def degree(self):
        """returns the degree of the constant polynomial tail

        :returns: 0
        :rtype: int
        """

        return 0

    @property
    def dim_tail(self):
        """returns the dimensionality of the constant polynomial space for a given dimension

        :param dim: Number of dimensions of the Cartesian space
        :type dim: int
        :returns: 1
        :rtype: int
        """

        return 1

    def eval(self, X):
        """evaluates the constant polynomial tail for a set of points

        :param X: Points to evaluate, of size npts x dim
        :type X: numpy.array
        :returns: A numpy.array of size npts x dim_tail(dim)
        :rtype: numpy.array
        """

        X = np.atleast_2d(X)
        if X.shape[1] != self.dim:
            raise ValueError("Input has the wrong number of dimensions")
        return np.ones((X.shape[0], 1))

    def deriv(self, x):
        """evaluates the gradient of the linear polynomial tail for one point

        :param x: Point to evaluate, of length dim
        :type x: numpy.array
        :returns: A numpy.array of size dim x dim_tail(dim)
        :rtype: numpy.array
        """

        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError("Input has the wrong number of dimensions")
        return np.zeros((x.shape[1], 1))


class RBFInterpolant(Surrogate):
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
    :ivar npts: Current number of points
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

    def __init__(self, dim, maxpts=500, kernel=None, tail=None, eta=1e-6):

        if kernel is None or tail is None:
            kernel = CubicKernel()
            tail = LinearTail(dim)

        assert(isinstance(kernel, Kernel))
        assert(isinstance(tail, Tail))

        self._dim = dim
        self._npts = 0
        self._maxpts = maxpts
        self._X = None
        self._fX = None

        self.kernel = kernel
        self.tail = tail
        self.ntail = tail.dim_tail
        self.A = None
        self.L = None
        self.U = None
        self.piv = None
        self.c = None
        self.rhs = None
        self.eta = eta
        self.dirty = True

        if self.kernel.order - 1 > self.tail.degree:
            raise ValueError("Kernel and tail mismatch")
        assert self.dim == self.tail.dim

    @property
    def dim(self):
        return self._dim

    @property
    def npts(self):
        return self._npts

    @property
    def maxpts(self):
        return self._maxpts

    @property
    def X(self):
        """Get the list of data points

        :return: List of data points
        :rtype: numpy.array
        """

        return self._X[:self.npts, :]

    @property
    def fX(self):
        """Get the list of function values for the data points.

        :return: List of function values
        :rtype: numpy.array
        """

        return self._fX[:self.npts]

    def reset(self):
        """Reset the RBF interpolant"""
        self._npts = 0
        self._X = None
        self._fX = None
        self.rhs = None
        self.L = None
        self.U = None
        self.piv = None
        self.c = None
        self.dirty = True

    def _realloc(self, extra=1):
        """Expand allocation to accommodate more points (if needed)

        :param extra: Number of additional points to accommodate
        :type extra: int
        """

        maxp = self.maxpts
        ntail = self.ntail
        if maxp < self.npts + extra or self.npts == 0:
            while maxp < self.npts + extra: maxp = 2 * maxp
            self._maxpts = maxp
            self._X = reallocate(self._X, (maxp, self.dim))
            self._fX = reallocate(self._fX, (maxp,))
            self.rhs = reallocate(self.rhs, (maxp + ntail,))
            self.piv = reallocate(self.piv, (maxp + ntail,), dtype=np.int)
            self.L = reallocate(self.L, (maxp + ntail, maxp + ntail))
            self.U = reallocate(self.U, (maxp + ntail, maxp + ntail))

    def coeffs(self):
        """Compute the expansion coefficients

        :return: Expansion coefficients
        :rtype: numpy.array
        """

        if self.dirty:

            n = self.npts
            ntail = self.ntail
            nact = ntail + n

            if self.c is None:  # Initial fit
                assert self.npts >= ntail

                X = self._X[0:n, :]
                D = scpspatial.distance.cdist(X, X)
                Phi = self.kernel.eval(D) + self.eta * np.eye(n)
                P = self.tail.eval(X)

                # Set up the systems matrix
                A1 = np.hstack((np.zeros((ntail, ntail)), P.T))
                A2 = np.hstack((P, Phi))
                A = np.vstack((A1, A2))

                [LU, piv] = scplinalg.lu_factor(A)
                self.L[:nact, :nact] = np.tril(LU, -1) + np.eye(nact)
                self.U[:nact, :nact] = np.triu(LU)

                # Construct the usual pivoting vector so that we can increment later
                self.piv[:nact] = np.arange(0, nact)
                for i in range(nact):
                    self.piv[i], self.piv[piv[i]] = self.piv[piv[i]], self.piv[i]

            else:  # Extend LU factorization
                k = self.c.shape[0] - ntail
                numnew = n - k
                kact = ntail + k
                self.piv[kact:nact] = np.arange(kact, nact)

                X = self._X[:n, :]
                XX = self._X[k:n, :]
                D = scpspatial.distance.cdist(X, XX)
                Pnew = np.vstack((self.tail.eval(XX).T, self.kernel.eval(D[:k, :])))
                Phinew = self.kernel.eval(D[k:, :]) + self.eta * np.eye(numnew)

                L21 = np.zeros((kact, numnew))
                U12 = np.zeros((kact, numnew))
                for i in range(numnew):  # Todo: Too bad we can't use level-3 BLAS here
                    L21[:, i] = scplinalg.solve_triangular(a=self.U[:kact, :kact], b=Pnew[:kact, i],
                                                           lower=False, trans='T')
                    U12[:, i] = scplinalg.solve_triangular(a=self.L[:kact, :kact], b=Pnew[self.piv[:kact], i],
                                                           lower=True, trans='N')
                L21 = L21.T
                try:
                    C = scplinalg.cholesky(a=Phinew - np.dot(L21, U12), lower=True)
                except Exception as e:  # Compute a new LU factorization if the Cholesky fails
                    self.c = None
                    return self.coeffs()

                self.L[kact:nact, :kact] = L21
                self.U[:kact, kact:nact] = U12
                self.L[kact:nact, kact:nact] = C
                self.U[kact:nact, kact:nact] = C.T

            # Update coefficients
            self.c = scplinalg.solve_triangular(a=self.L[:nact, :nact], b=self.rhs[self.piv[:nact]], lower=True)
            self.c = scplinalg.solve_triangular(a=self.U[:nact, :nact], b=self.c, lower=False)
            self.c = np.asmatrix(self.c).T
            self.dirty = False

        return self.c

    def add_points(self, xx, fx):
        """Add a new function evaluation

        :param xx: Points to add
        :type xx: numpy.array
        :param fx: The function value of the point to add
        :type fx: float
        """

        xx = np.atleast_2d(xx)
        newpts = xx.shape[0]
        self._realloc(extra=newpts)

        self._X[self.npts:self.npts + newpts, :] = xx
        self._fX[self.npts:self.npts + newpts] = fx
        self.rhs[self.ntail + self.npts:self.ntail + self.npts+newpts] = fx
        self._npts += newpts

        self.dirty = True

    def eval(self, x, ds=None):
        """Evaluate the RBF interpolant at the points x

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Distances between the centers and the points x, of size npts x ncenters
        :type ds: numpy.array
        :return: Values of the rbf interpolant at x, of length npts
        :rtype: numpy.array
        """

        x = np.atleast_2d(x)
        ntail = self.ntail
        c = self.coeffs()
        if ds is None:
            ds = scpspatial.distance.cdist(x, self._X[:self.npts, :])
        fx = self.kernel.eval(ds)*c[ntail:ntail + self.npts] + self.tail.eval(x)*c[:ntail]
        return fx

    def deriv(self, xx, ds=None):
        """Evaluate the derivative of the RBF interpolant at a point x

        :param xx: Point for which we want to compute the RBF gradient
        :type xx: numpy.array
        :param ds: Distances between the centers and the point x
        :type ds: numpy.array
        :return: Derivative of the RBF interpolant at x
        :rtype: numpy.array
        """

        xx = np.atleast_2d(xx)
        if xx.shape[1] != self.dim:
            raise ValueError("Input has incorrect number of dimensions")

        if ds is None:
            ds = scpspatial.distance.cdist(self.X, xx)
        elif not (ds.shape[0] == self.npts and ds.shape[1] == xx.shape[0]):
            raise ValueError("ds has incorrect size")
        ds[ds < np.finfo(float).tiny] = np.finfo(float).tiny  # Better safe than sorry

        dfxx = np.zeros((xx.shape[0], self.dim))
        for i in range(xx.shape[0]):
            x = np.atleast_2d(xx[i, :])
            ntail = self.ntail
            dpx = self.tail.deriv(x)
            c = self.coeffs()
            dfx = np.dot(dpx, c[:ntail]).transpose()
            dsx = - self.X
            dsx += x
            dss = np.atleast_2d(ds[:, i]).T
            dsx *= (np.multiply(self.kernel.deriv(dss), c[ntail:]) / dss)
            dfx += np.sum(dsx, 0)
            dfxx[i, :] = dfx

        return dfxx


class GPRegression(Surrogate):
    """Compute and evaluate a GP

    Gaussian Process Regression object.

    Depends on scitkit-learn==0.18.1.

    More details:
        http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
    """

    def __init__(self, dim, maxpts=100, gp=None):

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError as err:
            print("Failed to import sklearn.gaussian_process and sklearn.gaussian_process.kernels")
            raise err

        self._npts = 0
        self._maxpts = maxpts
        self._dim = dim
        self._X = None     # pylint: disable=invalid-name
        self._fX = None
        if gp is None:
            self.model = GaussianProcessRegressor(n_restarts_optimizer=10)
        else:
            self.model = gp
            if not isinstance(gp, GaussianProcessRegressor):
                raise TypeError("gp is not of type GaussianProcessRegressor")
        self.updated = False

    @property
    def dim(self):
        return self._dim

    @property
    def npts(self):
        return self._npts

    @property
    def maxpts(self):
        return self._maxpts

    @property
    def X(self):
        """Get the list of data points

        :return: List of data points
        :rtype: numpy.array
        """

        return self._X[:self.npts, :]

    @property
    def fX(self):
        """Get the list of function values for the data points.

        :return: List of function values
        :rtype: numpy.array
        """

        return self._fX[:self.npts]

    def reset(self):
        """Reset the interpolation."""

        self._npts = 0
        self._X = None
        self._fX = None
        self.updated = False

    def _realloc(self, extra=1):
        """Expand allocation to accommodate more points (if needed)

        :param extra: Number of additional points to accommodate
        :type extra: int
        """

        maxp = self.maxpts
        if maxp < self.npts + extra or self.npts == 0:
            while maxp < self.npts + extra: maxp = 2 * maxp
            self._maxpts = maxp
            self._X = reallocate(self._X, (maxp, self.dim))
            self._fX = reallocate(self._fX, (maxp,))

    def add_points(self, xx, fx):
        """Add new function evaluations

        :param xx: Points to add
        :type xx: numpy.array
        :param fx: The function value of the point to add
        :type fx: float
        """

        xx = np.atleast_2d(xx)
        newpts = xx.shape[0]
        self._realloc(extra=newpts)

        self._X[self.npts:self.npts + newpts, :] = xx
        self._fX[self.npts:self.npts + newpts] = fx
        self._npts += newpts

        self.updated = False

    def eval(self, x, ds=None):
        """Evaluate the GP regression object at the points x

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the GP regression object at x, of length npts
        :rtype: numpy.array
        """

        if self.updated is False:
            self.model.fit(self.X, self.fX)
        self.updated = True

        fx = np.zeros(shape=(x.shape[0], 1))
        fx[:, 0] = self.model.predict(x)
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


# class PolyRegression(Surrogate):
#     """Compute and evaluate a polynomial regression surface.
#
#     :param bounds: a (dims, 2) array of lower and upper bounds in each coordinate
#     :type bounds: numpy.array
#     :param basisp: a (nbasis, dims) array, where the ith basis function is
#         prod_j L_basisp(i,j)(x_j), L_k = the degree k Legendre polynomial
#     :type basisp: numpy.array
#     :param maxp: Initial point capacity
#     :type maxp: int
#
#     :ivar npts: Current number of points
#     :ivar maxp: Initial maximum number of points (can grow)
#     :ivar x: Interpolation points
#     :ivar fx: Function evaluations of interpolation points
#     :ivar bounds: Upper and lower bounds, one row per dimension
#     :ivar dim: Number of dimensions
#     :ivar basisp: Multi-indices representing terms in a tensor poly basis
#         Each row is a list of dim indices indicating a polynomial degree
#         in the associated dimension.
#     :ivar updated: True if the RBF coefficients are up to date
#     """
#
#     def __init__(self, bounds, basisp, maxp=100):
#         self.npts = 0
#         self.maxp = maxp
#         self.x = None     # pylint: disable=invalid-name
#         self.fx = None
#         self.bounds = bounds
#         self.dim = self.bounds.shape[0]
#         self.basisp = basisp
#         self.updated = False
#
#     def reset(self):
#         """Reset the object."""
#
#         self.npts = 0
#         self.x = None
#         self.fx = None
#         self.updated = False
#
#     def _normalize(self, x):
#         """Normalize points to the box [-1,1]^d"""
#
#         # Todo: This is broken
#         xx = np.copy(x)
#         for k in range(x.shape[0]):
#             l = self.bounds[k, 0]
#             u = self.bounds[k, 1]
#             w = u-l
#             xx[:, k] = (x[:, k]-l)/w + (x[:, k]-u)/w
#         return xx
#
#     def _alloc(self):
#         """Allocate storage for x and fx."""
#
#         maxp = self.maxp
#         self.x = np.zeros((maxp, self.dim))
#         self.fx = np.zeros((maxp, 1))
#
#     def _realloc(self, extra=1):
#         """Expand allocation to accommodate more points (if needed)
#
#         :param extra: Number of additional points to accommodate
#         :type extra: int
#         """
#
#         if self.npts == 0:
#             self._alloc()
#         elif self.npts + extra > self.maxp:
#             self.maxp = max(self.maxp*2, self.maxp + extra)
#             self.x.resize((self.maxp, self.dim))
#             self.fx.resize((self.maxp, 1))
#
#     def _plegendre(self, x):
#         """Evaluate basis functions.
#
#         :param x: Coordinates (one per row)
#         :type x: numpy.array
#         :return: Basis functions for each coordinate with shape (npts, nbasis)
#         :rtype: numpy.array
#         """
#
#         s = self.basisp
#         Px = legendre(x, np.max(s))
#         Ps = np.ones((x.shape[0], s.shape[0]))
#         for i in range(s.shape[0]):
#             for j in range(s.shape[1]):
#                 Ps[:, i] *= Px[:, j, s[i, j]]
#         return Ps
#
#     def _dplegendre(self, x):
#         """Evaluate basis function gradients.
#
#         :param x: Coordinates (one per row)
#         :type x: numpy.array
#         :return: Gradients for each coordinate with shape (npts, dim, nbasis)
#         :rtype: numpy.array
#         """
#
#         s = self.basisp
#         Px, dPx = dlegendre(x, np.max(s))
#         dPs = np.ones((x.shape[0], x.shape[1], s.shape[0]))
#         for i in range(s.shape[0]):
#             for j in range(s.shape[1]):
#                 for k in range(x.shape[1]):
#                     if k == j:
#                         dPs[:, k, i] *= dPx[:, j, s[i, j]]
#                     else:
#                         dPs[:, k, i] *= Px[:, j, s[i, j]]
#         return dPs
#
#     def _fit(self):
#         """Compute a least squares fit."""
#
#         A = self._plegendre(self._normalize(self.get_x()))
#         self.beta = la.lstsq(A, self.get_fx())[0]
#
#     def _predict(self, x):
#         """Evaluate on response surface."""
#
#         return np.dot(self._plegendre(self._normalize(x)), self.beta)
#
#     def _predict_deriv(self, xx):
#         """Predict derivative."""
#
#         dfx = np.dot(self._dplegendre(self._normalize(xx)), self.beta)
#         for j in range(xx.shape[1]):
#             dfx[:, j] /= (self.bounds[j, 1]-self.bounds[j, 0])/2
#         return dfx
#
#     def get_x(self):
#         """Get the list of data points
#
#         :return: List of data points
#         :rtype: numpy.array
#         """
#         return self.x[:self.npts, :]
#
#     def get_fx(self):
#         """Get the list of function values for the data points.
#
#         :return: List of function values
#         :rtype: numpy.array
#         """
#         return self.fx[:self.npts, :]
#
#     def add_points(self, xx, fx):
#         """Add a new function evaluation
#
#         :param xx: Point to add
#         :param fx: The function value of the point to add
#         """
#         self._realloc()
#         self.x[self.npts, :] = xx
#         self.fx[self.npts, :] = fx
#         self.npts += 1
#         self.updated = False
#
#     def eval(self, x, ds=None):
#         """Evaluate the regression surface at points x
#
#         :param x: Points where to evaluate, of size npts x dim
#         :type x: numpy.array
#         :param ds: Not used
#         :type ds: None
#         :return: Prediction at the points x
#         :rtype: float
#         """
#
#         if self.updated is False:
#             self._fit()
#         self.updated = True
#
#         return np.atleast_2d(self._predict(x))
#
#     def deriv(self, x, ds=None):
#         """Evaluate the derivative of the regression surface at a point x
#
#         :param x: Point where to evaluate
#         :type x: numpy.array
#         :param ds: Not used
#         :type ds: None
#         :return: Derivative of the polynomial at x
#         :rtype: numpy.array
#         """
#
#         if self.updated is False:
#             self._fit()
#         self.updated = True
#
#         x = np.expand_dims(x, axis=0)
#         dfx = self._predict_deriv(x)
#         return dfx[0]
#
#
# def legendre(x, d):
#     """Evaluate Legendre polynomials at all coordinates in x.
#
#     :param x: Array of coordinates
#     :type x: numpy.array
#     :param d: Max degree of polynomials
#     :type d: int
#     :return: A x.shape-by-d array of Legendre polynomial values
#     :rtype: numpy.array
#     """
#
#     x = np.array(x)
#     s = x.shape + (d+1,)
#     x = np.ravel(x)
#     P = np.zeros((x.shape[0], d+1))
#     P[:, 0] = 1
#     if d > 0:
#         P[:, 1] = x
#     for n in range(1, d):
#         P[:, n+1] = ((2*n+1)*(x*P[:, n]) - n*P[:, n-1])/(n+1)
#     return P.reshape(s)
#
#
# def dlegendre(x, d):
#     """Evaluate Legendre polynomial derivatives at all coordinates in x.
#
#     :param x: Array of coordinates
#     :type x: numpy.array
#     :param d: Max degree of polynomials
#     :type d: int
#     :return: x.shape-by-d arrays of Legendre polynomial values and derivatives
#     :rtype: numpy.array
#     """
#
#     x = np.array(x)
#     s = x.shape + (d+1,)
#     x = np.ravel(x)
#     P = np.zeros((x.shape[0], d+1))
#     dP = np.zeros((x.shape[0], d+1))
#     P[:, 0] = 1
#     if d > 0:
#         P[:, 1] = x
#         dP[:, 1] = 1
#     for n in range(1,d):
#         P[:, n+1] = ((2*n+1)*(x*P[:, n]) - n*P[:, n-1])/(n+1)
#         dP[:, n+1] = ((2*n+1)*(P[:, n] + x*dP[:, n]) - n*dP[:, n-1])/(n+1)
#     return P.reshape(s), dP.reshape(s)
#
#
# def basis_base(n, testf):
#     """Generate list of shape functions for a subset of a TP poly space.
#
#     :param n: Dimension of the space
#     :type n: int
#     :param testf: Return True if a given multi-index is in range
#     :type testf: Object
#     :return: An N-by-n matrix with S(i,j) = degree of variable j in shape i
#     :rtype: numpy.array
#     """
#
#     snext = np.zeros((n,), dtype=np.int32)
#     done = False
#
#     # Follow carry chain through
#     s = []
#     while not done:
#         s.append(snext.copy())
#         done = True
#         for i in range(n):
#             snext[i] += 1
#             if testf(snext):
#                 done = False
#                 break
#             snext[i] = 0
#     return np.array(s)
#
#
# def basis_TP(n, d):
#     """Generate list of shape functions for TP poly space.
#
#     :param n: Dimension of the space
#     :type n: int
#     :param d: Degree bound
#     :type d: int
#     :return: An N-by-n matrix with S(i,j) = degree of variable j in shape i
#            There are N = n^d shapes.
#     :rtype: numpy.array
#     """
#
#     return basis_base(n, lambda s: np.all(s <= d))
#
#
# def basis_TD(n, d):
#     """Generate list of shape functions for TD poly space.
#
#     :param n: Dimension of the space
#     :type n: int
#     :param d: Degree bound
#     :type d: int
#     :return: An N-by-n matrix with S(i,j) = degree of variable j in shape i
#     :rtype: numpy.array
#     """
#
#     return basis_base(n, lambda s: np.sum(s) <= d)
#
#
# def basis_HC(n, d):
#     """Generate list of shape functions for HC poly space.
#
#     :param n: Dimension of the space
#     :type n: int
#     :param d: Degree bound
#     :type d: int
#     :return: An N-by-n matrix with S(i,j) = degree of variable j in shape i
#     :rtype: numpy.array
#     """
#
#     return basis_base(n, lambda s: np.prod(s+1) <= d+1)
#
#
# def basis_SM(n, d):
#     """Generate list of shape functions for SM poly space.
#
#     :param n: Dimension of the space
#     :type n: int
#     :param d: Degree bound
#     :type d: int
#     :return: An N-by-n matrix with S(i,j) = degree of variable j in shape i
#     :rtype: numpy.array
#     """
#
#     def fSM(p):
#         return p if p < 2 else np.ceil(np.log2(p))
#
#     def fSMv(s):
#         f = 0
#         for j in range(s.shape[0]):
#             f += fSM(s[j])
#         return f
#
#     return basis_base(n, lambda s: fSMv(s) <= fSM(d))


class MARSInterpolant(Surrogate):
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

    :ivar npts: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar x: Interpolation points
    :ivar fx: Function evaluations of interpolation points
    :ivar dim: Number of dimensions
    :ivar model: MARS interpolation model
    """

    def __init__(self, dim, maxpts=100):

        try:
            from pyearth import Earth
        except ImportError as err:
            print("Failed to import pyearth")
            raise err

        self._npts = 0
        self._maxpts = maxpts
        self._X = None
        self._fX = None
        self._dim = dim
        self.model = Earth()
        self.updated = False

    @property
    def dim(self):
        return self._dim

    @property
    def npts(self):
        return self._npts

    @property
    def maxpts(self):
        return self._maxpts

    @property
    def X(self):
        """Get the list of data points

        :return: List of data points
        :rtype: numpy.array
        """

        return self._X[:self.npts, :]

    @property
    def fX(self):
        """Get the list of function values for the data points.

        :return: List of function values
        :rtype: numpy.array
        """

        return self._fX[:self.npts]

    def reset(self):
        """Reset the interpolation."""

        self._npts = 0
        self._X = None
        self._fX = None
        self.updated = False

    def _realloc(self, extra=1):
        """Expand allocation to accommodate more points (if needed)

        :param extra: Number of additional points to accommodate
        :type extra: int
        """

        maxp = self.maxpts
        if maxp < self.npts + extra or self.npts == 0:
            while maxp < self.npts + extra: maxp = 2 * maxp
            self._maxpts = maxp
            self._X = reallocate(self._X, (maxp, self.dim))
            self._fX = reallocate(self._fX, (maxp,))

    def add_points(self, xx, fx):
        """Add a new function evaluation

        :param xx: Points to add
        :type xx: numpy.ndarray
        :param fx: The function values of the point to add
        :type fx: numpy.array or float
        """

        xx = np.atleast_2d(xx)
        newpts = xx.shape[0]
        self._realloc(extra=newpts)

        self._X[self.npts:self.npts + newpts, :] = xx
        self._fX[self.npts:self.npts + newpts] = fx
        self._npts += newpts
        self.updated = False

    def eval(self, x, ds=None):
        """Evaluate the MARS interpolant at the points x

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the MARS interpolant at x, of length npts
        :rtype: numpy.array
        """

        if self.updated is False:
            self.model.fit(self.X, self.fX)
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
            self.model.fit(self.X, self.fX)
        self.updated = True

        x = np.expand_dims(x, axis=0)
        dfx = self.model.predict_deriv(x, variables=None)
        return dfx[0]


# class EnsembleSurrogate(Surrogate):
#     """Compute and evaluate an ensemble of interpolants.
#
#     Maintains a list of surrogates and decides how to weights them
#     by using Dempster-Shafer theory to assign pignistic probabilities
#     based on statistics computed using LOOCV.
#
#     :param model_list: List of surrogate models
#     :type model_list: list
#     :param maxp: Maximum number of points
#     :type maxp: int
#
#     :ivar npts: Current number of points
#     :ivar maxp: Initial maximum number of points (can grow)
#     :ivar rhs: Right hand side for interpolation system
#     :ivar x: Interpolation points
#     :ivar fx: Values at interpolation points
#     :ivar dim: Number of dimensions
#     :ivar model_list: List of surrogate models
#     :ivar weights: Weight for each surrogate model
#     :ivar surrogate_list: List of internal surrogate models for LOOCV
#     """
#
#     def __init__(self, model_list, maxp=100):
#
#         self.npts = 0
#         self.maxp = maxp
#         self.x = None     # pylint: disable=invalid-name
#         self.fx = None
#         self.dim = None
#         assert len(model_list) >= 2, "I need at least two models"
#         self.model_list = model_list
#         self.M = len(model_list)
#         for i in range(self.M):
#             self.model_list[i].reset()  # Models must be empty
#         self.weights = None
#         self.surrogate_list = None
#
#     def reset(self):
#         """Reset the ensemble surrogate."""
#
#         self.npts = 0
#         self.x = None
#         self.fx = None
#         for i in range(len(self.model_list)):
#             self.model_list[i].reset()
#         self.surrogate_list = None
#         self.weights = None
#
#     def _alloc(self, dim):
#         """Allocate storage for x, fx, surrogate_list
#
#         :param dim: Number of dimensions
#         :type dim: int
#         """
#
#         maxp = self.maxp
#         self.dim = dim
#         self.x = np.zeros((maxp, dim))
#         self.fx = np.zeros((maxp, 1))
#         self.surrogate_list = [
#             [None for _ in range(maxp)] for _ in range(self.M)]
#
#     def _realloc(self, dim, extra=1):
#         """Expand allocation to accommodate more points (if needed)
#
#         :param dim: Number of dimensions
#         :param dim: int
#         :param extra: Number of additional points to accommodate
#         :param extra: int
#         """
#
#         if self.npts == 0:
#             self._alloc(dim)
#         elif self.npts + extra > self.maxp - 1:
#             oldmaxp = self.maxp
#             self.maxp = max([self.maxp*2, self.maxp + extra])
#             self.x.resize((self.maxp, dim))
#             self.fx.resize((self.maxp, 1))
#             # Expand the surrogate lists
#             for i in range(self.M):
#                 for _ in range(self.maxp - oldmaxp):
#                     self.surrogate_list[i].append(None)
#
#     def _prob_to_mass(self, prob):
#         """Internal method for building a mass function from probabilities
#
#         :param prob: List of probabilities
#         :type prob: list
#         :return: A MassFunction object constructed from the pignistic probabilities
#         :rtype: MassFunction
#         """
#
#         dictlist = []
#         for i in range(len(prob)):
#             dictlist.append([str(i+1), prob[i]])
#         return MassFunction(dict(dictlist))
#
#     def _mean_squared_error(self, x, y):
#         """Mean squared error of x and y
#
#         Returns :math:`\frac{1}{n} \sum_{i=1}^n (x_i - y_i)^2`
#
#         :param x: Dataset 1, of length n
#         :type x: numpy.array
#         :param y: Dataset 1, of length n
#         :type y: numpy.array
#         :return: the MSE of x and y
#         :rtype: float
#         """
#
#         return np.sum((x - y) ** 2)/len(x)
#
#     def _mean_abs_err(self, x, y):
#         """Mean absolute error of x and y
#
#         Returns :math:`\frac{1}{n} \sum_{i=1}^n |x_i - y_i)|`
#
#         :param x: Dataset 1, of length n
#         :type x: numpy.array
#         :param y: Dataset 1, of length n
#         :type y: numpy.array
#         :return: the MAE of x and y
#         :rtype: float
#         """
#
#         return np.sum(np.abs(x - y))/len(x)
#
#     def compute_weights(self):
#         """Compute mode weights
#
#         Given n observations we use n surrogates built with n-1 of the points
#         in order to predict the value at the removed point. Based on these n
#         predictions we calculate three different statistics:
#
#             - Correlation coefficient with true function values
#             - Root mean square deviation
#             - Mean absolute error
#
#         Based on these three statistics we compute the model weights by
#         applying Dempster-Shafer theory to first compute the pignistic
#         probabilities, which are taken as model weights.
#
#         :return: Model weights
#         :rtype: numpy.array
#         """
#
#         # Do the leave-one-out experiments
#         loocv = np.zeros((self.M, self.npts))
#         for i in range(self.M):
#             for j in range(self.npts):
#                 loocv[i, j] = self.surrogate_list[i][j].eval(self.x[j, :])
#
#         # Compute the model characteristics
#         corr_coeff = np.ones(self.M)
#         for i in range(self.M):
#             corr_coeff[i] = np.corrcoef(np.vstack(
#                 (loocv[i, :], self.get_fx().flatten())))[0, 1]
#
#         root_mean_sq_err = np.ones(self.M)
#         for i in range(self.M):
#             root_mean_sq_err[i] = 1.0 / math.sqrt(
#                 self._mean_squared_error(self.get_fx().flatten(), loocv[i, :]))
#
#         mean_abs_err = np.ones(self.M)
#         for i in range(self.M):
#             mean_abs_err[i] = 1.0 / self._mean_abs_err(
#                 self.get_fx().flatten(), loocv[i, :])
#
#         # Make sure no correlations are negative
#         corr_coeff[np.where(corr_coeff < 0.0)] = 0.0
#         if np.max(corr_coeff) == 0.0:
#             corr_coeff += 1.0
#
#         # Normalize the test statistics
#         corr_coeff /= np.sum(corr_coeff)
#         root_mean_sq_err /= np.sum(root_mean_sq_err)
#         mean_abs_err /= np.sum(mean_abs_err)
#
#         # Create mass functions based on the model characteristics
#         m1 = self._prob_to_mass(corr_coeff)
#         m2 = self._prob_to_mass(root_mean_sq_err)
#         m3 = self._prob_to_mass(mean_abs_err)
#
#         # Compute pignistic probabilities from Dempster-Shafer theory
#         pignistic = m1.combine_conjunctive([m2, m3]).to_dict()
#         self.weights = np.ones(self.M)
#         for i in range(self.M):
#             self.weights[i] = pignistic.get(str(i+1))
#
#     def get_x(self):
#         """Get the list of data points
#
#         :return: List of data points
#         :rtype: numpy.array
#         """
#
#         return self.x[:self.npts, :]
#
#     def get_fx(self):
#         """Get the list of function values for the data points.
#
#         :return: List of function values
#         :rtype: numpy.array
#         """
#
#         return self.fx[:self.npts, :]
#
#     def add_points(self, xx, fx):
#         """Add a new function evaluation
#
#         This function also updates the list of LOOCV surrogate models by cleverly
#         just adding one point to n of the models. The scheme in which new models
#         are built is illustrated below:
#
#         2           1           1,2
#
#         2,3         1,3         1,2         1,2,3
#
#         2,3,4       1,3,4       1,2,4       1,2,3       1,2,3,4
#
#         2,3,4,5     1,3,4,5     1,2,4,5     1,2,3,5     1,2,3,4     1,2,3,4,5
#
#         :param xx: Point to add
#         :type xx: numpy.array
#         :param fx: The function value of the point to add
#         :type fx: float
#         """
#
#         dim = len(xx)
#         self._realloc(dim)
#         self.x[self.npts, :] = xx
#         self.fx[self.npts, :] = fx
#         self.npts += 1
#         # Update the leave-one-out models
#         if self.npts == 2:
#             for i in range(self.M):
#                 #  Add the first three models
#                 x0 = copy(self.x[0, :])
#                 x1 = copy(self.x[1, :])
#                 self.surrogate_list[i][0] = deepcopy(self.model_list[i])
#                 self.surrogate_list[i][0].add_points(x1, self.fx[1])
#                 self.surrogate_list[i][1] = deepcopy(self.model_list[i])
#                 self.surrogate_list[i][1].add_points(x0, self.fx[0])
#                 self.surrogate_list[i][2] = deepcopy(self.surrogate_list[i][1])
#                 self.surrogate_list[i][2].add_points(x1, self.fx[1])
#         elif self.npts > 2:
#             for i in range(self.M):
#                 for j in range(self.npts-1):
#                     self.surrogate_list[i][j].add_points(xx, fx)
#                 self.surrogate_list[i][self.npts] = deepcopy(
#                     self.surrogate_list[i][self.npts-1])
#                 self.surrogate_list[i][self.npts].add_points(xx, fx)
#                 # Point to the model with all points
#                 self.model_list[i] = self.surrogate_list[i][self.npts]
#         self.weights = None
#
#     def eval(self, x, ds=None):
#         """Evaluate the ensemble surrogate at the points xx
#
#         :param x: Points where to evaluate, of size npts x dim
#         :type x: numpy.array
#         :param ds: Distances between the centers and the points x, of size npts x ncenters
#         :type ds: numpy.array
#         :return: Values of the ensemble surrogate at x, of length npts
#         :rtype: numpy.array
#         """
#
#         if self.weights is None:
#             self.compute_weights()
#
#         vals = np.zeros((x.shape[0], 1))
#         for i in range(self.M):
#             vals += self.weights[i] * self.model_list[i].eval(x, ds)
#
#         return vals
#
#     def deriv(self, x, d=None):
#         """Evaluate the derivative of the ensemble surrogate at the point x
#
#         :param x: Point for which we want to compute the RBF gradient
#         :type x: numpy.array
#         :return: Derivative of the ensemble surrogate at x
#         :rtype: numpy.array
#         """
#         if self.weights is None:
#             self.compute_weights()
#
#         val = 0.0
#         for i in range(self.M):
#             val += self.weights[i]*self.model_list[i].deriv(x, d)
#         return val


class RSCapped(Surrogate):
    """Cap adapter for response surfaces.

    This adapter takes an existing response surface and replaces it
    with a modified version in which the function values are replaced
    according to some transformation. A very common transformation
    is to replace all values above the median by the median in order
    to reduce the influence of large function values.
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


class RSPenalty(Surrogate):
    """Penalty adapter for response surfaces.

    This adapter can be used for approximating an objective function plus
    a penalty function. The response surface is fitted only to the objective
    function and the penalty is added on after.
    """

    def __init__(self, model, evals, derivs):
        self.model = model
        self.eval_method = evals
        self.deriv_method = derivs

    @property
    def dim(self):
        return self.model.dim

    @property
    def npts(self):
        return self.model.npts

    @property
    def maxpts(self):
        return self.model.maxpts

    @property
    def X(self):
        return self.model.X

    @property
    def fX(self):
        return self.model.fX

    def reset(self):
        self.model.reset()

    def add_points(self, xx, fx):
        self.model.add_points(xx, fx)

    def eval(self, x, ds=None):
        return self.eval_method(self.model, x)

    def deriv(self, x, ds=None):
        return self.deriv_method(self.model, x)


class RSUnitbox(Surrogate):
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

    def add_points(self, xx, fx):
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
        self.model.add_points(to_unit_box(xx, self.data), fx)

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
