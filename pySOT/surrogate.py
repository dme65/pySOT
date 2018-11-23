"""
.. module:: surrogate
   :synopsis: Surrogate models

.. moduleauthor:: David Eriksson <dme65@cornell.edu>

:Module: surrogate
:Author: David Eriksson <dme65@cornell.edu>

"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.spatial as scpspatial
import scipy.linalg as scplinalg
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from pySOT.utils import to_unit_box, from_unit_box
import warnings


def reallocate(A, dims, **kwargs):
    """Reallocate A with at most 2 dimensions to have size according to dims
    
    TODO: Move to utils
    """
    if A is None:
        A = np.zeros(dims, **kwargs)
        return A

    assert(A.ndim <= 2 and A.ndim == len(dims))
    assert(np.all(dims >= A.shape))
    AA = np.zeros(dims, **kwargs)
    if A.ndim == 1:
        AA[:A.shape[0]] = A
    else:
        AA[:A.shape[0], :A.shape[1]] = A
    return AA


class Surrogate(ABC):
    def __init__(self):
        self.dim = None
        self.npts = None
        self.X = None
        self.fX = None
        self.updated = None

    def reset(self):
        """Reset the interpolation."""
        self.npts = 0
        self.X = np.empty([0, self.dim])
        self.fX = np.empty([0, 1])
        self.updated = False

    def add_points(self, xx, fx):
        """Add a new function evaluation

        This method SHOULD NOT trigger a new fit, it just updates X and fX

        :param xx: Points to add
        :type xx: numpy.ndarray
        :param fx: The function values of the point to add
        :type fx: numpy.array or float
        """

        xx = np.atleast_2d(xx)
        if fx.ndim == 0: fx = np.expand_dims(fx, axis=0)
        if fx.ndim == 1: fx = np.expand_dims(fx, axis=1)
        assert xx.shape[0] == fx.shape[0] and xx.shape[1] == self.dim
        newpts = xx.shape[0]
        self.X = np.vstack((self.X, xx))
        self.fX = np.vstack((self.fX, fx))
        self.npts += newpts
        self.updated = False

    @abstractmethod
    def eval(self, xx):  # pragma: no cover
        """Evaluate interpolant at points xx

        xx must be of size npts x dim or (dim, )
        """
        return

    @abstractmethod
    def deriv(self, X):  # pragma: no cover
        """Evaluate derivative of interpolant at points xx

        xx must be of size npts x dim or (dim, )
        """
        return


class Kernel(ABC):
    def __init__(self):  # pragma: no cover
        self.order = None

    @abstractmethod
    def eval(self, dists):  # pragma: no cover
        pass

    @abstractmethod
    def deriv(self, dists):  # pragma: no cover
        pass


class Tail(ABC):
    def __init__(self):  # pragma: no cover
        self.degree = None
        self.dim = None
        self.dim_tail = None

    @abstractmethod
    def eval(self, X):  # pragma: no cover
        pass

    @abstractmethod
    def deriv(self, x):  # pragma: no cover
        pass


class CubicKernel(Kernel):
    """Cubic RBF kernel

    This is a basic class for the Cubic RBF kernel: :math:`\\varphi(r) = r^3` which is
    conditionally positive definite of order 2.
    """

    def __init__(self):
        super().__init__()
        self.order = 2

    def eval(self, dists):
        """Evaluates the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\\|x_i - x_j \\|^3`
        :rtype: numpy.array
        """

        return dists ** 3

    def deriv(self, dists):
        """Evaluates the derivative of the Cubic kernel for a distance matrix.

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`3 \\| x_i - x_j \\|^2`
        :rtype: numpy.array
        """

        return 3 * dists ** 2


class TPSKernel(Kernel):
    """Thin-plate spline RBF kernel

    This is a basic class for the TPS RBF kernel: :math:`\\varphi(r) = r^2 \\log(r)` which is
    conditionally positive definite of order 2.
    """

    def __init__(self):
        super().__init__()
        self.order = 2

    def eval(self, dists):
        """Evaluates the TPS kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\\|x_i - x_j \\|^2 \\log (\\|x_i - x_j \\|)`
        :rtype: numpy.array
        """

        dists[dists < np.finfo(float).eps] = np.finfo(float).eps
        return (dists ** 2) * np.log(dists)

    def deriv(self, dists):
        """Evaluates the derivative of the TPS kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\\|x_i - x_j \\|(1 + 2 \\log (\\|x_i - x_j \\|) )`
        :rtype: numpy.array
        """

        dists[dists < np.finfo(float).eps] = np.finfo(float).eps
        return dists * (1 + 2 * np.log(dists))


class LinearKernel(Kernel):
    """Linear RBF kernel

     This is a basic class for the Linear RBF kernel: :math:`\\varphi(r) = r` which is
     conditionally positive definite of order 1.
     """

    def __init__(self):
        super().__init__()
        self.order = 1

    def eval(self, dists):
        """Evaluates the Linear kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\\|x_i - x_j \\|`
        :rtype: numpy.array
        """

        return dists

    def deriv(self, dists):
        """Evaluates the derivative of the Linear kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is 1
        :rtype: numpy.array
        """

        return np.ones(dists.shape)


class LinearTail(Tail):
    """Linear polynomial tail.

    This is a standard linear polynomial in d-dimension, built from the basis
    :math:`\\{1,x_1,x_2,\\ldots,x_d\\}`.
    """

    def __init__(self, dim):
        super().__init__()
        self.degree = 1
        self.dim = dim
        self.dim_tail = 1 + dim

    def eval(self, X):
        """Evaluates the linear polynomial tail for a set of points

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
        """Evaluates the gradient of the linear polynomial tail for one point

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
    :math:`\\{ 1 \\}`.
    """

    def __init__(self, dim):
        super().__init__()
        self.degree = 0
        self.dim = dim
        self.dim_tail = 1

    def eval(self, X):
        """Evaluates the constant polynomial tail for a set of points

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
        """Evaluates the gradient of the linear polynomial tail for one point

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

    TODO: Update this interface to match the abstract class
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

        if kernel.order - 1 > tail.degree:
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

    def transform_fx(self, fX):
        self._fX = fX
        self.rhs[self.ntail:self.ntail + self.npts] = fX

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
                except:  # Compute a new LU factorization if the Cholesky fails
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

    def eval(self, x):
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
        ds = scpspatial.distance.cdist(x, self._X[:self.npts, :])
        fx = self.kernel.eval(ds)*c[ntail:ntail + self.npts] + self.tail.eval(x)*c[:ntail]
        return fx

    def deriv(self, xx):
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

        ds = scpspatial.distance.cdist(self.X, xx)
        ds[ds < np.finfo(float).eps] = np.finfo(float).eps  # Better safe than sorry

        dfxx = np.zeros((xx.shape[0], self.dim))
        for i in range(xx.shape[0]):
            x = np.atleast_2d(xx[i, :])
            ntail = self.ntail
            dpx = self.tail.deriv(x)
            c = self.coeffs()
            dfx = np.dot(dpx, c[:ntail]).transpose()
            dsx = -self.X
            dsx += x
            dss = np.atleast_2d(ds[:, i]).T
            dsx *= (np.multiply(self.kernel.deriv(dss), c[ntail:]) / dss)
            dfx += np.sum(dsx, 0)
            dfxx[i, :] = dfx

        return dfxx


class GPRegressor(Surrogate):
    """Compute and evaluate a GP

    Gaussian Process Regression object.

    More details:
        http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
    """

    def __init__(self, dim, gp=None, n_restarts_optimizer=3):
        self.npts = 0
        self.dim = dim
        self.X = np.empty([0, dim])     # pylint: disable=invalid-name
        self.fX = np.empty([0, 1])
        if gp is None:
            kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (0.1, 100)) + WhiteKernel(1e-3, (1e-6, 1e-2))
            self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
        else:
            self.model = gp
            if not isinstance(gp, GaussianProcessRegressor):
                raise TypeError("gp is not of type GaussianProcessRegressor")
        self.updated = False

    def _fit(self):
        """Fit the model"""
        if not self.updated:
            self.model.fit(self.X, self.fX)
            self.updated = True

    def eval(self, x):
        """Evaluate the GP regression object at the points x

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the GP regression object at x, of length npts
        :rtype: numpy.array
        """
        self._fit()
        x = np.atleast_2d(x)
        return self.model.predict(x)

    def deriv(self, x):
        """Evaluate the GP regression object at a point x

        :param x: Point for which we want to compute the GP regression gradient
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Derivative of the GP regression object at x
        :rtype: numpy.array
        """

        raise NotImplementedError


class MARSInterpolant(Surrogate):
    """Compute and evaluate a MARS interpolant

    MARS builds a model of the form

    .. math::

        \\hat{f}(x) = \\sum_{i=1}^{k} c_i B_i(x).

    The model is a weighted sum of basis functions :math:`B_i(x)`. Each basis
    function :math:`B_i(x)` takes one of the following three forms:

    1. a constant 1.
    2. a hinge function of the form :math:`\\max(0, x - const)` or \
       :math:`\\max(0, const - x)`. MARS automatically selects variables \
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

    def __init__(self, dim):

        try:
            from pyearth import Earth
        except ImportError as err:
            print("Failed to import pyearth")
            raise err

        self.npts = 0
        self.X = np.empty([0, dim])
        self.fX = np.empty([0, 1])
        self.dim = dim
        self.model = Earth()
        self.updated = False

    def _fit(self):
        warnings.simplefilter("ignore")  # Surpress deprecation warnings from py-earth
        if self.updated is False:
            self.model.fit(self.X, self.fX)
            self.updated = True

    def eval(self, x):
        """Evaluate the MARS interpolant at the points x

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the MARS interpolant at x, of length npts
        :rtype: numpy.array
        """

        self._fit()
        x = np.atleast_2d(x)
        return np.expand_dims(self.model.predict(x), axis=1)

    def deriv(self, x):
        """Evaluate the derivative of the MARS interpolant at a point x

        :param x: Point for which we want to compute the MARS gradient
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Derivative of the MARS interpolant at x
        :rtype: numpy.array
        """

        self._fit()
        x = np.expand_dims(x, axis=0)
        dfx = self.model.predict_deriv(x, variables=None)
        return dfx[0]


class PolyRegressor(Surrogate):
    """Computes a polynomial regression model

    :param maxp: Initial capacity
    :type maxp: int

    :ivar npts: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar x: Interpolation points
    :ivar fx: Function evaluations of interpolation points
    :ivar dim: Number of dimensions
    """

    def __init__(self, dim, degree=2):
        self.npts = 0
        self.X = np.empty([0, dim])
        self.fX = np.empty([0, 1])
        self.dim = dim
        self.updated = False
        self.model = make_pipeline(PolynomialFeatures(degree), Ridge())

    def _fit(self):
        """Fit the model"""
        if not self.updated:
            self.model.fit(self.X, self.fX)
            self.updated = True

    def eval(self, x):
        """Evaluate the MARS interpolant at the points x

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the MARS interpolant at x, of length npts
        :rtype: numpy.array
        """
        self._fit()
        x = np.atleast_2d(x)
        return self.model.predict(x)

    def deriv(self, x):
        """TODO: Not implemented"""
        raise NotImplementedError


class SurrogateCapped(Surrogate):
    """Cap adapter for response surfaces.

    This adapter takes an existing response surface and replaces it
    with a modified version in which the function values are replaced
    according to some transformation. A very common transformation
    is to replace all values above the median by the median in order
    to reduce the influence of large function values.
    """

    def __init__(self, model, transformation=None):
        self.npts = 0
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

        assert(isinstance(model, Surrogate))
        self.model = model

    def reset(self):
        super().reset()
        self.model.reset()

    def add_points(self, xx, fx):
        super().add_points(xx, fx)
        self.model.add_points(xx, fx)
        self.model.fX = self.transformation(np.copy(self.fX))  # Apply transformation

    def eval(self, x):
        return self.model.eval(x)

    def deriv(self, x):
        return self.model.deriv(x)


class SurrogateUnitBox(Surrogate):
    """Unit box adapter for response surfaces

    This adapter takes an existing response surface and replaces it
    with a modified version where the domain is rescaled to the unit
    box. This is useful for response surfaces that are sensitive to
    scaling, such as radial basis functions.

    :param model: Original response surface object
    :type model: Object

    :ivar model: original response surface object
    """

    def __init__(self, model, lb, ub):
        self.npts = 0
        self.X = np.empty([0, model.dim])
        self.fX = np.empty([0, 1])
        self.dim = model.dim
        self.updated = False

        assert(isinstance(model, Surrogate))
        self.model = model
        self.lb = lb
        self.ub = ub

    def reset(self):
        super().reset()
        self.model.reset()

    def add_points(self, xx, fx):
        super().add_points(xx, fx)
        self.model.add_points(to_unit_box(xx, self.lb, self.ub), fx)

    def eval(self, x):
        return self.model.eval(to_unit_box(x, self.lb, self.ub))

    def deriv(self, x):
        """Remember the chain rule.

        f'(x) = (d/dx) g((x-a)/(b-a)) = g'((x-a)/(b-a)) * 1/(b-a)
        """
        return self.model.deriv(to_unit_box(x, self.lb, self.ub)) / (self.ub - self.lb)
