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
from pySOT.utils import to_unit_box
import warnings


class Surrogate(ABC):
    """Base class for a surrogate model.

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    """
    def __init__(self):  # pragma: no cover
        self.dim = None
        self.num_pts = None
        self.X = None
        self.fX = None
        self.updated = None

    def reset(self):
        """Reset the surrogate."""
        self.num_pts = 0
        self.X = np.empty([0, self.dim])
        self.fX = np.empty([0, 1])
        self.updated = False

    def add_points(self, xx, fx):
        """Add new function evaluations.

        This method SHOULD NOT trigger a new fit, it just updates X
        and fX but leaves the original surrogate object intact

        :param xx: Points to add
        :type xx: numpy.ndarray
        :param fx: The function values of the point to add
        :type fx: numpy.array or float
        """
        xx = np.atleast_2d(xx)
        if isinstance(fx, float):
            fx = np.array([fx])
        if fx.ndim == 0:
            fx = np.expand_dims(fx, axis=0)
        if fx.ndim == 1:
            fx = np.expand_dims(fx, axis=1)
        assert xx.shape[0] == fx.shape[0] and xx.shape[1] == self.dim
        newpts = xx.shape[0]
        self.X = np.vstack((self.X, xx))
        self.fX = np.vstack((self.fX, fx))
        self.num_pts += newpts
        self.updated = False

    @abstractmethod
    def predict(self, xx):  # pragma: no cover
        """Evaluate surroagte at points xx.

        :param xx: xx must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Surrogate predictions, of size num_pts x 1
        :rtype: numpy.ndarray
        """
        return

    @abstractmethod
    def predict_deriv(self, xx):  # pragma: no cover
        """Evaluate derivative of interpolant at points xx.

        :param xx: xx must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Surrogate derivative predictions, of size num_pts x dim
        :rtype: numpy.ndarray
        """
        return


class Kernel(ABC):
    """Base class for a radial kernel.

    :ivar order: Order of the conditionally positive definite kernel
    """
    def __init__(self):  # pragma: no cover
        self.order = None

    @abstractmethod
    def eval(self, dists):  # pragma: no cover
        """Evaluate the radial kernel.

        :param dists: Array of size n x n with pairwise distances
        :type dists: numpy.ndarray

        :return: Array of size n x n with kernel values
        :rtype: numpy.ndarray
        """
        pass

    @abstractmethod
    def deriv(self, dists):  # pragma: no cover
        """Evaluate derivatives of radial kernel wrt distance.

        :param dists: Array of size n x n with pairwise distances
        :type dists: numpy.ndarray

        :return: Array of size n x n with kernel derivatives
        :rtype: numpy.ndarray
        """
        pass


class Tail(ABC):
    """Base class for a polynomial tail.

    "ivar dim: Dimensionality of the original space
    :ivar dim_tail: Dimensionality of the polynomial space \
        (number of basis functions)
    """
    def __init__(self):  # pragma: no cover
        self.degree = None
        self.dim = None
        self.dim_tail = None

    @abstractmethod
    def eval(self, X):  # pragma: no cover
        """Evaluate the polynomial tail.

        :param X: Array of size num_pts x dim
        :type X: numpy.ndarray

        :return: Array of size num_pts x dim_tail
        :rtype: numpy.ndarray
        """
        pass

    @abstractmethod
    def deriv(self, x):  # pragma: no cover
        """Evaluate derivative of the polynomial tail.

        :param x: Array of size 1 x dim or (dim,)
        :type x: numpy.ndarray

        :return: Array of size dim_tail x dim
        :rtype: numpy.ndarray
        """
        pass


class CubicKernel(Kernel):
    """Cubic RBF kernel

    This is a class for the Cubic RBF kernel: :math:`\\varphi(r) = r^3` which
    is conditionally positive definite of order 2.
    """

    def __init__(self):
        super().__init__()
        self.order = 2

    def eval(self, dists):
        """Evaluates the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\|x_i - x_j \\|^3`
        :rtype: numpy.array
        """
        return dists ** 3

    def deriv(self, dists):
        """Evaluates the derivative of the Cubic kernel for a distance matrix.

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is
            :math:`3 \\| x_i - x_j \\|^2`
        :rtype: numpy.array
        """
        return 3 * dists ** 2


class TPSKernel(Kernel):
    """Thin-plate spline RBF kernel.

    This is a basic class for the TPS RBF kernel:
    :math:`\\varphi(r) = r^2 \\log(r)` which is
    conditionally positive definite of order 2.
    """
    def __init__(self):
        super().__init__()
        self.order = 2

    def eval(self, dists):
        """Evaluate the TPS kernel.

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\|x_i - x_j \\|^2 \\log (\\|x_i - x_j \\|)`
        :rtype: numpy.array
        """
        dists[dists < np.finfo(float).eps] = np.finfo(float).eps
        return (dists ** 2) * np.log(dists)

    def deriv(self, dists):
        """Evaluate the derivative of the TPS kernel.

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\|x_i - x_j \\|(1 + 2 \\log (\\|x_i - x_j \\|) )`
        :rtype: numpy.array
        """
        dists[dists < np.finfo(float).eps] = np.finfo(float).eps
        return dists * (1 + 2 * np.log(dists))


class LinearKernel(Kernel):
    """Linear RBF kernel.

     This is a basic class for the Linear RBF kernel:
     :math:`\\varphi(r) = r` which is
     conditionally positive definite of order 1.
     """
    def __init__(self):
        super().__init__()
        self.order = 1

    def eval(self, dists):
        """Evaluate the Linear kernel.

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\|x_i - x_j \\|`
        :rtype: numpy.array
        """
        return dists

    def deriv(self, dists):
        """Evaluate the derivative of the Linear kernel.

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is 1
        :rtype: numpy.array
        """
        return np.ones(dists.shape)


class LinearTail(Tail):
    """Linear polynomial tail.

    This is a standard linear polynomial in d-dimension, built from
    the basis :math:`\\{1,x_1,x_2,\\ldots,x_d\\}`.
    """
    def __init__(self, dim):
        super().__init__()
        self.degree = 1
        self.dim = dim
        self.dim_tail = 1 + dim

    def eval(self, X):
        """Evaluate the linear polynomial tail.

        :param X: Points to evaluate, of size num_pts x dim
        :type X: numpy.array

        :returns: A numpy.array of size num_pts x dim_tail
        :rtype: numpy.array
        """
        X = np.atleast_2d(X)
        if X.shape[1] != self.dim:
            raise ValueError("Input has the wrong number of dimensions")
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def deriv(self, x):
        """Evaluate the derivative of the linear polynomial tail

        :param x: Point to evaluate, of size (1, dim) or (dim,)
        :type x: numpy.array

        :returns: A numpy.array of size dim_tail x dim
        :rtype: numpy.array
        """
        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError("Input has the wrong number of dimensions")
        return np.hstack((np.zeros((x.shape[1], 1)), np.eye((x.shape[1]))))


class ConstantTail(Tail):
    """Constant polynomial tail.

    Constant polynomial in d-dimension, built from the basis :math:`\\{ 1 \\}`.
    """
    def __init__(self, dim):
        super().__init__()
        self.degree = 0
        self.dim = dim
        self.dim_tail = 1

    def eval(self, X):
        """Evaluate the constant polynomial tail.

        :param X: Points to evaluate, of size num_pts x dim
        :type X: numpy.array

        :returns: A numpy.array of size num_pts x dim_tail(dim)
        :rtype: numpy.array
        """
        X = np.atleast_2d(X)
        if X.shape[1] != self.dim:
            raise ValueError("Input has the wrong number of dimensions")
        return np.ones((X.shape[0], 1))

    def deriv(self, x):
        """Evaluate the derivative of the constant polynomial tail.

        :param x: Point to evaluate, of size (1, dim) or (dim,)
        :type x: numpy.array

        :returns: A numpy.array of size dim_tail x dim
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

        s(x) = \\sum_j c_j \\phi(\\|x-x_j\\|) + \\sum_j \\lambda_j p_j(x)

    where the functions :math:`p_j(x)` are low-degree polynomials.
    The fitting equations are

    .. math::
        \\begin{bmatrix} \\eta I & P^T \\\\ P & \\Phi+\\eta I \\end{bmatrix}
        \\begin{bmatrix} \\lambda \\\\ c \\end{bmatrix} =
        \\begin{bmatrix} 0 \\\\ f \\end{bmatrix}

    where :math:`P_{ij} = p_j(x_i)` and :math:`\\Phi_{ij}=\\phi(\\|x_i-x_j\\|)`
    The regularization parameter :math:`\\eta` allows us to avoid problems
    with potential poor conditioning of the system. Consider using the
    SurrogateUnitBox wrapper or manually scaling the domain to the unit
    hypercube to avoid issues with the domain scaling.

    We add k new points to the RBFInterpolant in :math:`O(kn^2)` flops by
    updating the LU factorization of the old RBF system. This is better
    than computing the RBF coefficients from scratch, which costs
    :math:`O(n^3)` flops.

    :param dim: Number of dimensions
    :type dim: int
    :param kernel: RBF kernel object
    :type kernel: Kernel
    :param tail: RBF polynomial tail object
    :type tail: Tail
    :param eta: Regularization parameter
    :type eta: float

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar kernel: RBF kernel
    :ivar tail: RBF tail
    :ivar eta: Regularization parameter
    """
    def __init__(self, dim, kernel=None, tail=None, eta=1e-6):
        self.num_pts = 0
        self.dim = dim
        self.X = np.empty([0, dim])     # pylint: disable=invalid-name
        self.fX = np.empty([0, 1])
        self.updated = False

        if kernel is None or tail is None:
            kernel = CubicKernel()
            tail = LinearTail(dim)
        assert(isinstance(kernel, Kernel) and isinstance(tail, Tail))

        self.kernel = kernel
        self.tail = tail
        self.ntail = tail.dim_tail
        self.A = None
        self.L = None
        self.U = None
        self.piv = None
        self.c = None
        self.eta = eta

        if kernel.order - 1 > tail.degree:
            raise ValueError("Kernel and tail mismatch")
        assert self.dim == self.tail.dim

    def reset(self):
        """Reset the RBF interpolant."""
        super().reset()
        self.L = None
        self.U = None
        self.piv = None
        self.c = None

    def _fit(self):
        """Compute new coefficients if the RBF is not updated.

        We try to update an existing LU factorization by computing a Cholesky
        factorization of the Schur complemented system. This may fail if the
        system is ill-conditioned, in which case we compute a new LU
        factorization.
        """
        if not self.updated:
            n = self.num_pts
            ntail = self.ntail
            nact = ntail + n

            if self.c is None:  # Initial fit
                assert self.num_pts >= ntail

                X = self.X[0:n, :]
                D = scpspatial.distance.cdist(X, X)
                Phi = self.kernel.eval(D) + self.eta * np.eye(n)
                P = self.tail.eval(X)

                # Set up the systems matrix
                A1 = np.hstack((np.zeros((ntail, ntail)), P.T))
                A2 = np.hstack((P, Phi))
                A = np.vstack((A1, A2))

                [LU, piv] = scplinalg.lu_factor(A)
                self.L = np.tril(LU, -1) + np.eye(nact)
                self.U = np.triu(LU)

                # Construct the usual pivoting vector so that we can increment
                self.piv = np.arange(0, nact)
                for i in range(nact):
                    self.piv[i], self.piv[piv[i]] = \
                        self.piv[piv[i]], self.piv[i]

            else:  # Extend LU factorization
                k = self.c.shape[0] - ntail
                numnew = n - k
                kact = ntail + k

                X = self.X[:n, :]
                XX = self.X[k:n, :]
                D = scpspatial.distance.cdist(X, XX)
                Pnew = np.vstack((self.tail.eval(XX).T,
                                  self.kernel.eval(D[:k, :])))
                Phinew = self.kernel.eval(D[k:, :]) + self.eta * np.eye(numnew)

                L21 = np.zeros((kact, numnew))
                U12 = np.zeros((kact, numnew))
                for i in range(numnew):  # TODO: Can we use level-3 BLAS?
                    L21[:, i] = scplinalg.solve_triangular(
                        a=self.U, b=Pnew[:kact, i], lower=False, trans='T')
                    U12[:, i] = scplinalg.solve_triangular(
                        a=self.L, b=Pnew[self.piv[:kact], i],
                        lower=True, trans='N')
                L21 = L21.T
                try:  # Compute Cholesky factorization of the Schur complement
                    C = scplinalg.cholesky(
                        a=Phinew - np.dot(L21, U12), lower=True)
                finally:  # Compute a new LU factorization if Cholesky fails
                    self.c = None
                    return self._fit()

                self.piv = np.hstack((self.piv, np.arange(kact, nact)))
                self.L = np.vstack((self.L, L21))
                L2 = np.vstack((np.zeros((kact, numnew)), C))
                self.L = np.hstack((self.L, L2))
                self.U = np.hstack((self.U, U12))
                U2 = np.hstack((np.zeros((numnew, kact)), C.T))
                self.U = np.vstack((self.U, U2))

            # Update coefficients
            rhs = np.vstack((np.zeros((ntail, 1)), self.fX))
            self.c = scplinalg.solve_triangular(
                a=self.L, b=rhs[self.piv], lower=True)
            self.c = scplinalg.solve_triangular(
                a=self.U, b=self.c, lower=False)
            self.updated = True

    def predict(self, xx):
        """Evaluate the RBF interpolant at the points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = np.atleast_2d(xx)
        ds = scpspatial.distance.cdist(xx, self.X)
        ntail = self.ntail
        return np.dot(self.kernel.eval(ds),
                      self.c[ntail:ntail + self.num_pts]) + \
            np.dot(self.tail.eval(xx), self.c[:ntail])

    def predict_deriv(self, xx):
        """Evaluate the derivative of the RBF interpolant at a point xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.array

        :return: Derivative of the RBF interpolant at xx
        :rtype: numpy.array
        """
        self._fit()
        xx = np.atleast_2d(xx)
        if xx.shape[1] != self.dim:
            raise ValueError("Input has incorrect number of dimensions")
        ds = scpspatial.distance.cdist(self.X, xx)
        ds[ds < np.finfo(float).eps] = np.finfo(float).eps  # Avoid 0*inf

        dfxx = np.zeros((xx.shape[0], self.dim))
        for i in range(xx.shape[0]):
            x = np.atleast_2d(xx[i, :])
            ntail = self.ntail
            dpx = self.tail.deriv(x)
            dfx = np.dot(dpx, self.c[:ntail]).transpose()
            dsx = -(self.X.copy())
            dsx += x
            dss = np.atleast_2d(ds[:, i]).T
            dsx *= (np.multiply(self.kernel.deriv(dss), self.c[ntail:]) / dss)
            dfx += np.sum(dsx, 0)
            dfxx[i, :] = dfx

        return dfxx


class GPRegressor(Surrogate):
    """Gaussian process (GP) regressor.

    Wrapper around the GPRegressor in scikit-learn.

    :param dim: Number of dimensions
    :type dim: int
    :param gp: GPRegressor model
    :type gp: object
    :param n_restarts_optimizer: Number of restarts in hyperparam fitting
    :type n_restarts_optimizer: int

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar model: GPRegressor object
    """
    def __init__(self, dim, gp=None, n_restarts_optimizer=3):
        self.num_pts = 0
        self.dim = dim
        self.X = np.empty([0, dim])     # pylint: disable=invalid-name
        self.fX = np.empty([0, 1])
        self.updated = False

        if gp is None:  # Use the SE kernel
            kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(1, (0.1, 100)) + \
                WhiteKernel(1e-3, (1e-6, 1e-2))
            self.model = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
        else:
            self.model = gp
            if not isinstance(gp, GaussianProcessRegressor):
                raise TypeError("gp is not of type GaussianProcessRegressor")

    def _fit(self):
        """Compute new coefficients if the GP is not updated."""
        if not self.updated:
            self.model.fit(self.X, self.fX)
            self.updated = True

    def predict(self, xx):
        """Evaluate the GP regressor at the points xx.

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = np.atleast_2d(xx)
        return self.model.predict(xx)

    def predict_std(self, xx):
        """Predict standard deviation at points xx.

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Predicted standard deviation, of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = np.atleast_2d(xx)
        _, std = self.model.predict(xx, return_std=True)
        return np.expand_dims(std, axis=1)

    def predict_deriv(self, xx):
        """TODO: Not implemented"""
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

    :param dim: Number of dimensions
    :type dim: int

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar model: Earth object
    """
    def __init__(self, dim):
        self.num_pts = 0
        self.X = np.empty([0, dim])
        self.fX = np.empty([0, 1])
        self.dim = dim
        self.updated = False

        try:
            from pyearth import Earth
            self.model = Earth()
        except ImportError as err:
            print("Failed to import pyearth")
            raise err

    def _fit(self):
        """Compute new coefficients if the MARS interpolant is not updated."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Surpress deprecation warnings
            if self.updated is False:
                self.model.fit(self.X, self.fX)
                self.updated = True

    def predict(self, xx):
        """Evaluate the MARS interpolant at the points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = np.atleast_2d(xx)
        return np.expand_dims(self.model.predict(xx), axis=1)

    def predict_deriv(self, xx):
        """Evaluate the derivative of the MARS interpolant at points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.array

        :return: Derivative of the RBF interpolant at xx
        :rtype: numpy.array
        """
        self._fit()
        xx = np.expand_dims(xx, axis=0)
        dfx = self.model.predict_deriv(xx, variables=None)
        return dfx[0]


class PolyRegressor(Surrogate):
    """Multi-variate polynomial regression with cross-terms

    :param dim: Number of dimensions
    :type dim: int
    :param degree: Polynomial degree
    :type degree: int

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar model: scikit-learn pipeline for polynomial regression
    """
    def __init__(self, dim, degree=2):
        self.num_pts = 0
        self.X = np.empty([0, dim])
        self.fX = np.empty([0, 1])
        self.dim = dim
        self.updated = False
        self.model = make_pipeline(PolynomialFeatures(degree), Ridge())

    def _fit(self):
        """Update the polynomial regression model."""
        if not self.updated:
            self.model.fit(self.X, self.fX)
            self.updated = True

    def predict(self, xx):
        """Evaluate the polynomial regressor at the points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = np.atleast_2d(xx)
        return self.model.predict(xx)

    def predict_deriv(self, xx):
        """TODO: Not implemented"""
        raise NotImplementedError


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

        assert(isinstance(model, Surrogate))
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


class SurrogateUnitBox(Surrogate):
    """Unit box adapter for surrogate models.

    This adapter takes an existing surrogate model and replaces it
    by a modified version where the domain is rescaled to the unit
    hypercube. This is useful for surrogate models that are sensitive to
    scaling, such as RBFs.

    :param model: Original surrogate model (must implement Surrogate)
    :type model: object
    :param lb: Lower variable bounds, of size 1 x dim
    :type lb: function
    :param ub: Upper variable bounds, of size 1 x dim
    :type ub: function

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

        assert(isinstance(model, Surrogate))
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
        self.model.add_points(
            to_unit_box(xx, self.lb, self.ub), fx)

    def predict(self, xx):
        """Evaluate the surrogate model at the points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        return self.model.predict(
            to_unit_box(xx, self.lb, self.ub))

    def predict_std(self, x):
        """Predict standard deviation at points xx.

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Predicted standard deviation, of size num_pts x 1
        :rtype: numpy.ndarray
        """
        return self.model.predict_std(
            to_unit_box(x, self.lb, self.ub))

    def predict_deriv(self, x):
        """Evaluate the derivative of the surrogate model at points xx

        Remember the chain rule:
            f'(x) = (d/dx) g((x-a)/(b-a)) = g'((x-a)/(b-a)) * 1/(b-a)

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.array

        :return: Derivative of the RBF interpolant at xx
        :rtype: numpy.array
        """
        return self.model.predict_deriv(
            to_unit_box(x, self.lb, self.ub)) / (self.ub - self.lb)
