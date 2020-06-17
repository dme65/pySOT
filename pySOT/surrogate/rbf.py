import numpy as np
import scipy.linalg as scplinalg
import scipy.spatial as scpspatial

from ..utils import to_unit_box
from .kernels import CubicKernel, Kernel
from .surrogate import Surrogate
from .tails import LinearTail, Tail


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
    :param lb: Lower variable bounds
    :type lb: numpy.array
    :param ub: Upper variable bounds
    :type ub: numpy.array
    :param output_transformation: Transformation applied to values before fitting
    :type output_transformation: Callable
    :param kernel: RBF kernel object
    :type kernel: Kernel
    :param tail: RBF polynomial tail object
    :type tail: Tail
    :param eta: Regularization parameter
    :type eta: float

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar output_transformation: Transformation to apply to function values before fitting
    :ivar num_pts: Number of points in surrogate model
    :ivar X: Point incorporated in surrogate model (num_pts x dim)
    :ivar fX: Function values in surrogate model (num_pts x 1)
    :ivar updated: True if model is up-to-date (no refit needed)
    :ivar kernel: RBF kernel
    :ivar tail: RBF tail
    :ivar eta: Regularization parameter
    """

    def __init__(self, dim, lb, ub, output_transformation=None, kernel=None, tail=None, eta=1e-6):
        super().__init__(dim=dim, lb=lb, ub=ub, output_transformation=output_transformation)

        if kernel is None or tail is None:
            kernel = CubicKernel()
            tail = LinearTail(dim)
        assert isinstance(kernel, Kernel) and isinstance(tail, Tail)

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

                X = self._X[0:n, :]
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
                    self.piv[i], self.piv[piv[i]] = self.piv[piv[i]], self.piv[i]

            else:  # Extend LU factorization
                k = self.c.shape[0] - ntail
                numnew = n - k
                kact = ntail + k

                X = self._X[:n, :]
                XX = self._X[k:n, :]
                D = scpspatial.distance.cdist(X, XX)
                Pnew = np.vstack((self.tail.eval(XX).T, self.kernel.eval(D[:k, :])))
                Phinew = self.kernel.eval(D[k:, :]) + self.eta * np.eye(numnew)

                L21 = np.zeros((kact, numnew))
                U12 = np.zeros((kact, numnew))
                for i in range(numnew):  # TODO: Can we use level-3 BLAS?
                    L21[:, i] = scplinalg.solve_triangular(a=self.U, b=Pnew[:kact, i], lower=False, trans="T")
                    U12[:, i] = scplinalg.solve_triangular(a=self.L, b=Pnew[self.piv[:kact], i], lower=True, trans="N")
                L21 = L21.T
                try:  # Compute Cholesky factorization of the Schur complement
                    C = scplinalg.cholesky(a=Phinew - np.dot(L21, U12), lower=True)
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
            fX = self.output_transformation(self.fX.copy())
            rhs = np.vstack((np.zeros((ntail, 1)), fX))
            self.c = scplinalg.solve_triangular(a=self.L, b=rhs[self.piv], lower=True)
            self.c = scplinalg.solve_triangular(a=self.U, b=self.c, lower=False)
            self.updated = True

    def predict(self, xx):
        """Evaluate the RBF interpolant at the points xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.ndarray

        :return: Prediction of size num_pts x 1
        :rtype: numpy.ndarray
        """
        self._fit()
        xx = to_unit_box(np.atleast_2d(xx), self.lb, self.ub)
        ds = scpspatial.distance.cdist(xx, self._X)
        ntail = self.ntail
        return np.dot(self.kernel.eval(ds), self.c[ntail : ntail + self.num_pts]) + np.dot(
            self.tail.eval(xx), self.c[:ntail]
        )

    def predict_deriv(self, xx):
        """Evaluate the derivative of the RBF interpolant at a point xx

        :param xx: Prediction points, must be of size num_pts x dim or (dim, )
        :type xx: numpy.array

        :return: Derivative of the RBF interpolant at xx
        :rtype: numpy.array
        """
        self._fit()
        xx = to_unit_box(np.atleast_2d(xx), self.lb, self.ub)
        if xx.shape[1] != self.dim:
            raise ValueError("Input has incorrect number of dimensions")
        ds = scpspatial.distance.cdist(self._X, xx)
        ds[ds < np.finfo(float).eps] = np.finfo(float).eps  # Avoid 0*inf

        dfxx = np.zeros((xx.shape[0], self.dim))
        for i in range(xx.shape[0]):
            x = np.atleast_2d(xx[i, :])
            ntail = self.ntail
            dpx = self.tail.deriv(x)
            dfx = np.dot(dpx, self.c[:ntail]).transpose()
            dsx = -(self._X.copy())
            dsx += x
            dss = np.atleast_2d(ds[:, i]).T
            dsx *= np.multiply(self.kernel.deriv(dss), self.c[ntail:]) / dss
            dfx += np.sum(dsx, 0)
            dfxx[i, :] = dfx
        return dfxx / (self.ub - self.lb)
