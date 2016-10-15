import numpy as np
import numpy.linalg as la
import scipy.spatial as scpspatial
import scipy.linalg as scplinalg

""" Kernels """


class CubicKernel(object):

    def order(self):
        return 2

    def phiZero(self):
        return 0

    def eval(self, dists):
        return np.multiply(dists, np.multiply(dists, dists))

    def deriv(self, dists):
        return 3 * np.multiply(dists, dists)


class TPSKernel(object):

    def order(self):
        return 2

    def phiZero(self):
        return 0

    def eval(self, dists):
        return np.multiply(np.multiply(dists, dists), np.log(dists + np.finfo(float).tiny))

    def deriv(self, dists):
        return np.multiply(dists, 1 + 2 * np.log(dists + np.finfo(float).tiny))


class LinearKernel(object):

    def order(self):
        return 1

    def phiZero(self):
        return 0

    def eval(self, dists):
        return dists

    def deriv(self, dists):
        return np.ones((dists.shape[0], dists.shape[1]))

""" TAILS """


class LinearTail(object):

    def degree(self):
        return 1

    def dimTail(self, dim):
        return 1 + dim

    # Each row is the basis evaluated at X[i, :]
    def eval(self, X):
        if len(X.shape) == 1:
            X = np.atleast_2d(X)
        return np.hstack((np.ones((X.shape[0], 1)), X))

    # Evaluate the derivative at one point
    # Row i is the gradient wrt x_i
    def deriv(self, x):
        return np.hstack((np.zeros((len(x), 1)), np.eye((len(x)))))


class ConstantTail(object):

    def degree(self):
        return 0

    def dimTail(self, dim):
        return 1

    # Each row is the basis evaluated at X[i, :]
    def eval(self, X):
        if len(X.shape) == 1:
            X = np.atleast_2d(X)
        return np.ones((X.shape[0], 1))

    # Evaluate the derivative at one point
    # Row i is the gradient wrt x_i
    def deriv(self, x):
        return np.ones((len(x), 1))


""" RBF Interpolant """


class RBFInterpolant(object):

    def __init__(self, kernel=None, tail=None, maxp=500, dim=None, eta=1e-8):
        if kernel is None or tail is None:
            kernel = CubicKernel
            tail = LinearTail

        self.maxp = maxp
        self.nump = 0
        self.kernel = kernel()
        self.tail = tail()
        self.ntail = self.tail.dimTail(dim)
        self.A = None
        self.LU = None
        self.piv = None
        self.c = None
        self.dim = dim
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
        """Re-set the interpolation."""
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
        :param ntail: Number of tail functions
        """
        maxp = self.maxp
        self.dim = dim
        self.ntail = ntail
        self.x = np.zeros((maxp, dim))
        self.fx = np.zeros((maxp, 1))
        self.rhs = np.zeros((maxp+ntail, 1))
        self.A = np.zeros((maxp+ntail, maxp+ntail))

    def _realloc(self, dim, ntail, extra=1):
        """Expand allocation to accommodate more points (if needed)

        :param dim: Number of dimensions
        :param ntail: Number of tail functions
        :param extra: Number of additional points to accommodate
        """
        if self.nump == 0:
            self._alloc(dim, ntail)
        elif self.nump + extra > self.maxp:
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

        self._realloc(self.dim, self.ntail)

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

    def transform_fx(self, fx, d=None):
        """Replace f with transformed function values for the fitting

        :param fx: Transformed function value
        """
        self.rhs[self.ntail:self.ntail+self.nump] = fx
        self.LU = None
        self.piv = None
        self.c = None

    def eval(self, x, ds=None):
        """Evaluate the rbf interpolant at the point x

        :param xx: Point where to evaluate
        :return: Value of the rbf interpolant at x
        """

        px = self.tail.eval(x)
        ntail = self.ntail
        c = self.coeffs()
        if ds is None:
            ds = scpspatial.distance.cdist(np.atleast_2d(x), self.x[:self.nump, :])
        fx = np.dot(px, c[:ntail]) + np.dot(self.kernel.eval(ds), c[ntail:ntail+self.nump])
        return fx[0][0]

    def evals(self, x, ds=None):
        """Evaluate the rbf interpolant at the points x

        :param xx: Points where to evaluate
        :return: Values of the rbf interpolant at x
        """
        ntail = self.ntail
        c = np.asmatrix(self.coeffs())
        if ds is None:
            ds = scpspatial.distance.cdist(x, self.x[:self.nump, :])
        fx = self.kernel.eval(ds)*c[ntail:ntail+self.nump] + self.tail.eval(x)*c[:ntail]
        return fx

    def deriv(self, x):
        """Evaluate the derivative of the rbf interpolant at x

        :param x: Data point
        :return: Derivative of the rbf interpolant at x (of size 1 x dim)
        """

        if len(x.shape) == 1:
            x = np.atleast_2d(x)  # Make x 1-by-dim
        ntail = self.ntail
        dpx = self.tail.deriv(x.transpose())
        c = self.coeffs()
        dfx = np.dot(dpx, c[:ntail]).transpose()
        ds = scpspatial.distance.cdist(self.x[:self.nump, :], x)
        ds[ds < 1e-10] = 1e-10  # Better safe than sorry
        dsx = - self.x[:self.nump, :]
        dsx += x
        dsx *= (np.multiply(self.kernel.deriv(ds), c[ntail:]) / ds)
        dfx += np.sum(dsx, 0)

        return dfx
