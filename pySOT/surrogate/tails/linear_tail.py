import numpy as np

from .tail import Tail


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
