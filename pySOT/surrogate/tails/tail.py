from abc import ABC, abstractmethod


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
