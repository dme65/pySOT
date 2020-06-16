from abc import ABC, abstractmethod


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
