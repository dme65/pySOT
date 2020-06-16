import numpy as np

from .experimental_design import ExperimentalDesign, _expdes_dist


class SymmetricLatinHypercube(ExperimentalDesign):
    """Symmetric Latin hypercube experimental design.

    :param dim: Number of dimensions
    :type dim: int
    :param num_pts: Number of desired sampling points
    :type num_pts: int
    :param iterations: Number of designs to generate and pick the best from
    :type iterations: int

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in the experimental design
    :ivar iterations: Number of points in the experimental design
    """

    def __init__(self, dim, num_pts, iterations=1000):
        self.dim = dim
        self.num_pts = num_pts
        self.iterations = iterations

    def generate_points(self, lb=None, ub=None, int_var=None):
        """Generate a new experimental design.

        You can specify lb, ub, int_var to have the design mapped to a
        specific domain. These inputs are ignored if one of lb
        or ub is None. The design is generated in [0, 1]^d in this case.

        :param lb: Lower bounds
        :type lb: numpy.array
        :param ub: Upper bounds
        :type ub: numpy.array
        :param int_var: Indices of integer variables. If None, [], or
                        np.array([]) we assume all variables are continuous.
        :type int_var: numpy.array

        :return: Experimental design of size num_pts x dim
        :rtype: numpy.ndarray
        """
        if int_var is None or len(int_var) == 0:
            int_var = np.array([])

        def wrapper():
            return self._slhd()

        return _expdes_dist(wrapper, self.iterations, lb, ub, int_var)

    def _slhd(self):
        """Generate a symmetric Latin hypercube design in the unit hypercube.

        :return: Symmetric Latin hypercube design in the unit hypercube
            of size num_pts x dim
        :rtype: numpy.ndarray
        """
        # Generate a one-dimensional array based on sample number
        points = np.zeros([self.num_pts, self.dim])
        points[:, 0] = np.arange(1, self.num_pts + 1)

        # Get the last index of the row in the top half of the hypercube
        middleind = self.num_pts // 2

        # special manipulation if odd number of rows
        if self.num_pts % 2 == 1:
            points[middleind, :] = middleind + 1

        # Generate the top half of the hypercube matrix
        for j in range(1, self.dim):
            for i in range(middleind):
                if np.random.random() < 0.5:
                    points[i, j] = self.num_pts - i
                else:
                    points[i, j] = i + 1
            np.random.shuffle(points[:middleind, j])

        # Generate the bottom half of the hypercube matrix
        for i in range(middleind, self.num_pts):
            points[i, :] = self.num_pts + 1 - points[self.num_pts - 1 - i, :]

        return (points - 1) / (self.num_pts - 1)  # Map to [0, 1]^d
