"""
.. module:: experimental_design
  :synopsis: Methods for generating an experimental design.

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,
                 Yi Shen <ys623@cornell.edu>

:Module: experimental_design
:Author: David Eriksson <dme65@cornell.edu>
        Yi Shen <ys623@cornell.edu>
"""

import abc

import numpy as np
import six
from numpy.linalg import matrix_rank as rank
from scipy.spatial.distance import cdist

from ..utils import from_unit_box, round_vars


@six.add_metaclass(abc.ABCMeta)
class ExperimentalDesign(object):
    """Base class for experimental designs.

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in the experimental design
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):  # pragma: no cover
        self.dim = None
        self.num_pts = None

    @abc.abstractmethod
    def generate_points(self, lb=None, ub=None, int_var=None):  # pragma: no cover
        pass


def _expdes_dist(gen, iterations, lb, ub, int_var):
    """Helper method for picking the best experimental design.

    We generate iterations designs and picks the one the maximizes the
    minimum distance between points. This isn't a perfect criterion, but
    it will help avoid rank-defficient designs such as y=x.

    :param lb: Lower bounds
    :type lb: numpy.array
    :param ub: Upper bounds
    :type ub: numpy.array
    :param int_var: Indices of integer variables.
    :type int_var: numpy.array

    :return: Experimental design of size num_pts x dim
    :rtype: numpy.ndarray
    """

    X = None
    best_score = 0
    for _ in range(iterations):
        cand = gen()  # Generate a new design
        if all([x is not None for x in [lb, ub]]):  # Map and round
            cand = round_vars(from_unit_box(cand, lb, ub), int_var, lb, ub)

        dists = cdist(cand, cand)
        np.fill_diagonal(dists, np.inf)  # Since these are zero
        score = dists.min().min()

        if score > best_score and rank(cand) == cand.shape[1]:
            best_score = score
            X = cand.copy()

    if X is None:
        raise ValueError("No valid design found, increase num_pts?")
    return X
