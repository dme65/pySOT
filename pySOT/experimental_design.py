#!/usr/bin/env python
"""
..module:: experimental_design
  :synopsis: Methods for generating an experimental design.
..moduleauthor:: David Eriksson <dme65@cornell.edu> 
                 Yi Shen <ys623@cornell.edu>
"""

import numpy as np
import pyDOE as pydoe

try:
    import pyKriging.samplingplan as pykrig_sampling
except ImportError:
    print("pyKriging not installed -- FullFactorial is unavailable")


class ExperimentalDesign(object):
    """Manage list of experimental designs.

    Attributes:
        strategies: list of experimental design objects
    """

    def __init__(self, strategies):
        self.strategies = strategies


class LatinHypercube(object):
    def __init__(self, dim, npts, criterion='c'):
        """
        Args:
           dim: dimensionality of the input
           npts: number of desired sampling points for the experiment
           criterion: a string that tells lhs how to sample the
               points (default: None which simply randomizes the points
               within the intervals):
               - "center" or "c": center the points within the sampling
                 intervals
               - "maximin" or "m": maximize the minimum distance
                 between points, but place the point in a randomized
                 location within its interval
               - "centermaximin" or "cm": same as "maximin", but
                 centered within the intervals
               - "correlation" or "corr": minimize the maximum
                 correlation coefficient
        """
        self.dim = dim
        self.npts = npts
        self.criterion = criterion

    def generate_points(self):
        """Generate a matrix with the initial sample points,
        scaled to the unit"""
        return pydoe.lhs(self.dim, self.npts, self.criterion)

    def num_points(self):
        return self.npts


class SymmetricLatinHypercube(object):
    def __init__(self, dim, npts):
        """
        Args:
            dim: dimensionality of the input
            npts: number of desired sampling points for the experiment
        """
        self.dim = dim
        self.npts = npts

    def slhd(self):
        """Generate matrix of sample points in the unit box"""

        # Generate a one-dimensional array based on sample number
        points = np.zeros([self.npts, self.dim])
        points[:, 0] = np.arange(1, self.npts+1)

        # Get the last index of the row in the top half of the hypercube
        middleind = self.npts//2

        # special manipulation if odd number of rows
        if self.npts % 2 == 1:
            points[middleind, :] = middleind + 1

        # Generate the top half of the hypercube matrix
        for j in range(1, self.dim):
            for i in range(middleind):
                if np.random.random() < 0.5:
                    points[i, j] = self.npts-i
                else:
                    points[i, j] = i + 1
            np.random.shuffle(points[:middleind, j])

        # Generate the bottom half of the hypercube matrix
        for i in range(middleind, self.npts):
            points[i, :] = self.npts+1 - points[self.npts-1-i, :]

        return points/self.npts

    def generate_points(self):
        """Generate sample points using SLHD with no rank deficiency"""
        rank_pmat = 0
        pmat = np.ones((self.npts, self.dim+1))
        while rank_pmat != self.dim + 1:
            xsample = self.slhd()
            pmat[:, 1:] = xsample
            rank_pmat = np.linalg.matrix_rank(pmat)
        return xsample

    def num_points(self):
        return self.npts



# ========================= For Test =======================

def main():
    print("========================= LHD =======================")
    lhs = LatinHypercube(4, 10, criterion='c')
    print(lhs.generate_points())
    print(lhs.num_points())

    print("\n========================= SLHD =======================")
    slhd = SymmetricLatinHypercube(3, 10)
    print(slhd.generate_points())
    print(slhd.num_points())

if __name__ == "__main__":
    main()
