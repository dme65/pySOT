"""
.. module:: merit_functions
   :synopsis: Merit functions for the adaptive sampling

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,
                David Bindel <bindel@cornell.edu>

:Module: merit_functions
:Author: David Eriksson <dme65@cornell.edu>,
        David Bindel <bindel@cornell.edu>

"""

from pySOT.utils import *
import scipy.spatial as scp


def candidate_merit_weighted_distance(cand, npts=1):
    """Weighted distance merit function for the candidate points based methods

    :param cand: Candidate point object
    :type cand: Object
    :param npts: Number of points selected for evaluation
    :type npts: int

    :return: Points selected for evaluation, of size npts x dim
    :rtype: numpy.array
    """

    new_points = np.ones((npts,  cand.data.dim))

    for i in range(npts):
        ii = cand.next_weight
        weight = cand.weights[(ii+len(cand.weights)) % len(cand.weights)]
        merit = weight*cand.fhvals + \
            (1-weight)*(1.0 - unit_rescale(cand.dmerit))

        merit[cand.dmerit < cand.dtol] = np.inf
        jj = np.argmin(merit)
        cand.fhvals[jj] = np.inf
        new_points[i, :] = cand.xcand[jj, :]

        # Update distances and weights
        ds = scp.distance.cdist(cand.xcand, np.atleast_2d(new_points[i, :]))
        cand.dmerit = np.minimum(cand.dmerit, ds)
        cand.next_weight += 1

    return new_points
