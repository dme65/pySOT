"""
.. module:: adaptive_sampling
   :synopsis: Ways of finding the next point to evaluate in the adaptive phase

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,
                David Bindel <bindel@cornell.edu>

:Module: adaptive_sampling
:Author: David Eriksson <dme65@cornell.edu>,
        David Bindel <bindel@cornell.edu>

"""

import abc
import math
import numpy as np
import scipy.spatial as scpspatial
import scipy.stats as stats
from pySOT.utils import GeneticAlgorithm as GA
from pySOT.utils import unit_rescale
from scipy.optimize import minimize
from scipy.stats import norm


def weighted_distance_merit(
    num_pts, surrogate, X, fX, cand, weights, Xpend=None, dtol=1e-3):
    
    # Distance
    dim = X.shape[1]
    if Xpend is None:  # cdist can't handle None arguments
        Xpend = np.empty([0, dim])
    dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
    dmerit = np.amin(np.asmatrix(dists), axis=1)

    # Values
    fvals = surrogate.eval(cand)
    fvals = unit_rescale(fvals)

    # Pick candidate points
    new_points = np.ones((num_pts,  dim))
    for i in range(num_pts):
        w = weights[i]
        merit = w*fvals + (1.0-w)*(1.0 - unit_rescale(np.copy(dmerit)))

        merit[dmerit < dtol] = np.inf
        jj = np.argmin(merit)
        fvals[jj] = np.inf
        new_points[i, :] = cand[jj, :].copy()

        # Update distances and weights
        ds = scpspatial.distance.cdist(
            cand, np.atleast_2d(new_points[i, :]))
        dmerit = np.minimum(dmerit, ds)

    return new_points


def candidate_srbf(
    num_pts, opt_prob, surrogate, X, fX, weights, Xpend=None,
    sampling_radius=0.2, subset=None, dtol=1e-3, num_cand=None):

    # Find best solution
    xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None: 
        subset = np.arange(0, opt_prob.dim)
    
    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    # Generate candidate points
    cand = np.multiply(np.ones((num_cand,  opt_prob.dim)), xbest)
    for i in subset:
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        cand[:, i] = stats.truncnorm.rvs(
            a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma,
            loc=xbest[i], scale=sigma, size=num_cand)

    # Make selections
    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX, 
        Xpend=Xpend, cand=cand, dtol=dtol, weights=weights)


def candidate_dycors(
    num_pts, opt_prob, surrogate, X, fX, weights, prob_perturb, 
    Xpend=None, sampling_radius=0.2, subset=None, dtol=1e-3, 
    num_cand=None):

     # Find best solution
    xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None: 
        subset = np.arange(0, opt_prob.dim)
    
    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    # Generate candidate points
    if len(subset) == 1: # Fix when nlen is 1
        ar = np.ones((num_cand, 1))
    else:
        ar = (np.random.rand(num_cand, len(subset)) < prob_perturb)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    for i in subset:
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        ind = np.where(ar[:, i] == 1)[0]
        cand[ind, subset[i]] = stats.truncnorm.rvs(
            a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma,
            loc=xbest[i], scale=sigma, size=len(ind))

    # Make selections
    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX, 
        Xpend=Xpend, cand=cand, dtol=dtol, weights=weights)


def candidate_uniform(
    num_pts, opt_prob, surrogate, X, fX, weights, Xpend=None, 
    subset=None, dtol=1e-3, num_cand=None):

     # Find best solution
    xbest = np.copy(X[np.argmin(fX), :]).ravel()

     # Fix default values   
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None: 
        subset = np.arange(0, opt_prob.dim)

    # Generate uniformly random candidate points
    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    cand[:, subset] = np.random.uniform(
        opt_prob.lb[subset], opt_prob.ub[subset], 
        (num_cand, len(subset)))

    # Make selections
    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX, 
        Xpend=Xpend, cand=cand, dtol=dtol, weights=weights)