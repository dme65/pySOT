"""
.. module:: auxiliary_problems
   :synopsis: Ways of finding the next point to evaluate in the adaptive phase

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,
                David Bindel <bindel@cornell.edu>

:Module: auxiliary_problems
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
    fvals = surrogate.predict(cand)
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


def ei_merit(X, surrogate, fX, XX=None, dtol=0):
    """Compute the expected improvement"""
    mu, sig = surrogate.predict(X), surrogate.predict_std(X)
    gamma = (np.min(fX) - mu) / sig
    beta = gamma * norm.cdf(gamma) + norm.pdf(gamma)
    ei = sig * beta
    
    if dtol > 0:
        dists = scpspatial.distance.cdist(X, XX)
        dmerit = np.amin(np.asmatrix(dists), axis=1)
        ei[dmerit < dtol] = 0.0

    return ei


def expected_improvement_ga(
    num_pts, opt_prob, surrogate, X, fX,
    Xpend=None, dtol=1e-3, ei_tol=1e-6):
    """Just use a GA for now."""
    
    if Xpend is None:  # cdist can't handle None arguments
        Xpend = np.empty([0, opt_prob.dim])
    XX = np.vstack((X, Xpend))

    new_points = np.zeros((num_pts, opt_prob.dim))
    for i in range(num_pts):
        def obj(Y):
            ei = ei_merit(X=Y, surrogate=surrogate, fX=fX, XX=XX, dtol=dtol)
            return -ei  # Remember that we are minimizing!!!

        ga = GA(
            function=obj, dim=opt_prob.dim, lb=opt_prob.lb, 
            ub=opt_prob.ub, int_var=opt_prob.int_var, 
            pop_size=max([2*opt_prob.dim, 100]), num_gen=100)
        x_best, f_min = ga.optimize()

        ei_max = -f_min
        if ei_max < ei_tol:
            return None  # Give up

        new_points[i, :] = x_best
        XX = np.vstack((XX, x_best))

    return new_points


def expected_improvement_uniform(
    num_pts, opt_prob, surrogate, X, fX,
    Xpend=None, dtol=1e-3, ei_tol=1e-6, num_cand=None):
    """Pick the best from a set of uniform points."""
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if Xpend is None:  # cdist can't handle None arguments
        Xpend = np.empty([0, opt_prob.dim])
    XX = np.vstack((X, Xpend))

    new_points = np.zeros((num_pts, opt_prob.dim))
    for i in range(num_pts):

        # Fix default values   
        if num_cand is None:
            num_cand = 100*opt_prob.dim

        # Generate uniformly random candidate points
        cand = np.random.uniform(
            opt_prob.lb, opt_prob.ub, (num_cand, opt_prob.dim))
        ei = ei_merit(X=cand, surrogate=surrogate, fX=fX, XX=XX, dtol=dtol)

        jj = np.argmax(ei)
        ei_max = ei[jj]

        if ei_max < ei_tol:
            return None  # Give up
        new_points[i, :] = cand[jj, :].copy()

        XX = np.vstack((XX, cand[jj, :].copy()))

    return new_points