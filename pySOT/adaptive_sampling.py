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


class AuxiliaryProblem(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def make_points(self, X, fX, surrogate, Xpend, sampling_radius, npts):  # pragma: no cover
        pass


class CandidateSRBF(AuxiliaryProblem):
    """An implementation of Stochastic RBF

    This is an implementation of the candidate points method that is
    proposed in the first SRBF paper. Candidate points are generated
    by making normally distributed perturbations with standard
    deviation sigma around the best solution. The candidate point that
    minimizes a specified merit function is selected as the next
    point to evaluate.

    :param data: Optimization problem object
    :type data: Object
    :param num_cand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type num_cand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]
    """

    def __init__(self, opt_prob, num_cand=None, weights=None):
        self.opt_prob = opt_prob
        self.xrange = self.opt_prob.ub - self.opt_prob.lb
        self.dtol = 1e-3 * math.sqrt(opt_prob.dim)
        self.weights = weights
        if self.weights is None:
            self.weights = [0.3, 0.5, 0.8, 0.95]
        self.next_weight = 0

        self.num_cand = num_cand
        if self.num_cand is None:
            self.num_cand = min([5000, 100*opt_prob.dim])

        # Check that the inputs make sense
        if not(isinstance(self.num_cand, int) and self.num_cand > 0):
            raise ValueError("The number of candidate points has to be a positive integer")
        if not((isinstance(self.weights, np.ndarray) or isinstance(self.weights, list))
               and max(self.weights) <= 1 and min(self.weights) >= 0):
            raise ValueError("Incorrect weights")

    def __generate_cand__(self, scalefactors, xbest, subset):
        cand = np.multiply(np.ones((self.num_cand,  self.opt_prob.dim)), xbest)
        for i in subset:
            lower, upper, sigma = self.opt_prob.lb[i], self.opt_prob.ub[i], scalefactors[i]
            cand[:, i] = stats.truncnorm.rvs(
                a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma,
                loc=xbest[i], scale=sigma, size=self.num_cand)
        return cand

    def make_points(self, npts, surrogate, X, fX, Xpend=None, sampling_radius=0.2, subset=None):
        """Proposes npts new points to evaluate

        :param npts: Number of points to select
        :type npts: int
        :param xbest: Best solution found so far
        :type xbest: numpy.array
        :param sigma: Current sampling radius w.r.t the unit box
        :type sigma: float
        :param subset: Coordinates to perturb, the others are fixed
        :type subset: numpy.array

        :return: Points selected for evaluation, of size npts x dim
        :rtype: numpy.array
        """

        if Xpend is None:
            Xpend = np.empty([0, self.opt_prob.dim])
        ind = np.argmin(fX)
        xbest = np.copy(X[ind, :]).ravel()

        if subset is None: 
            subset = np.arange(0, self.opt_prob.dim)
        scalefactors = sampling_radius * self.xrange

        # Make sure the scale factors are correct for integer variables (at least 1)
        ind = np.intersect1d(self.opt_prob.int_var, subset)
        if len(ind) > 0:
            scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

        # Generate candidate points
        cand = self.__generate_cand__(scalefactors, xbest, subset)
        
        # Distance
        dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
        dmerit = np.amin(np.asmatrix(dists), axis=1)

        # Values
        fvals = surrogate.eval(cand)
        fvals = unit_rescale(fvals)

        # Pick candidate points
        new_points = np.ones((npts,  self.opt_prob.dim))
        for i in range(npts):
            ii = self.next_weight
            weight = self.weights[(ii + len(self.weights)) % len(self.weights)]
            merit = weight*fvals + (1.0-weight)*(1.0-unit_rescale(np.copy(dmerit)))

            merit[dmerit < self.dtol] = np.inf
            jj = np.argmin(merit)
            fvals[jj] = np.inf
            new_points[i, :] = cand[jj, :]

            # Update distances and weights
            ds = scpspatial.distance.cdist(cand, np.atleast_2d(new_points[i, :]))
            dmerit = np.minimum(dmerit, ds)
            self.next_weight += 1

        return new_points


class CandidateUniform(CandidateSRBF):
    """Create Candidate points by sampling uniformly in the domain
    :param data: Optimization problem object
    :type data: Object
    :param num_cand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type num_cand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array
    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]
    """

    def make_points(self, npts, surrogate, X, fX, Xpend, sampling_radius=0.2, subset=None):
        return CandidateSRBF.make_points(self, npts=npts, surrogate=surrogate, X=X, fX=fX, 
            Xpend=Xpend, sampling_radius=sampling_radius, subset=subset)

    def __generate_cand__(self, scalefactors, xbest, subset):
        cand = np.multiply(np.ones((self.num_cand, self.opt_prob.dim)), xbest)
        cand[:, subset] = np.random.uniform(
            self.opt_prob.lb[subset], self.opt_prob.ub[subset], 
            (self.num_cand, len(subset)))
        return cand

class CandidateDYCORS(CandidateSRBF):
    """An implementation of the DYCORS method
    The DYCORS method only perturbs a subset of the dimensions when
    perturbing the best solution. The probability for a dimension
    to be perturbed decreases after each evaluation and is capped
    in order to guarantee global convergence.
    :param data: Optimization problem object
    :type data: Object
    :param num_cand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type num_cand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array
    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]
    """

    def __init__(self, opt_prob, max_evals, num_cand=None, weights=None):
        CandidateSRBF.__init__(self, opt_prob=opt_prob, num_cand=num_cand, weights=weights)
        self.min_prob = np.min([1.0, 1.0/opt_prob.dim])
        self.n0 = None
        self.max_evals = max_evals

        if opt_prob.dim <= 1:
            raise ValueError("You can't use DYCORS on a 1d problem")

        def prob_fun(num_evals, budget):
            if budget < 2: return 0.0
            return min([20.0/opt_prob.dim, 1.0]) * (1.0 - (np.log(num_evals + 1.0) / np.log(budget)))
        self.prob_fun = prob_fun

    def make_points(self, npts, surrogate, X, fX, Xpend, sampling_radius=0.2, subset=None):
        if self.n0 is None: # Initialize n0
            self.n0 = len(X) + len(Xpend)
        self.num_evals = len(X) + len(Xpend)
        self.dds_prob = self.prob_fun(self.num_evals - self.n0, self.max_evals - self.n0)
        self.dds_prob = np.max([self.min_prob, self.dds_prob])

        return CandidateSRBF.make_points(self, npts=npts, surrogate=surrogate, X=X, fX=fX, 
            Xpend=Xpend, sampling_radius=sampling_radius, subset=subset)

    def __generate_cand__(self, sigmas, xbest, subset):  
        if len(subset) == 1: # Fix when nlen is 1
            ar = np.ones((self.num_cand, 1))
        else:
            ar = (np.random.rand(self.num_cand, len(subset)) < self.dds_prob)
            ind = np.where(np.sum(ar, axis=1) == 0)[0]
            ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

        cand = np.multiply(np.ones((self.num_cand, self.opt_prob.dim)), xbest)
        for i in subset:
            lower, upper, sigma = self.opt_prob.lb[i], self.opt_prob.ub[i], sigmas[i]
            ind = np.where(ar[:, i] == 1)[0]
            cand[ind, subset[i]] = stats.truncnorm.rvs(
                a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma,
                loc=xbest[i], scale=sigma, size=len(ind))
        return cand


class MultiSampling(AuxiliaryProblem):
    """Maintains a list of adaptive sampling methods"""
    def __init__(self, opt_prob, sampling_list, cycle=None):
        if cycle is None:
            cycle = range(len(sampling_list))
        if (not all(isinstance(i, int) for i in cycle)) or \
                np.min(cycle) < 0 or \
                np.max(cycle) > len(sampling_list)-1:
            raise ValueError("Incorrect cycle!!")
        self.opt_prob = opt_prob
        self.sampling_list = sampling_list
        self.cycle = cycle
        self.next = 0

    def make_points(self, npts, surrogate, X, fX, Xpend, sampling_radius=0.2, subset=None):
        """Proposes npts new points to evaluate"""
        new_points = np.zeros((npts, self.opt_prob.dim))
        Xpend_new = np.copy(Xpend)

        # Now generate the points from one strategy at the time
        for i in range(npts):
            ind = self.cycle[(self.next + len(self.cycle)) % len(self.cycle)]
            new_points[i, :] = self.sampling_list[ind].make_points(
                npts=1, X=X, fX=fX, surrogate=surrogate, Xpend=Xpend_new, 
                sampling_radius=sampling_radius, subset=subset)
            Xpend_new = np.vstack((Xpend_new, new_points[i, :].copy()))
            self.next += 1

        return new_points