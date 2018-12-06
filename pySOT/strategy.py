"""
.. module:: strategy
   :synopsis: Surrogate optimization strategies

.. moduleauthor:: David Eriksson <dme65@cornell.edu>
                David Bindel <bindel@cornell.edu>,

:Module: strategy
:Author: David Eriksson <dme65@cornell.edu>
        David Bindel <bindel@cornell.edu>,
"""

import abc
import dill
import logging
import math
import numpy as np
import os
import time
from poap.strategy import BaseStrategy, Proposal, RetryStrategy

from pySOT.auxiliary_problems import candidate_srbf, candidate_dycors
from pySOT.auxiliary_problems import expected_improvement_ga, \
    lower_confidence_bound_ga
from pySOT.experimental_design import ExperimentalDesign
from pySOT.optimization_problems import OptimizationProblem
from pySOT.surrogate import Surrogate, GPRegressor
from pySOT.utils import from_unit_box, round_vars, check_opt_prob

# Get module-level logger
logger = logging.getLogger(__name__)


class RandomSampling(BaseStrategy):
    """Random sampling strategy.

    We generate and evaluate a fixed number of points using all resources.
    The optimization problem must implement OptimizationProblem and max_evals
    must be a positive integer.

    :param max_evals: Evaluation budget
    :type max_evals: int
    :param opt_prob: Optimization problem
    :type opt_prob: OptimizationProblem
    """
    def __init__(self, max_evals, opt_prob):
        check_opt_prob(opt_prob)
        if not isinstance(max_evals, int) and max_evals > 0:
            raise ValueError("max_evals must be an integer >= exp_des.num_pts")

        self.opt_prob = opt_prob
        self.max_evals = max_evals
        self.retry = RetryStrategy()
        for _ in range(max_evals):  # Generate the random points
            x = np.random.uniform(low=opt_prob.lb, high=opt_prob.ub)
            proposal = self.propose_eval(x)
            self.retry.rput(proposal)

    def propose_action(self):
        """Propose an action based on outstanding points."""
        if not self.retry.empty():  # Propose next point
            return self.retry.get()
        elif self.retry.num_eval_outstanding == 0:  # Budget exhausted
            return self.propose_terminate()


class SurrogateBaseStrategy(BaseStrategy):
    __metaclass__ = abc.ABCMeta
    """Surrogate base strategy.

    This is a base strategy for surrogate optimization. This class is abstract
    and inheriting classes must implement generate_evals(self, num_pts)
    which proposes num_pts new evaluations. This strategy follows the general
    idea of surrogate optimization:

    1.  Generate an initial experimental design
    2.  Evaluate the points in the experimental design
    3.  Build a Surrogate model from the data
    4.  Repeat until stopping criterion met
    5.      Generate num_pts new points to evaluate
    6.      Evaluate the new point(s)
    7.      Update the surrogate model

    The optimization problem, experimental design, and surrogate model
    must implement an abstract class definition to make sure they are
    compatible with the framework. More information about required
    methods and attributes is explained in the pySOT documentation:
        https://pysot.readthedocs.io/

    We support function evaluations in serial, synchronous parallel, and
    asynchronous parallel. Serial evaluations can be achieved by using the
    SerialController in POAP and either run this strategy asynchronously
    or synchronously with batch_size equal to 1. The main difference between
    asynchronous and synchronous parallel is that we launch new function
    evaluations as soon as a worker becomes available when using asynchronous
    parallel, while we wait for the entire batch to finish when using
    synchronous parallel. It is important that the experimental design
    generates enough points to construct an initial surrogate model.

    The user can supply additional points to add to the experimental design
    using the extra_points and extra_vals inputs. This is of interest if
    a good starting point is known. The surrogate model is reset by default
    to make sure no old points are present in the surrogate model, but this
    behavior can be modified using the reset_surrogate argument.

    The strategy stops proposing new evaluations once the attribute
    self.terminate has been set to True or when the evaluation budget has been
    exceeded. We wait for pending evaluations to finish before the strategy
    terminates.

    :param max_evals: Evaluation budget
    :type max_evals: int
    :param opt_prob: Optimization problem object
    :type opt_prob: OptimizationProblem
    :param exp_design: Experimental design object
    :type exp_design: ExperimentalDesign
    :param surrogate: Surrogate object
    :type surrogate: Surrogate
    :param asynchronous: Whether or not to use asynchrony (True/False)
    :type asynchronous: bool
    :param batch_size: Size of batch (use 1 for serial, ignored if async)
    :type batch_size: int
    :param extra_points: Extra points to add to the experimental design
    :type extra_points: numpy.array of size n x dim
    :param extra_vals: Values for extra_points (np.nan/np.inf if unknown)
    :type extra_vals: numpy.array of size n x 1
    :param reset_surrogate: Whether or not to reset surrogate model
    :type reset_surrogate: bool
    """
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                 asynchronous=True, batch_size=None, extra_points=None,
                 extra_vals=None, reset_surrogate=True):
        self.asynchronous = asynchronous
        self.batch_size = batch_size

        # Save the objects
        self.opt_prob = opt_prob
        self.exp_design = exp_design
        if reset_surrogate:
            surrogate.reset()
        self.surrogate = surrogate

        # Sampler state
        self.proposal_counter = 0
        self.terminate = False
        self.accepted_count = 0
        self.rejected_count = 0

        # Initial design info
        self.extra_points = extra_points
        self.extra_vals = extra_vals
        self.batch_queue = []        # Unassigned points in initial experiment
        self.init_pending = 0        # Number of outstanding initial fevals
        self.phase = 1               # 1 for initial, 2 for adaptive

        # Budgeting state
        self.num_evals = 0             # Number of completed fevals
        self.max_evals = max_evals     # Remaining feval budget
        self.pending_evals = 0         # Number of outstanding fevals

        # Completed evaluations
        self.X = np.empty([0, opt_prob.dim])
        self.fX = np.empty([0, 1])
        self.Xpend = np.empty([0, opt_prob.dim])
        self.fevals = []

        # Check inputs (implemented by each strategy)
        self.check_input()

        # Start with first experimental design
        self.sample_initial()

    @abc.abstractmethod  # pragma: no cover
    def generate_evals(self, num_pts):
        pass

    def check_input(self):
        """Check the inputs to the optimization strategt. """
        if not isinstance(self.surrogate, Surrogate):
            raise ValueError("surrogate must implement Surrogate")
        if not isinstance(self.exp_design, ExperimentalDesign):
            raise ValueError("exp_design must implement ExperimentalDesign")
        check_opt_prob(self.opt_prob)
        if not self.asynchronous and self.batch_size is None:
            raise ValueError("You must specify batch size in synchronous mode "
                             "(use 1 for serial)")
        if not isinstance(self.max_evals, int) and self.max_evals > 0 and \
                self.max_evals >= self.exp_design.num_pts:
            raise ValueError("max_evals must be an integer >= exp_des.num_pts")

    def save(self, fname):
        """Save the state of the strategy.

        We do this in a 3-step procedure
            1) Save to temp file
            2) Move temp file to save file
            3) Remove temp file

        :param fname: Filename
        :type fname: string
        """
        temp_fname = "temp_" + fname
        with open(temp_fname, 'wb') as output:
            dill.dump(self, output, dill.HIGHEST_PROTOCOL)
        os.rename(temp_fname, fname)

    def resume(self):
        """Resume a terminated run."""
        if self.phase == 1:  # Put the points back in the queue in init phase
            self.init_pending = 0
            for x in self.Xpend:
                self.batch_queue.append(np.copy(x))

        # Remove everything that is pending
        self.pending_evals = 0
        self.Xpend = np.empty([0, self.opt_prob.dim])

    def log_completion(self, record):
        """Record a completed evaluation to the log.

        :param record: Record of the function evaluation
        :type record: EvalRecord
        """
        xstr = np.array_str(
            record.params[0], max_line_width=np.inf,
            precision=5, suppress_small=True)
        logger.info("{} {:.3e} @ {}".format(
            self.num_evals, record.value, xstr))

    def sample_initial(self):
        """Generate and queue an initial experimental design."""
        logger.info("=== Start ===")
        self.surrogate.reset()

        start_sample = self.exp_design.generate_points()
        assert start_sample.shape[1] == self.opt_prob.dim, \
            "Dimension mismatch between problem and experimental design"
        start_sample = from_unit_box(
            start_sample, self.opt_prob.lb, self.opt_prob.ub)
        start_sample = round_vars(
            start_sample, self.opt_prob.int_var,
            self.opt_prob.lb, self.opt_prob.ub)

        for j in range(self.exp_design.num_pts):
            self.batch_queue.append(start_sample[j, :])

        if self.extra_points is not None:
            for i in range(self.extra_points.shape[0]):
                if self.extra_vals is None or \
                        np.all(np.isnan(self.extra_vals[i])):  # Unknown value
                    self.batch_queue.append(self.extra_points[i, :])
                else:  # Known value, save point and add to surrogate model
                    x = np.copy(self.extra_points[i, :])
                    self.X = np.vstack((self.X, x))
                    self.fX = np.vstack((self.fX, self.extra_vals[i]))
                    self.surrogate.add_points(x, self.extra_vals[i])

    def propose_action(self):
        """Propose an action.

        NB: We allow workers to continue to the adaptive phase if
        the initial queue is empty. This implies that we need enough
        points in the experimental design for us to construct a
        surrogate.
        """
        if self.terminate:  # Check if termination has been triggered
            if self.pending_evals == 0:
                return Proposal('terminate')
        elif self.num_evals + self.pending_evals >= self.max_evals or \
                self.terminate:
            if self.pending_evals == 0:  # Only terminate if nothing is pending
                return Proposal('terminate')
        elif self.batch_queue:  # Propose point from the batch_queue
            if self.phase == 1:
                return self.init_proposal()
            else:
                return self.adapt_proposal()
        else:  # Make new proposal in the adaptive phase
            self.phase = 2
            if self.asynchronous:  # Always make proposal with asynchrony
                self.generate_evals(num_pts=1)
            elif self.pending_evals == 0:  # Make sure the entire batch is done
                self.generate_evals(num_pts=self.batch_size)

            if self.terminate:  # Check if termination has been triggered
                if self.pending_evals == 0:
                    return Proposal('terminate')

            # Launch evaluation (the others will be triggered later)
            return self.adapt_proposal()

    def make_proposal(self, x):
        """Create proposal and update counters and budgets."""
        proposal = Proposal('eval', x)
        self.pending_evals += 1
        self.Xpend = np.vstack((self.Xpend, np.copy(x)))
        return proposal

    def remove_pending(self, x):
        """Delete a pending point from self.Xpend."""
        idx = np.where((self.Xpend == x).all(axis=1))[0]
        self.Xpend = np.delete(self.Xpend, idx, axis=0)

    # == Processing in initial phase ==

    def init_proposal(self):
        """Propose a point from the initial experimental design."""
        proposal = self.make_proposal(self.batch_queue.pop())
        proposal.add_callback(self.on_initial_proposal)
        return proposal

    def on_initial_proposal(self, proposal):
        """Handle accept/reject of proposal from initial design."""
        if proposal.accepted:
            self.on_initial_accepted(proposal)
        else:
            self.on_initial_rejected(proposal)

    def on_initial_accepted(self, proposal):
        """Handle proposal accept from initial design."""
        self.accepted_count += 1
        proposal.record.add_callback(self.on_initial_update)

    def on_initial_rejected(self, proposal):
        """Handle proposal rejection from initial design."""
        self.rejected_count += 1
        self.pending_evals -= 1
        xx = proposal.args[0]
        self.batch_queue.append(xx)  # Add back to queue
        self.Xpend = np.vstack((self.Xpend, np.copy(xx)))
        self.remove_pending(xx)

    def on_initial_update(self, record):
        """Handle update of feval from initial design."""
        if record.status == 'completed':
            self.on_initial_completed(record)
        elif record.is_done:
            self.on_initial_aborted(record)

    def on_initial_completed(self, record):
        """Handle successful completion of feval from initial design."""
        self.num_evals += 1
        self.pending_evals -= 1

        xx, fx = np.copy(record.params[0]), record.value
        self.X = np.vstack((self.X, np.asmatrix(xx)))
        self.fX = np.vstack((self.fX, fx))

        self.surrogate.add_points(xx, fx)
        self.remove_pending(xx)

        self.log_completion(record)
        self.fevals.append(record)

    def on_initial_aborted(self, record):
        """Handle aborted feval from initial design."""
        self.pending_evals -= 1
        xx = record.params[0]
        self.batch_queue.append(xx)
        self.remove_pending(xx)
        self.fevals.append(record)

    # == Processing in adaptive phase ==

    def adapt_proposal(self):
        """Propose a point from the batch_queue."""
        if self.batch_queue:
            proposal = self.make_proposal(self.batch_queue.pop())
            proposal.add_callback(self.on_adapt_proposal)
            return proposal

    def on_adapt_proposal(self, proposal):
        """Handle accept/reject of proposal from sampling phase."""
        if proposal.accepted:
            self.on_adapt_accept(proposal)
        else:
            self.on_adapt_reject(proposal)

    def on_adapt_accept(self, proposal):
        """Handle accepted proposal from sampling phase."""
        self.accepted_count += 1
        proposal.record.add_callback(self.on_adapt_update)

    def on_adapt_reject(self, proposal):
        """Handle rejected proposal from sampling phase."""
        self.rejected_count += 1
        self.pending_evals -= 1
        xx = np.copy(proposal.args[0])
        self.remove_pending(xx)
        if not self.asynchronous:  # Add back to the queue in synchronous case
            self.batch_queue.append(xx)

    def on_adapt_update(self, record):
        """Handle update of feval from sampling phase."""
        if record.status == 'completed':
            self.on_adapt_completed(record)
        elif record.is_done:
            self.on_adapt_aborted(record)

    def on_adapt_completed(self, record):
        """Handle completion of feval from sampling phase."""
        self.num_evals += 1
        self.pending_evals -= 1

        xx, fx = np.copy(record.params[0]), record.value
        self.X = np.vstack((self.X, np.asmatrix(xx)))
        self.fX = np.vstack((self.fX, fx))
        self.surrogate.add_points(xx, fx)
        self.remove_pending(xx)

        self.log_completion(record)
        self.fevals.append(record)

    def on_adapt_aborted(self, record):
        """Handle aborted feval from sampling phase."""
        self.pending_evals -= 1
        xx = np.copy(record.params[0])
        self.remove_pending(xx)
        self.fevals.append(record)


class SRBFStrategy(SurrogateBaseStrategy):
    """Stochastic RBF (SRBF) optimization strategy.

    This is an implementation of the SRBF strategy by Regis and Shoemaker:

    Rommel G Regis and Christine A Shoemaker.
    A stochastic radial basis function method for the \
        global optimization of expensive functions.
    INFORMS Journal on Computing, 19(4): 497–509, 2007.

    Rommel G Regis and Christine A Shoemaker.
    Parallel stochastic global optimization using radial basis functions.
    INFORMS Journal on Computing, 21(3):411–426, 2009.

    The main idea is to pick the new evaluations from a set of candidate
    points where each candidate point is generated as an N(0, sigma^2)
    distributed perturbation from the current best solution. The value of
    sigma is modified based on progress and follows the same logic as in many
    trust region methods; we increase sigma if we make a lot of progress
    (the surrogate is accurate) and decrease sigma when we aren't able to
    make progress (the surrogate model is inaccurate). More details about how
    sigma is updated is given in the original papers.

    After generating the candidate points we predict their objective function
    value and compute the minimum distance to previously evaluated point. Let
    the candidate points be denoted by C and let the function value predictions
    be s(x_i) and the distance values be d(x_i), both rescaled through a linear
    transformation to the interval [0,1]. This is done to put the values on the
    same scale. The next point selected for evaluation is the candidate point
    x that minimizes the weighted-distance merit function:

    merit(x) := w * s(x) + (1 - w) * (1 - d(x))

    where 0 <= w <= 1. That is, we want a small function value prediction and a
    large minimum distance from previously evalauted points. The weight w is
    commonly cycled between a few values to achieve both exploitation and
    exploration. When w is close to zero we do pure exploration while w close
    to 1 corresponds to explotation.

    This strategy has two additional arguments than the base class:

    weights:  Specify a list of weights to cycle through
              Default = [0.3, 0.5, 0.8, 0.95]
    num_cand: Number of candidate to use when generating new evaluations
              Default = 100 * dim

    :param max_evals: Evaluation budget
    :type max_evals: int
    :param opt_prob: Optimization problem object
    :type opt_prob: OptimizationProblem
    :param exp_design: Experimental design object
    :type exp_design: ExperimentalDesign
    :param surrogate: Surrogate object
    :type surrogate: Surrogate
    :param asynchronous: Whether or not to use asynchrony (True/False)
    :type asynchronous: bool
    :param batch_size: Size of the batch (use 1 for serial, ignored if async)
    :type batch_size: int
    :param extra_points: Extra points to add to the experimental design
    :type extra_points: numpy.array of size n x dim
    :param extra_vals: Values for extra_points (np.nan/np.inf if unknown)
    :type extra_vals: numpy.array of size n x 1
    :param reset_surrogate: Whether or not to reset surrogate model
    :type reset_surrogate: bool
    :param weights: Weights for merit function, default = [0.3, 0.5, 0.8, 0.95]
    :type weights: list of np.array
    :param num_cand: Number of candidate points, default = 100*dim
    :type num_cand: int
    """
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                 asynchronous=True, batch_size=None, extra_points=None,
                 extra_vals=None, reset_surrogate=True, weights=None,
                 num_cand=None):

        self.fbest = np.inf  # Current best function value

        self.dtol = 1e-3 * math.sqrt(opt_prob.dim)
        if weights is None:
            weights = [0.3, 0.5, 0.8, 0.95]
        self.weights = weights
        self.next_weight = 0

        if num_cand is None:
            num_cand = 100*opt_prob.dim
        self.num_cand = num_cand

        self.sampling_radius_min = 0.2 * (0.5 ** 6)
        self.sampling_radius_max = 0.2
        self.sampling_radius = 0.2

        if asynchronous:
            d = float(opt_prob.dim)
            self.failtol = int(max(np.ceil(d), 4.0))
        else:
            d, p = float(opt_prob.dim), float(batch_size)
            self.failtol = int(max(np.ceil(d / p), np.ceil(4 / p)))
        self.succtol = 3
        self.maxfailtol = 4 * self.failtol

        self.status = 0          # Status counter
        self.failcount = 0       # Failure counter

        self.record_queue = []  # Completed records that haven't been processed

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals,
                         reset_surrogate=reset_surrogate)

    def check_input(self):
        """Check inputs."""
        assert isinstance(self.weights, list) or \
            isinstance(self.weights, np.array)
        for w in self.weights:
            assert isinstance(w, float) and w >= 0.0 and w <= 1.0
        super().check_input()

    def on_adapt_completed(self, record):
        """Handle completed evaluation."""
        super().on_adapt_completed(record)
        self.record_queue.append(record)

        if self.asynchronous:  # Process immediately
            self.adjust_step()
        elif (not self.batch_queue) and self.pending_evals == 0:  # Batch
            self.adjust_step()

    def get_weights(self, num_pts):
        """Generate the nextw weights."""
        weights = []
        for _ in range(num_pts):
            weights.append(self.weights[self.next_weight])
            self.next_weight = (self.next_weight + 1) % len(self.weights)
        return weights

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        weights = self.get_weights(num_pts=num_pts)
        new_points = candidate_srbf(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self.X, fX=self.fX, Xpend=self.Xpend, weights=weights,
            sampling_radius=self.sampling_radius, num_cand=self.num_cand)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

    def adjust_step(self):
        """Adjust the sampling radius sigma.

        After succtol successful steps, we cut the sampling radius;
        after failtol failed steps, we double the sampling radius.
        """
        # Check if we succeeded at significant improvement
        fbest_new = min([record.value for record in self.record_queue])
        if fbest_new < self.fbest - 1e-3*math.fabs(self.fbest) or \
                np.isinf(self.fbest):  # Improvement
            self.fbest = fbest_new
            self.status = max(1, self.status + 1)
            self.failcount = 0
        else:
            self.status = min(-1, self.status - 1)  # No improvement
            self.failcount += 1

        # Check if step needs adjusting
        if self.status <= -self.failtol:
            self.status = 0
            self.sampling_radius /= 2
            logger.info("Reducing sampling radius")
        if self.status >= self.succtol:
            self.status = 0
            self.sampling_radius = min([2.0 * self.sampling_radius,
                                        self.sampling_radius_max])
            logger.info("Increasing sampling radius")

        # Check if we want to terminate
        if self.failcount >= self.maxfailtol or \
                self.sampling_radius <= self.sampling_radius_min:
            self.terminate = True

        # Empty the queue
        self.record_queue = []


class DYCORSStrategy(SRBFStrategy):
    """DYCORS optimization strategy.

    This is an implementation of the DYCORS strategy by Regis and Shoemaker:

    Rommel G Regis and Christine A Shoemaker.
    Combining radial basis function surrogates and dynamic coordinate \
        search in high-dimensional expensive black-box optimization.
    Engineering Optimization, 45(5): 529–555, 2013.

    This is an extension of the SRBF strategy that changes how the candidate
    points are generated. The main idea is that many objective functions depend
    only on a few directions so it may be advantageous to perturb only a few
    directions. In particular, we use a perturbation probability to perturb a
    given coordinate and decrease this probability after each function
    evaluation so fewer coordinates are perturbed later in the optimization.

    :param max_evals: Evaluation budget
    :type max_evals: int
    :param opt_prob: Optimization problem object
    :type opt_prob: OptimizationProblem
    :param exp_design: Experimental design object
    :type exp_design: ExperimentalDesign
    :param surrogate: Surrogate object
    :type surrogate: Surrogate
    :param asynchronous: Whether or not to use asynchrony (True/False)
    :type asynchronous: bool
    :param batch_size: Size of the batch (use 1 for serial, ignored if async)
    :type batch_size: int
    :param extra_points: Extra points to add to the experimental design
    :type extra_points: numpy.array of size n x dim
    :param extra_vals: Values for extra_points (np.nan/np.inf if unknown)
    :type extra_vals: numpy.array of size n x 1
    :param reset_surrogate: Whether or not to reset surrogate model
    :type reset_surrogate: bool
    :param weights: Weights for merit function, default = [0.3, 0.5, 0.8, 0.95]
    :type weights: list of np.array
    :param num_cand: Number of candidate points, default = 100*dim
    :type num_cand: int
    """
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                 asynchronous=True, batch_size=None, extra_points=None,
                 extra_vals=None, weights=None, num_cand=None):

        self.num_exp = exp_design.num_pts  # We need this later

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals,
                         weights=weights, num_cand=num_cand)

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)
        new_points = candidate_dycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self.X, fX=self.fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))


class EIStrategy(SurrogateBaseStrategy):
    """Expected Improvement strategy.

    This is an implementation of Expected Improvement (EI), arguably the most
    popular acquisition function in Bayesian optimization. Under a Gaussian
    process (GP) prior, the expected value of the improvement:

    I(x) := max(f_best - f(x), 0)
    EI[x] := E[I(x)]

    can be computed analytically, where f_best is the best observed function
    value.EI is one-step optimal in the sense that selecting the maximizer of
    EI is the optimal action if we have exactly one function value remaining
    and must return a solution with a known function value.

    When using parallelism, we constrain each new evaluation to be a distance
    dtol away from previous and pending evaluations to avoid that the same
    point is being evaluated multiple times. We use a default value of
    dtol = 1e-3 * norm(ub - lb), but note that this value has not been
    tuned carefully and may be far from optimal.

    The optimization strategy terminates when the evaluatio budget has been
    exceeded or when the EI of the next point falls below some threshold,
    where the default threshold is 1e-6 * (max(fX) -  min(fX)).

    :param max_evals: Evaluation budget
    :type max_evals: int
    :param opt_prob: Optimization problem object
    :type opt_prob: OptimizationProblem
    :param exp_design: Experimental design object
    :type exp_design: ExperimentalDesign
    :param surrogate: Surrogate object
    :type surrogate: Surrogate
    :param asynchronous: Whether or not to use asynchrony (True/False)
    :type asynchronous: bool
    :param batch_size: Size of the batch (use 1 for serial, ignored if async)
    :type batch_size: int
    :param extra_points: Extra points to add to the experimental design
    :type extra_points: numpy.array of size n x dim
    :param extra_vals: Values for extra_points (np.nan/np.inf if unknown)
    :type extra_vals: numpy.array of size n x 1
    :param reset_surrogate: Whether or not to reset surrogate model
    :type reset_surrogate: bool
    :param ei_tol: Terminate if the largest EI falls below this threshold
        Default: 1e-6 * (max(fX) -  min(fX))
    :type ei_tol: float
    :param dtol: Minimum distance between new and pending evaluations
        Default: 1e-3 * norm(ub - lb)
    :type dtol: float
    """
    def __init__(self, max_evals, opt_prob, exp_design,
                 surrogate, asynchronous=True, batch_size=None,
                 extra_points=None, extra_vals=None,
                 reset_surrogate=True, ei_tol=None, dtol=None):

        if dtol is None:
            dtol = 1e-3 * np.linalg.norm(opt_prob.ub - opt_prob.lb)
        self.dtol = dtol
        self.ei_tol = ei_tol

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals)

    def check_input(self):
        super().check_input()
        assert isinstance(self.surrogate, GPRegressor)

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        ei_tol = self.ei_tol
        if ei_tol is None:
            ei_tol = 1e-6 * (self.fX.max() - self.fX.min())

        new_points = expected_improvement_ga(
            num_pts=num_pts, opt_prob=self.opt_prob, surrogate=self.surrogate,
            X=self.X, fX=self.fX, Xpend=self.Xpend, dtol=self.dtol,
            ei_tol=ei_tol)

        if new_points is None:  # Not enough improvement
            self.terminate = True
        else:
            for i in range(num_pts):
                self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))


class LCBStrategy(SurrogateBaseStrategy):
    """Lower confidence bound strategy.

    This is an implementation of Lower Confidence Bound (LCB), a
    popular acquisition function in Bayesian optimization. The main idea
    is to minimize:

    LCB(x) := E[x] - kappa * sqrt(V[x])

    where E[x] is the predicted function value, V[x] is the predicted
    variance, and kappa is a constant that balances exploration and
    exploitation. We use a default value of kappa = 2.

    When using parallelism, we constrain each new evaluation to be a distance
    dtol away from previous and pending evaluations to avoid that the same
    point is being evaluated multiple times. We use a default value of
    dtol = 1e-3 * norm(ub - lb), but note that this value has not been
    tuned carefully and may be far from optimal.

    The optimization strategy terminates when the evaluatio budget has been
    exceeded or when the LCB of the next point falls below some threshold,
    where the default threshold is 1e-6 * (max(fX) -  min(fX)).

    :param max_evals: Evaluation budget
    :type max_evals: int
    :param opt_prob: Optimization problem object
    :type opt_prob: OptimizationProblem
    :param exp_design: Experimental design object
    :type exp_design: ExperimentalDesign
    :param surrogate: Surrogate object
    :type surrogate: Surrogate
    :param asynchronous: Whether or not to use asynchrony (True/False)
    :type asynchronous: bool
    :param batch_size: Size of the batch (use 1 for serial, ignored if async)
    :type batch_size: int
    :param extra_points: Extra points to add to the experimental design
    :type extra_points: numpy.array of size n x dim
    :param extra_vals: Values for extra_points (np.nan/np.inf if unknown)
    :type extra_vals: numpy.array of size n x 1
    :param reset_surrogate: Whether or not to reset surrogate model
    :type reset_surrogate: bool
    :param kappa: Constant in front of sigma(x)
    :type kappa: float
    :param dtol: Minimum distance between new and pending evaluations
        Default: 1e-3 * norm(ub - lb)
    :type dtol: float
    :param lcb_tol: Terminate if max(fX) - min(LCB(x)) < lcb_tol
        Default: 1e-6 * (max(fX) -  min(fX))
    :type lcb_tol: float
    """
    def __init__(self, max_evals, opt_prob, exp_design,
                 surrogate, asynchronous=True, batch_size=None,
                 extra_points=None, extra_vals=None,
                 reset_surrogate=True, kappa=2.0, dtol=None,
                 lcb_tol=None):

        if dtol is None:
            dtol = 1e-3 * np.linalg.norm(opt_prob.ub - opt_prob.lb)
        self.dtol = dtol
        self.lcb_tol = lcb_tol
        self.kappa = kappa

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals)

    def check_input(self):
        super().check_input()
        assert isinstance(self.surrogate, GPRegressor)

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        lcb_tol = self.lcb_tol
        if lcb_tol is None:
            lcb_tol = 1e-6 * (self.fX.max() - self.fX.min())
        lcb_target = self.fX.min() - lcb_tol

        new_points = lower_confidence_bound_ga(
            num_pts=num_pts, opt_prob=self.opt_prob, surrogate=self.surrogate,
            X=self.X, fX=self.fX, Xpend=self.Xpend, kappa=self.kappa,
            dtol=self.dtol, lcb_target=lcb_target)

        if new_points is None:  # Not enough improvement
            self.terminate = True
        else:
            for i in range(num_pts):
                self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))
