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
import scipy.spatial as scp
import os
from poap.strategy import BaseStrategy, Proposal, RetryStrategy

from pySOT.auxiliary_problems import candidate_srbf, candidate_dycors
from pySOT.auxiliary_problems import expected_improvement_ga, \
    lower_confidence_bound_ga
from pySOT.experimental_design import ExperimentalDesign
from pySOT.optimization_problems import OptimizationProblem
from pySOT.surrogate import Surrogate, GPRegressor
from pySOT.utils import check_opt_prob, nd_sorting,\
    check_radius_rule, POSITIVE_INFINITY

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
    :param use_restarts: Whether or not to restart after convergence
    :type use_restarts: bool
    """
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                 asynchronous=True, batch_size=None, extra_points=None,
                 extra_vals=None,use_restarts=True):
        self.asynchronous = asynchronous
        self.batch_size = batch_size

        # Save the objects
        self.opt_prob = opt_prob
        self.exp_design = exp_design
        self.surrogate = surrogate

        # Sampler state
        self.use_restarts = use_restarts  # Whether we are using restarts
        self.terminate = False  # Termination criterion (eval count by default) reached
        self.converged = False  # Current optimization has converged, we restart if possible
        self.proposal_counter = 0
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

        # Completed evaluations in the current run
        self._X = np.empty([0, opt_prob.dim])
        self._fX = np.empty([0, 1])

        # Event indices to keep track of if points where proposed before a restart
        self.ev_restart = 0
        self.ev_next = 1

        # Check inputs (implemented by each strategy)
        self.check_input()

        # Start with first experimental design
        self.sample_initial()

    def get_ev(self):
        """Return event index and increase the counter."""
        ev = self.ev_next
        self.ev_next += 1
        return ev

    @abc.abstractmethod  # pragma: no cover
    def generate_evals(self, num_pts):
        pass

    def check_input(self):
        """Check the inputs to the optimization strategt. """
        if not isinstance(self.surrogate, Surrogate):
            raise ValueError("surrogate must implement Surrogate")
        if not isinstance(self.exp_design, ExperimentalDesign):
            raise ValueError("exp_design must implement ExperimentalDesign")
        if not isinstance(self.opt_prob, OptimizationProblem):
            raise ValueError("opt_prob must implement OptimizationProblem")
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

        # Remove everything that was pending
        self.pending_evals = 0
        self.Xpend = np.empty([0, self.opt_prob.dim])

    def check_termination(self):
        """Check if evaluation based termination criterion has been reached."""
        if self.num_evals + self.pending_evals >= self.max_evals:  # Budget exhausted
            self.terminate = True

    def sample_restart(self):
        """Restart a run after convergence."""
        self.ev_restart = self.get_ev()

        # Reset data in current run
        self._X = np.empty([0, self.opt_prob.dim])
        self._fX = np.empty([0, 1])

        # Reset flags
        self.converged = False
        self.phase = 1

        # Generate new initial design
        self.sample_initial()

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

        # Reset surrogate model
        self.surrogate.reset()

        # NB: Experimental designs can now handle the mapping
        start_sample = self.exp_design.generate_points(
            lb=self.opt_prob.lb, ub=self.opt_prob.ub,
            int_var=self.opt_prob.int_var)
        assert start_sample.shape[1] == self.opt_prob.dim, \
            "Dimension mismatch between problem and experimental design"

        for j in range(self.exp_design.num_pts):
            self.batch_queue.append(start_sample[j, :])

        # We only evaluate these points before the first restart
        if self.extra_points is not None and len(self.X) == 0:
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

        We pop points from the initial batch queue if we are still in phase 1, otherwise we make proposals
        in the adaptive phase. We check two flags:
            self.converged: We restart if restarts are enabled
            self.terminate: We terminate the optimization procedure if no evaluations are pending

        NB: We allow workers to continue to the adaptive phase if
        the initial queue is empty. This implies that we need enough
        points in the experimental design for us to construct a
        surrogate.
        """

        # Check if termination criterion has been reached
        self.check_termination()

        # Decide the next action to take
        if self.terminate:  # Check if termination has been triggered
            if self.pending_evals == 0:  # Only terminate if nothing is pending, otherwise take no action
                return Proposal('terminate')
        elif self.converged:
            if self.use_restarts:  # Start a new run
                if self.asynchronous or self.pending_evals == 0:  # We can restart immidiately, else wait
                    self.sample_restart() # Trigger the restart
                    return self.init_proposal()  # We are now in phase 1, so make an initial proposal
                else:
                    return
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

            # We allow generate_evals to trigger a restart, so check if status has changed
            if self.converged:
                if self.use_restarts:  # Start a new run
                    if self.asynchronous or self.pending_evals == 0:  # We can restart immidiately, else wait
                        self.sample_restart() # Trigger the restart
                        return self.init_proposal()  # We are now in phase 1, so make an initial proposal
                    else:
                        return

            # Launch the new evaluations (the others will be triggered later)
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
        proposal.record.ev_id = self.get_ev()

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
        self.X = np.vstack((self.X, np.atleast_2d(xx)))
        self.fX = np.vstack((self.fX, fx))
        self.remove_pending(xx)

        # Only count point if it was proposed after the last restart
        if record.ev_id > self.ev_restart:
            self._X = np.vstack((self._X, np.atleast_2d(xx)))
            self._fX = np.vstack((self._fX, fx))
            self.surrogate.add_points(xx, fx)

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
        proposal.record.ev_id = self.get_ev()

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
        self.X = np.vstack((self.X, np.atleast_2d(xx)))
        self.fX = np.vstack((self.fX, fx))
        self.remove_pending(xx)

        # Only count point if it was proposed after the last restart
        if record.ev_id > self.ev_restart:
            self._X = np.vstack((self._X, np.atleast_2d(xx)))
            self._fX = np.vstack((self._fX, fx))
            self.surrogate.add_points(xx, fx)

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
    :param use_restarts: Whether or not to restart after convergence
    :type use_restarts: bool
    :param weights: Weights for merit function, default = [0.3, 0.5, 0.8, 0.95]
    :type weights: list of np.array
    :param num_cand: Number of candidate points, default = 100*dim
    :type num_cand: int
    """
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                 asynchronous=True, batch_size=None, extra_points=None,
                 extra_vals=None, use_restarts=True,
                 weights=None, num_cand=None):

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

        if asynchronous:
            d = float(opt_prob.dim)
            self.failtol = int(max(np.ceil(d), 4.0))
        else:
            d, p = float(opt_prob.dim), float(batch_size)
            self.failtol = int(max(np.ceil(d / p), np.ceil(4 / p)))
        self.succtol = 3
        self.maxfailtol = 4 * self.failtol

        self.record_queue = []  # Completed records that haven't been processed

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals,
                         use_restarts=use_restarts)

    def check_input(self):
        """Check inputs."""
        assert isinstance(self.weights, list) or \
            isinstance(self.weights, np.array)
        for w in self.weights:
            assert isinstance(w, float) and w >= 0.0 and w <= 1.0
        super().check_input()

    def sample_initial(self):
        super().sample_initial()
        self.status = 0          # Status counter
        self.failcount = 0       # Failure counter
        self.sampling_radius = 0.2
        self._fbest = np.inf  # Current best function value

    def on_adapt_completed(self, record):
        """Handle completed evaluation."""
        super().on_adapt_completed(record)

        if record.ev_id > self.ev_restart:  # Only process fresh records
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
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
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
        if fbest_new < self._fbest - 1e-3*math.fabs(self._fbest) or np.isinf(self._fbest):  # Improvement
            self._fbest = fbest_new
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

        # Check if we have converged
        if self.failcount >= self.maxfailtol or \
                self.sampling_radius <= self.sampling_radius_min:
            self.converged = True

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
    :param use_restarts: Whether or not to restart after convergence
    :type use_restarts: bool
    :param weights: Weights for merit function, default = [0.3, 0.5, 0.8, 0.95]
    :type weights: list of np.array
    :param num_cand: Number of candidate points, default = 100*dim
    :type num_cand: int
    """
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                 asynchronous=True, batch_size=None, extra_points=None,
                 extra_vals=None, use_restarts=True, weights=None, num_cand=None):

        self.num_exp = exp_design.num_pts  # We need this later

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals,
                         use_restarts=use_restarts,  weights=weights,
                         num_cand=num_cand)

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
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
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
    :type reset_surrogate: boo
    :param use_restarts: Whether or not to restart after convergence
    :type use_restarts: bool
    :param ei_tol: Terminate if the largest EI falls below this threshold
        Default: 1e-6 * (max(fX) -  min(fX))
    :type ei_tol: float
    :param dtol: Minimum distance between new and pending/finished evaluations
        Default: 1e-3 * norm(ub - lb)
    :type dtol: float
    """
    def __init__(self, max_evals, opt_prob, exp_design,
                 surrogate, asynchronous=True, batch_size=None,
                 extra_points=None, extra_vals=None,
                 use_restarts=True, ei_tol=None, dtol=None):

        if dtol is None:
            dtol = 1e-3 * np.linalg.norm(opt_prob.ub - opt_prob.lb)
        self.dtol = dtol
        self.ei_tol = ei_tol

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals,
                         use_restarts=use_restarts)

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
            X=self._X, fX=self._fX, Xpend=self.Xpend, dtol=self.dtol,
            ei_tol=ei_tol)

        if new_points is None:  # Not enough improvement
            self.converged = True
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
    :param kappa: Constant in the LCB merit function
    :type kappa: float
    :param dtol: Minimum distance between new and pending evaluations
        Default: 1e-3 * norm(ub - lb)
    :type dtol: float
    :param lcb_tol: Terminate if min(fX) - min(LCB(x)) < lcb_tol
        Default: 1e-6 * (max(fX) -  min(fX))
    :type lcb_tol: float
    """
    def __init__(self, max_evals, opt_prob, exp_design,
                 surrogate, asynchronous=True, batch_size=None,
                 extra_points=None, extra_vals=None,
                 use_restarts=True, kappa=2.0, dtol=None, lcb_tol=None):

        if dtol is None:
            dtol = 1e-3 * np.linalg.norm(opt_prob.ub - opt_prob.lb)
        self.dtol = dtol
        self.lcb_tol = lcb_tol
        self.kappa = kappa

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals,
                         use_restarts=use_restarts)

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
            X=self._X, fX=self._fX, Xpend=self.Xpend, kappa=self.kappa,
            dtol=self.dtol, lcb_target=lcb_target)

        if new_points is None:  # Not enough improvement
            self.converged = True
        else:
            for i in range(num_pts):
                self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))


class _SopRecord():
    """A custom record that stores memory attributes of a SOP-related record

    A multi-attribute record that stores the evaluation point and corresponding
    attributes including objective function value, failure count, elapsed tabu
    count, non-domination rank and search radius. Failure count, tabu count,
    rank and sigma are updated after a new function evaluation is completed.

    :param x: Decision variable
    :type x: numpy array
    :param fx: objective function value
    :type fx: float
    :param sigma: Candidate search radius
    :type sigma: float
    """
    def __init__(self, x, fx, sigma):
        self.x = x
        self.fx = fx
        self.rank = POSITIVE_INFINITY  # To-Do: Update ranks in future
        self._nfail = 0  # Count of failures(int)
        self._ntabu = 0  # Elapsed tabu tenure
        self._sigma = sigma

    @property
    def sigma(self):
        """Get value of radius / sigma"""
        return self._sigma

    @property
    def nfail(self):
        """Get failure count"""
        return self._nfail

    @property
    def ntabu(self):
        """Get elapsed tabu tenure count"""
        return self._ntabu

    def reduce_sigma(self):
        """Reduce sigma / search radius"""
        self._sigma = self._sigma/2.0

    def increment_failure_count(self):
        """Increase failure count"""
        self._nfail += 1

    def make_tabu(self, sigma):
        """Make this point tabu"""
        self._nfail = 0
        self._ntabu = 1
        self._sigma = sigma

    def increment_tabu_tenure(self):
        """Increment the elapsed tabu tenure"""
        self._ntabu += 1

    def reset(self, sigma):
        """Reset memory attributes"""
        self._nfail = 0
        self._ntabu = 0
        self._sigma = sigma


class _SopCenter():
    """ A custom reference record that stores information for a SOP center

    A multi-attribute reference record that stores the decision vector value
    of a SOP center, and correspondingly, its location in the list of evaluated
    SOP Records, the new point it generates and location of new point in list
    of evaluated records.

    :param xc: decision vector value of center
    :type xc: numpy array
    :param index: index location in list of evaluated points
    :type index: int
    """
    def __init__(self, xc, index):
        self.xc = xc
        self.index = index
        self._new_point = None  # New point proposed for eval around xc
        self._new_index = None  # Location of new point in list of evals

    @property
    def new_point(self):
        """Get new proposed point"""
        return self._new_point

    @new_point.setter
    def new_point(self, value):
        """Set new point, raise error if array length is diff from self.xc"""
        if len(value) != len(self.xc):
            raise ValueError('Dimension mismatch between center and new point')
        else:
            self._new_point = value

    @property
    def new_index(self):
        """Get location / index of new point"""
        return self._new_index

    @new_index.setter
    def new_index(self, value):
        """Set location of new point, raise error if not integer"""
        if not isinstance(value, int):
            raise ValueError('Index location is not an integer')
        else:
            self._new_index = value


class SOPStrategy(SurrogateBaseStrategy):
    """Surrogate Optimization with Pareto Selection Strategy

    This is an implementation of the SOP strategy by Krityakierne et. al:

    Tipaluck Krityakierne, Taimoor Akhtar and Christine A. Shoemaker.
    SOP: parallel surrogate global optimization with Pareto \
        center selection for computationally expensive \
        single objective problems.
    Journal of Global Optimization, 66(3): 417–437, 2016.

    The core idea of SOP is to maintain a ranked archive of all previously
    evaluated points, as per non-dominated sorting between two objectives,
    i.e., i)Objective function value(minimize) and ii)Minimum distance from
    other evaluated points(maximize). A sub-archive of center points is
    subsequently maintained via selection from the ranked evalauted points.
    The number of  points in the sub-archive of centers should be equal to
    the number of parallel threads (or greater than). Candidate points are
    generated around each 'center point' via the DYCORS sampling strategy,
    i.e., an N(0, sigma^2) distributed perturbation of a subset of decision
    variables. A separate value of sigma is maintained for each center
    point, where sigma is decreased if no progress is registered in the
    bi-objective objective value and distance criterion trade-off. One point
    is selected for expensive evaluation from each set of candidate points,
    based on the surrogate approximation only. Hence the merit function is
    s(x), where s(x) is the surrogate prediction.

    This strategy has two additional arguments than the base class:

    ncenters: Specify no. of centers (should be greater than no. of threads)
              Default = 4
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
    :param ncenters: Number of center points
    :type ncenters:  int
    :param asynchronous: Whether or not to use asynchrony (True/False)
    :type asynchronous: bool
    :param batch_size: Size of batch (Make sure batch_size<=ncenters for sync)
    :type batch_size: int
    :param extra_points: Extra points to add to the experimental design
    :type extra_points: numpy.array of size n x dim
    :param extra_vals: Values for extra_points (np.nan/np.inf if unknown)
    :type extra_vals: numpy.array of size n x 1
    :param reset_surrogate: Whether or not to reset surrogate model
    :type reset_surrogate: bool
    :param use_restarts: Whether or not to restart after convergence
    :type use_restarts: bool
    :param num_cand: Number of candidate points, default = 100*dim
    :type num_cand: int
    """
    def __init__(self, max_evals, opt_prob, exp_design, surrogate, ncenters=4,
                 asynchronous=True, batch_size=None, extra_points=None,
                 extra_vals=None, use_restarts=True, num_cand=None):

        self.dtol = 1e-3 * math.sqrt(opt_prob.dim)

        if num_cand is None:
            num_cand = 100*opt_prob.dim
        self.num_cand = num_cand

        self.sampling_radius = 0.2
        self.record_queue = []  # Completed records that haven't been processed
        self.num_exp = exp_design.num_pts  # We need this later
        self.ncenters = ncenters
        self.evals = []  # List of all eval points stored as _SOPRecord
        self.centers = []  # List of current center points as _SOPCenter
        self.F_ranked = None  # Evaluated points stored as numpy array
        self.d_thresh = 1.0

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals,
                         use_restarts=use_restarts)

    def check_input(self):
        """Check inputs."""
        super().check_input()
        if not isinstance(self.ncenters, int) and self.ncenters > 3:
            raise ValueError("ncenters should be an integer greater than 3")
        if not self.asynchronous:
            if not self.ncenters >= self.batch_size:
                raise ValueError("Batch size should be less than or equal"
                                 " to ncenters")

    def on_initial_completed(self, record):
        """Handle completed evaluation in initial phase"""
        super().on_initial_completed(record)

        if record.ev_id >= self.ev_restart:
            srec = _SopRecord(np.copy(record.params[0]), record.value,
                            self.sampling_radius)
            self.evals.append(srec)

    def on_adapt_completed(self, record):
        """Handle completed evaluation in phase 2."""
        super().on_adapt_completed(record)

        if record.ev_id >= self.ev_restart:
            self.record_queue.append(record)

            # Initiate a new SOP Record for new completed evaluation
            center_index = None
            srec = _SopRecord(np.copy(record.params[0]), record.value,
                            self.sampling_radius)
            self.evals.append(srec)

            ncenters = len(self.centers)
            for i in range(ncenters):  # Update location of new point in center
                if np.array_equal(np.copy(record.params[0]),
                                self.centers[i].new_point):
                    self.centers[i].new_index = self.num_evals - 1
                    center_index = i
                    break

            if self.asynchronous:  # Process immediately
                self.adjust_memory(center_index)
            elif (not self.batch_queue) and self.pending_evals == 0:  # Batch
                self.adjust_memory()

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""

        # Update the list of center points
        if self.F_ranked is None:  # If this is the start of adaptive phase
            self.update_ranks()
        self.update_center_list()

        # Compute dycors perturbation probability
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        # Perturb each center to propose one new eval per center
        new_points = np.zeros((num_pts, self.opt_prob.dim))
        weights = [1.0]
        for i in range(num_pts):
            # Deduce index of next available center
            center_index = 0
            for center in self.centers:
                if center.new_point is None:
                    break
                center_index += 1
            # Select new point by candidate search around center
            X_c = self.centers[center_index].xc
            sampling_radius =\
                self.evals[self.centers[center_index].index].sigma
            new_points[i, :] =\
                candidate_dycors(num_pts=1, opt_prob=self.opt_prob,
                                 surrogate=self.surrogate, X=self._X,
                                 fX=self._fX, weights=weights,
                                 sampling_radius=sampling_radius,
                                 num_cand=self.num_cand, Xpend=self.Xpend,
                                 prob_perturb=prob_perturb, xbest=X_c)

            self.centers[center_index].new_point = new_points[i, :]

        # submit the new points
        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

    def adjust_memory(self, index=None):
        """Update the memory attributes of evaluated points

        For each evaluated point (stored as _SOPRecord instance) update
        i) Failure count, ii) Tabu status and iii) Sampling radius.

        """

        if index is None:  # Batch mode - update memory for all centers
            indices = range(self.ncenters)
        else:  # asynchronous mode
            indices = [index]

        # Re-evaluate bi-objective ranks after adding new point(s)
        # NOTE: Re-evaluation needed because minimum distance is updated
        # for all points
        nevals = self.num_evals
        self.update_ranks()

        # Step 2 -- Adjust memory attributes of center point associated
        # with new eval by checking if we succeeded at improving the
        # distance-objective tradeoff
        for i in indices:
            cp = self.centers[i]
            center_index = cp.index
            check = 0
            new_index = cp.new_index
            rank = self.F_ranked[new_index, self.opt_prob.dim+3]
            if rank == 1:  # new point is in the non-dominated front
                check = 1  # success
            if check == 0:  # If no success increase failure count
                self.evals[center_index].increment_failure_count()
                self.evals[center_index].reduce_sigma()

        # Step 3 --- Update tabu list, i.e., i) include a center in
        # tabu list if its failure count is more than n_fail and ii)
        # remove a center from tabu list if it has been in the tabu
        # list for more than n_tenure iterations
        for i in range(nevals):  # check if pts are to be removed from tabu
            if self.evals[i].ntabu > 0:
                if self.evals[i].ntabu < 5:  # NOTE: Tabu tenure is 5
                    self.evals[i].increment_tabu_tenure()
                else:
                    self.evals[i].reset(self.sampling_radius)

        for i in indices:  # add a point to Tabu list if failures > fail_thresh
            cp = self.centers[i]
            index = cp.index
            if self.evals[index].nfail > 3:  # NOTE: max failure count is 4
                self.evals[index].make_tabu(self.sampling_radius)

        self.update_F()  # make sure that memory is updated in ranked F

    def update_F(self):
        """Update F_ranked numpy array"""
        nevals = self.num_evals
        F = np.zeros((nevals, self.opt_prob.dim+5))
        F[:, 0:self.opt_prob.dim] = [(val.x - self.opt_prob.lb) /
                                     (self.opt_prob.ub - self.opt_prob.lb)
                                     for val in self.evals]
        F[:, self.opt_prob.dim] = [val.fx for val in self.evals]
        F[:, self.opt_prob.dim+2] = [val.ntabu for val in self.evals]
        F[:, self.opt_prob.dim+3] = [val.rank for val in self.evals]
        F[:, self.opt_prob.dim+4] = [val.nfail for val in self.evals]
        self.F_ranked = np.copy(F)

    def update_ranks(self):
        """Updated ND ranks of evaluated points

        Non-dominated ranks of evaluated points are updated after
        new points have been evaluated.

        """
        nevals = self.num_evals
        self.update_F()
        F = np.copy(self.F_ranked)
        dists = scp.distance.cdist(F[:, 0:self.opt_prob.dim],
                                   F[:, 0:self.opt_prob.dim])
        for i in range(nevals):
            a = dists[i, :]
            F[i, self.opt_prob.dim+1] = -1.0*np.min(a[np.nonzero(a)])

        nmax = 100  # Maximum number of points that may be selected as centers
        ranks = nd_sorting(F[:, self.opt_prob.dim:self.opt_prob.dim+2]
                           .transpose(), nmax)  # Perform ND Sorting
        F[:, self.opt_prob.dim+3] = ranks.transpose()
        self.F_ranked = np.copy(F)

    def update_center_list(self):
        """This method for Updating the list of centers"

        This function updates the list of center points after new points
        have been evaluated. In a synchronous setting where batch_size
        = number of centers, the set of old centers is simply replaced
        by new centers. Otherwise, only the center point that was just
        processed, is replaced by a new center. Centers are selected from
        all evaluated points, after they are sorted according to i) ND rank
        and ii) Objective function value (Tabu points are pushed to the end
        in selection order).

        """
        nevals = self.num_evals
        F = np.copy(self.F_ranked)
        self.d_thresh = 1.0 - float(nevals - self.num_exp)\
            / float(self.max_evals - self.num_exp)

        # Step 1 - Remove center points around which new points have been
        # proposed and evaluated
        if len(self.centers) > 0:
            finished_centers = []
            for i in range(len(self.centers)):
                if self.centers[i].new_index is not None:
                    finished_centers.append(i)
                else:  # If a center is being processed, tag it as tabu
                    F[self.centers[i].index, self.opt_prob.dim+2] = 2
            for index in reversed(finished_centers):
                self.centers.pop(index)

        # Step 2 - Sort all evaluated points according to
        # i) tabu status, ii) rank, iii) obj value
        ind = np.lexsort((F[:, self.opt_prob.dim],
                          F[:, self.opt_prob.dim+3],
                          F[:, self.opt_prob.dim+2]))
        min_index = np.argmin(F[:, self.opt_prob.dim])
        if min_index == ind[0] or F[min_index, self.opt_prob.dim+2] == 2:
            ind_new = np.copy(ind)
        else:  # Put xbest at the top of sorted points (ind_new),
            # regardless of tabu status unless it is being processed
            ind_new = np.copy(ind)
            ind_new[0] = min_index
            check = 0
            i = 1
            while check == 0:
                ind_new[i] = ind[i-1]
                if ind[i] == min_index:
                    check = 1
                i = i + 1

        # Append new points to list of centers until length of centers is not
        # equal to number of centers
        num_pending_centers = len(self.centers)
        center_count = self.ncenters - num_pending_centers
        if center_count > 0:
            center_index = -1*np.ones((center_count,), dtype=np.int)
            center_index[0] = ind_new[0]
            check = 1
            i = 1
            while check < center_count:
                if i < nevals:
                    flag = check_radius_rule(
                        F[ind_new[i], 0:self.opt_prob.dim],
                        F[center_index, :], self.sampling_radius,
                        self.opt_prob.dim, check,
                        d_thresh=self.d_thresh)  # Radius Rule
                    if flag == 1:
                        center_index[check] = ind_new[i]
                        check = check + 1
                    i = i + 1
                else:
                    check_prev = check
                    while check < center_count:
                        center_index[check] = \
                            center_index[np.remainder(check, check_prev)]
                        check = check + 1
            # Initialize the center point list
            for index in center_index:
                crec = _SopCenter(self.evals[index].x, index)
                self.centers.append(crec)
