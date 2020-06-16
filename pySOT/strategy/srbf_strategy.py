import logging
import math

import numpy as np

from ..auxiliary_problems import candidate_srbf
from .surrogate_strategy import SurrogateBaseStrategy

# Get module-level logger
logger = logging.getLogger(__name__)


class SRBFStrategy(SurrogateBaseStrategy):
    """Stochastic RBF (SRBF) optimization strategy.

    This is an implementation of the SRBF strategy by Regis and Shoemaker:

    Rommel G Regis and Christine A Shoemaker.
    A stochastic radial basis function method for the \
        global optimization of expensive functions.
    INFORMS Journal on Computing, 19(4): 497-509, 2007.

    Rommel G Regis and Christine A Shoemaker.
    Parallel stochastic global optimization using radial basis functions.
    INFORMS Journal on Computing, 21(3): 411-426, 2009.

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

    def __init__(
        self,
        max_evals,
        opt_prob,
        exp_design,
        surrogate,
        asynchronous=True,
        batch_size=None,
        extra_points=None,
        extra_vals=None,
        use_restarts=True,
        weights=None,
        num_cand=None,
    ):

        self.dtol = 1e-3 * math.sqrt(opt_prob.dim)
        if weights is None:
            weights = [0.3, 0.5, 0.8, 0.95]
        self.weights = weights
        self.next_weight = 0

        if num_cand is None:
            num_cand = 100 * opt_prob.dim
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

        super().__init__(
            max_evals=max_evals,
            opt_prob=opt_prob,
            exp_design=exp_design,
            surrogate=surrogate,
            asynchronous=asynchronous,
            batch_size=batch_size,
            extra_points=extra_points,
            extra_vals=extra_vals,
            use_restarts=use_restarts,
        )

    def check_input(self):
        """Check inputs."""
        assert isinstance(self.weights, list) or isinstance(self.weights, np.array)
        for w in self.weights:
            assert isinstance(w, float) and w >= 0.0 and w <= 1.0
        super().check_input()

    def sample_initial(self):
        super().sample_initial()
        self.status = 0  # Status counter
        self.failcount = 0  # Failure counter
        self.sampling_radius = 0.2
        self._fbest = np.inf  # Current best function value

    def on_adapt_completed(self, record):
        """Handle completed evaluation."""
        super().on_adapt_completed(record)

        if record.ev_id > self.ev_last:  # Only process fresh records
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
            opt_prob=self.opt_prob,
            num_pts=num_pts,
            surrogate=self.surrogate,
            X=self._X,
            fX=self._fX,
            Xpend=self.Xpend,
            weights=weights,
            sampling_radius=self.sampling_radius,
            num_cand=self.num_cand,
        )

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

    def adjust_step(self):
        """Adjust the sampling radius sigma.

        After succtol successful steps, we cut the sampling radius;
        after failtol failed steps, we double the sampling radius.
        """
        # Check if we succeeded at significant improvement
        fbest_new = min([record.value for record in self.record_queue])
        if fbest_new < self._fbest - 1e-3 * math.fabs(self._fbest) or np.isinf(self._fbest):  # Improvement
            self._fbest = fbest_new
            self.status = max(1, self.status + 1)
            self.failcount = 0
        else:
            self.status = min(-1, self.status - 1)  # No improvement
            self.failcount += 1

        # Check if step needs adjusting
        if self.status <= -self.failtol:
            self.ev_last = self.get_ev()  # Update the event id
            self.status = 0
            self.sampling_radius /= 2
            logger.info("Reducing sampling radius")
        if self.status >= self.succtol:
            self.ev_last = self.get_ev()  # Update the event id
            self.status = 0
            self.sampling_radius = min([2.0 * self.sampling_radius, self.sampling_radius_max])
            logger.info("Increasing sampling radius")

        # Check if we have converged
        if self.failcount >= self.maxfailtol or self.sampling_radius <= self.sampling_radius_min:
            self.converged = True

        # Empty the queue
        self.record_queue = []
