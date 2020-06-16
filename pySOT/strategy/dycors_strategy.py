import logging

import numpy as np

from ..auxiliary_problems import candidate_dycors
from .srbf_strategy import SRBFStrategy

# Get module-level logger
logger = logging.getLogger(__name__)


class DYCORSStrategy(SRBFStrategy):
    """DYCORS optimization strategy.

    This is an implementation of the DYCORS strategy by Regis and Shoemaker:

    Rommel G Regis and Christine A Shoemaker.
    Combining radial basis function surrogates and dynamic coordinate \
        search in high-dimensional expensive black-box optimization.
    Engineering Optimization, 45(5): 529-555, 2013.

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

        self.num_exp = exp_design.num_pts  # We need this later

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
            weights=weights,
            num_cand=num_cand,
        )

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0 / self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min([20.0 / self.opt_prob.dim, 1.0]) * (1.0 - (np.log(num_evals) / np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)
        new_points = candidate_dycors(
            opt_prob=self.opt_prob,
            num_pts=num_pts,
            surrogate=self.surrogate,
            X=self._X,
            fX=self._fX,
            Xpend=self.Xpend,
            weights=weights,
            num_cand=self.num_cand,
            sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb,
        )

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))
