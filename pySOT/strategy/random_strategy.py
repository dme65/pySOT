import logging

import numpy as np
from poap.strategy import BaseStrategy, RetryStrategy

from ..utils import check_opt_prob

# Get module-level logger
logger = logging.getLogger(__name__)


class RandomStrategy(BaseStrategy):
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
