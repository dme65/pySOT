import logging

import numpy as np

from ..auxiliary_problems import ei_ga
from ..surrogate import GPRegressor
from .surrogate_strategy import SurrogateBaseStrategy

# Get module-level logger
logger = logging.getLogger(__name__)


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
        ei_tol=None,
        dtol=None,
    ):

        if dtol is None:
            dtol = 1e-3 * np.linalg.norm(opt_prob.ub - opt_prob.lb)
        self.dtol = dtol
        self.ei_tol = ei_tol

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
        super().check_input()
        assert isinstance(self.surrogate, GPRegressor)

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        ei_tol = self.ei_tol
        if ei_tol is None:
            ei_tol = 1e-6 * (self.fX.max() - self.fX.min())

        new_points = ei_ga(
            num_pts=num_pts,
            opt_prob=self.opt_prob,
            surrogate=self.surrogate,
            X=self._X,
            fX=self._fX,
            Xpend=self.Xpend,
            dtol=self.dtol,
            ei_tol=ei_tol,
        )

        if new_points is None:  # Not enough improvement
            self.converged = True
        else:
            for i in range(num_pts):
                self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))
