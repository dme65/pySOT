import logging

import numpy as np

from ..auxiliary_problems import lcb_ga
from ..surrogate import GPRegressor
from .surrogate_strategy import SurrogateBaseStrategy

# Get module-level logger
logger = logging.getLogger(__name__)


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
        kappa=2.0,
        dtol=None,
        lcb_tol=None,
    ):

        if dtol is None:
            dtol = 1e-3 * np.linalg.norm(opt_prob.ub - opt_prob.lb)
        self.dtol = dtol
        self.lcb_tol = lcb_tol
        self.kappa = kappa

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
        lcb_tol = self.lcb_tol
        if lcb_tol is None:
            lcb_tol = 1e-6 * (self.fX.max() - self.fX.min())
        lcb_target = self.fX.min() - lcb_tol

        new_points = lcb_ga(
            num_pts=num_pts,
            opt_prob=self.opt_prob,
            surrogate=self.surrogate,
            X=self._X,
            fX=self._fX,
            Xpend=self.Xpend,
            kappa=self.kappa,
            dtol=self.dtol,
            lcb_target=lcb_target,
        )

        if new_points is None:  # Not enough improvement
            self.converged = True
        else:
            for i in range(num_pts):
                self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))
