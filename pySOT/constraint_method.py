"""
.. module:: constraint_method
   :synopsis: Penalty-based constraint formulation
.. moduleauthor:: David Eriksson <dme65@cornell.edu>

:Module: constraint_method
:Author: David Eriksson <dme65@cornell.edu>
"""

import numpy as np


class PenaltyMethod(object):
    """
    This is a standard penalty method that can be used
    in order to incorporate non-bound constraints. The
    penalty is initially set to 1.0 and multiplied by 10
    each time the best solution is infeasible so that the
    algorithm is forced to restart

    Given a penalty :math:`\\mu`, an objective function
    :math:`f(x)`, and constraints :math:`g_i(x) \\leq 0`
    we try to minimize the function:

    .. math::
        f(x) + \\mu \\sum_i \\max(0,g_i(x))^2

    :ivar initpenalty: Initial penalty
    :ivar penalty: Current penalty
    """

    def __init__(self, penalty=1.0):
        """
        Initialization

        :param penalty: Initial penalty (default is 1.0)
        """
        self.outerLoop = True
        self.only_choose_feasible = True
        self.penalty = penalty
        self.initpenalty = penalty

    def eval(self, data, x, val):
        """
        Method that replaces the function of :math:`f(x)` by the
        value of :math:`f(x) + \\mu \\sum_i \\max(0,g_i(x))^2`

        :param data: Optimization problem
        :param x: Point to evaluate the objective function at
        :param val: The value of f(x)
        :return: :math:`f(x) + \\mu \\sum_i \\max(0,g_i(x))^2`
        """
        if data.constraints is not None:
            # Get the constraint violations
            vec = np.array(data.eval_ineq_constraints(x))
            # Now apply the penalty for the constraint violation
            vec[np.where(vec < 0.0)] = 0.0
            vec **= 2
            return val + self.penalty * np.sum(vec)
        else:
            return val

    def feasible(self, data, x):
        """
        Returns True if x is feasible and False otherwise

        :param data: Optimization problem
        :param x: Point for which to check feasibility
        :return: True if x is feasible, False otherwise
        """
        if data.constraints is not None:
            vec = np.array(data.eval_ineq_constraints(x))
            return np.amax(vec) <= 0.0
        else:
            return True

    def update_penalty(self):
        """
        Updates the penalty by multiplying it by a factor 10
        """
        self.penalty *= 10

    def reset(self):
        """
        Resets the penalty to the initial penalty
        """
        self.penalty = self.initpenalty
