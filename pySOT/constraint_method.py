"""
.. module:: constraint_method
   :synopsis: Penalty-based constraint formulation
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

import numpy as np

class PenaltyMethod(object):
    def __init__(self, penalty=1):
        self.outerLoop = True
        self.only_choose_feasible = True
        self.penalty = penalty
        self.initpenalty = penalty

    def eval(self, data, x, val):
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
        if data.constraints is not None:
            vec = np.array(data.eval_ineq_constraints(x))
            return np.amax(vec) <= 0.0
        else:
            return True

    def update_penalty(self):
        self.penalty *= 10

    def reset(self):
        self.penalty = self.initpenalty
