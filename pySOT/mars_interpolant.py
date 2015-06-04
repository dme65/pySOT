#!/usr/bin/env python
"""
.. module:: earth_interpolant
   :synopsis: eartch model interpolation
.. moduleauthor:: Yi Shen <ys623@cornell.edu>
"""

import numpy as np
import numpy.linalg as la
from pyearth import Earth
import math

class MARSInterpolant(Earth):
    """Compute and evaluate Earth interpolants.

    Attributes:
        nump: Current number of points
        maxp: Initial maximum number of points (can grow)
        x: Interpolation points
        fx: Function evaluations of interpolation points
    """

    def __init__(self, maxp=100):
        self.nump = 0
        self.maxp = maxp
        self.x = None     # pylint: disable=invalid-name
        self.fx = None
        self.dim = None
        self.model= Earth()

    def reset(self):
        "Re-set the interpolation."
        self.nump = 0
        self.x = None
        self.fx = None

    def _alloc(self, dim):
        "Allocate storage for x, fx and rhs."
        maxp = self.maxp
        self.dim = dim
        self.x = np.zeros((maxp, dim))
        self.fx = np.zeros((maxp, 1))

    def _realloc(self, dim, extra=1):
        "Expand allocation to accomodate more points (if needed)."
        if self.nump == 0:
            self._alloc(dim)
        elif self.nump+extra > self.maxp:
            self.maxp = max(self.maxp*2, self.maxp+extra)
            self.x.resize((self.maxp, dim))
            self.fx.resize((self.maxp, 1))

    def get_x(self):
        "Get the list of data points."
        return self.x[:self.nump, :]

    def get_fx(self):
        "Get the list of data points."
        return self.fx[:self.nump, :]

    def add_point(self, xx, fx):
        "Add a new function evaluation."
        dim = len(xx)
        self._realloc(dim)
        self.x[self.nump, :] = xx
        self.fx[self.nump, :] = fx
        self.model.fit(self.x, self.fx)
        self.nump += 1

    def eval(self, xx):
        "Evaluate the interpolant at x."
        xx = np.expand_dims(xx, axis=0)
        fx = self.model.predict(xx)
        return fx[0]
        
    def evals(self, xx):
        "Evaluate the interpolant at xx points."
        fx = self.model.predict(xx)
        return fx

    def deriv(self, x):
        "Evaluate the derivative of the interpolant at x."
        x = np.expand_dims(x, axis=0)
        dfx = self.model.predict_deriv(x, variables=None)
        return dfx[0]

# ====================================================================
def main():
    "Main test routine"

    def test_f(x):
        "Test function"
        fx = x[1]*math.sin(x[0]) + x[0]*math.cos(x[1])
        return fx

    def test_df(x):
        "Derivative of test function"
        dfx = np.array([x[1]*math.cos(x[0])+math.cos(x[1]),
                        math.sin(x[0])-x[0]*math.sin(x[1])])
        return dfx

    # Set up Earth model
    fhat = EarthInterpolant(20)

    # Set up initial points to train the Earth model
    xs = np.random.rand(15, 2)
    for x in xs:
        fhat.add_point(x,test_f(x))

    x = np.random.rand(10, 2)
    fhx = fhat.evals(x)

    print(" \n------ (fx - fhx)/|fx| ----- ")
    for i in range(10):
        fx = test_f(x[i, :])
        print("Err: %e" % (abs(fx-fhx[i])/abs(fx)))

    print(" \n ------ (fx - fhx)/|fx| , |dfx-dfhx|/|dfx| -----")
    for i in range(10):
        xx = x[i, :]
        fx = test_f(xx)
        dfx = test_df(xx)
        fhx = fhat.eval(xx)
        dfhx = fhat.deriv(xx)
        print("Err (interp): %e : %e" % (abs(fx-fhx)/abs(fx),
                                         la.norm(dfx-dfhx)/la.norm(dfx)))


if __name__ == "__main__":
    main()
