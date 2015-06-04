"""
.. module:: rs_capped
   :synopsis: Median-capped interpolation
.. moduleauthor:: David Bindel <bindel@cornell.edu>
"""

import numpy as np


class RSCapped(object):
    """Cap adapter for RBF response surface.

    This adapter takes an existing response surface and replaces it
    with a modified version in which any function values above the
    median are replaced by the median value.

    Attributes:
        fhat: Original response surface
        fvalues: Function values
        nump: Number of points currently active
        needs_update: Flag if we need to update fhat
    """

    def __init__(self, fhat):
        """Initialize the response surface adapter.

        Args:
            fhat: Original response surface object.
        """
        self.needs_update = False
        self.fhat = fhat
        self.fvalues = np.zeros((100, 1))
        self.nump = 0

    def reset(self):
        "Reset the response surface."
        self.fhat.reset()
        self.fvalues[:] = 0
        self.nump = 0

    def add_point(self, xx, fx):
        "Add a new point/value pair."
        if self.nump >= self.fvalues.shape[0]:
            self.fvalues.resize(2*self.fvalues.shape[0], 1)
        self.fvalues[self.nump] = fx
        self.nump += 1
        self.needs_update = True
        self.fhat.add_point(xx, fx)

    def get_x(self):
        "Get the list of points."
        return self.fhat.get_x()

    def get_fx(self):
        "Get the list of points."
        return self.fhat.get_fx()

    def eval(self, xx):
        "Evaluate at a point."
        self._apply_cap()
        return self.fhat.eval(xx)

    def evals(self, xx):
        "Evaluate at several points."
        self._apply_cap()
        return self.fhat.evals(xx)

    def deriv(self, xx):
        "Differentiate at a point."
        self._apply_cap()
        return self.fhat.deriv(xx)

    def _apply_cap(self):
        "Apply the cap to the function values."
        fvalues = np.copy(self.fvalues[0:self.nump])
        medf = np.median(fvalues)
        fvalues[fvalues > medf] = medf
        self.fhat.transform_fx(fvalues)
