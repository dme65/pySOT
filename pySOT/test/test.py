#!/usr/bin/env python
"""
.. module:: test
  :synopsis: A test routine that uses pySOT to minimize the 30
    dimensional Ackley function using 4 threads and 1000
    function eval
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
                 David Bindel <bindel@cornell.edu>

:Module: test
:Author: David Eriksson <dme65@cornell.edu>,
    David Bindel <bindel@cornell.edu>

A test routine that uses pySOT to minimize the 30
dimensional Ackley function using 4 threads and 1000
function evaluations
"""

from pySOT import CandidateDyCORS,  LatinHypercube, PenaltyMethod, RSCapped, \
    RBFInterpolant, phi_cubic, linear_tail, dphi_cubic, \
    dlinear_tail, Ackley, optimize
import numpy as np


def _main():
    print('This is a demo of the Surrogate Optimization Toolbox')
    # Number of threads
    nthreads = 4
    # Maximal number of function evaluations
    maxeval = 1000
    # Maximal number of simultaneous evaluations
    nsample = nthreads
    # Optimization problem
    data = Ackley(dim=30)
    # Experimental design
    ed = LatinHypercube(dim=data.dim, npts=2*data.dim+1)
    # Surrogate
    rs = RSCapped(RBFInterpolant(phi=phi_cubic, P=linear_tail,
                                 dphi=dphi_cubic, dP=dlinear_tail,
                                 eta=1e-8, maxp=maxeval))
    # How to generate new points to evaluate
    ss = CandidateDyCORS(data=data, numcand=100*data.dim)
    # Constraint handling
    ch = PenaltyMethod(penalty=1.0)

    print(data.info)
    # Optimize the objective function in synchronous parallel using nthreads
    # threads and at most maxeval function evaluations
    xbest, fbest = optimize(nthreads=nthreads, data=data, maxeval=maxeval,
                            nsample=nsample, response_surface=rs,
                            experimental_design=ed, search_strategies=ss,
                            constraint_handler=ch)

    # Print the best solution found
    print("\nBest function value: %f" % fbest)
    print("Best solution: ")
    np.set_printoptions(suppress=True)
    print(xbest)

if __name__ == "__main__":
    _main()
