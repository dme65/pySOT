#!/usr/bin/env python
"""
..module:: test_problems.py
  :synopsis: Test routine for Surrogate optimization
..moduleauthor:: David Eriksson <dme65@cornell.edu>
                 David Bindel <bindel@cornell.edu>
"""

from pySOT import CandidateDyCORS,  LatinHypercube, PenaltyMethod, RSCapped, \
				  RBFInterpolant, phi_cubic, linear_tail, dphi_cubic, \
				  dlinear_tail, Ackley, optimize
import numpy as np

if __name__ == "__main__":
    print('This is a demo of the Surrogate Optimization Toolbox')
    nthreads = 4  # Number of threads
    maxeval = 1000  # Maximal number of function evaluations
    nsample = nthreads  # Maximal number of simultaneous evaluations
    data = Ackley(dim=30)  # Optimization problem
    ed = LatinHypercube(dim=data.dim, npts=2*data.dim+1)  # Experimental design
    rs = RSCapped(RBFInterpolant(phi=phi_cubic, P=linear_tail,
                                 dphi=dphi_cubic, dP=dlinear_tail,
                                 eta=1e-8, maxp=maxeval))  # Surrogate
    ss = CandidateDyCORS(data=data, numcand=100*data.dim)  # How to generate new points to evaluate
    ch = PenaltyMethod(penalty=1.0)  # Constraint handling

    print(data.info)
    # Optimize the objective function in synchronous parallel using nthreads
    # threads and at most maxeval function evaluations
    xbest, fbest = optimize(nthreads=nthreads, data=data, maxeval=maxeval,
                            nsample=nsample, response_surface=rs,
                            experimental_design=ed, search_strategies=ss,
                            constraint_handler=ch)

    # Print the best solution found
    print("\nBest function value: %f" % fbest)
    print("Best solution: " % xbest)
    np.set_printoptions(suppress=True)
    print(xbest)