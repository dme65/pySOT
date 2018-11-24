"""
.. module:: example_multisampling
  :synopsis: Example multisampling strategy
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.optimization_problems import Ackley
from pySOT.adaptive_sampling import CandidateDYCORS, CandidateSRBF, CandidateUniform, MultiSampling
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.experimental_design import SymmetricLatinHypercube
from poap.controller import SerialController
import numpy as np
import os.path
import logging


def example_multisampling():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/example_multisampling.log"):
        os.remove("./logfiles/example_multisampling.log")
    logging.basicConfig(filename="./logfiles/example_multisampling.log",
                        level=logging.INFO)

    print("\nNumber of threads: 1")
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS, CandidateSRBF, CandidateUniform")
    print("Experimental design: Latin Hypercube")
    print("Surrogate: Cubic RBF")

    max_evals = 500
    ackley = Ackley(dim=10)
    print(ackley.info)

    # Create a strategy and a controller
    rbf = RBFInterpolant(dim=ackley.dim, kernel=CubicKernel(), 
                         tail=LinearTail(ackley.dim))
    sampling_list = [CandidateDYCORS(opt_prob=ackley, numcand=100*ackley.dim, max_evals=max_evals),
                     CandidateSRBF(opt_prob=ackley, numcand=100*ackley.dim),
                     CandidateUniform(opt_prob=ackley, numcand=100*ackley.dim)]
    multi_sampling = MultiSampling(opt_prob=ackley, sampling_list=sampling_list, cycle=[0, 1, 2])
    slhd = SymmetricLatinHypercube(dim=ackley.dim, npts=2*(ackley.dim+1))

    controller = SerialController(ackley.eval)
    controller.strategy = \
            SRBFStrategy(max_evals=max_evals, opt_prob=ackley, asynchronous=False,
                         exp_design=slhd, surrogate=rbf, adapt_sampling=multi_sampling,
                         batch_size=1)

    result = controller.run()
    best, xbest = result.value, result.params[0]

    print('Best value: {0}'.format(best))
    print('Best solution: {0}\n'.format(
        np.array_str(xbest, max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    example_multisampling()
