"""
.. module:: test_multisampling
  :synopsis: Test multisampling strategy
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

import logging
from pySOT import *
from poap.controller import SerialController
import numpy as np
import os.path


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_multisampling.log"):
        os.remove("./logfiles/test_multisampling.log")
    logging.basicConfig(filename="./logfiles/test_multisampling.log",
                        level=logging.INFO)

    print("\nNumber of threads: 1")
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS, Genetic Algorithm, Multi-Start Gradient")
    print("Experimental design: Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 1
    maxeval = 500
    nsamples = nthreads

    data = Ackley(dim=10)
    print(data.info)

    # Create a strategy and a controller
    sampling_method = [CandidateDYCORS(data=data, numcand=100*data.dim),
                       GeneticAlgorithm(data=data), MultiStartGradient(data=data)]
    controller = SerialController(data.objfunction)
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval, dim=data.dim),
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim + 1)),
            sampling_method=MultiSampling(sampling_method, [0, 1, 0, 2]))

    result = controller.run()
    best, xbest = result.value, result.params[0]

    print('Best value: {0}'.format(best))
    print('Best solution: {0}\n'.format(
        np.array_str(xbest, max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    main()
