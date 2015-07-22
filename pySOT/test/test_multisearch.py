"""
.. module:: test_multisearch
  :synopsis: Test multi-search strategy
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
    logging.basicConfig(filename="./logfiles/test_multisearch.log",
                        level=logging.INFO)

    print("\nNumber of threads: 1")
    print("Maximum number of evaluations: 200")
    print("Search strategy: CandidateDYCORS, Generic Algorithm, Multi-Start Gradient")
    print("Experimental design: Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 1
    maxeval = 200
    nsamples = nthreads

    data = Ackley(dim=10)
    print(data.info)

    # Create a strategy and a controller
    sp = [CandidateDYCORS(data=data, numcand=100*data.dim),
          GeneticAlgorithm(data=data), MultiStartGradient(data=data)]
    controller = SerialController(data.objfunction)
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            response_surface=RBFInterpolant(surftype=CubicRBFSurface, maxp=maxeval),
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            search_procedure=MultiSearchStrategy(sp, [0, 0, 1, 0, 0, 2]))

    result = controller.run()
    best, xbest = result.value, result.params[0]

    print('Best value: {0}'.format(best))
    print('Best solution: {0}\n'.format(
        np.array_str(xbest, max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    main()
