"""
.. module:: test_penalty
  :synopsis: Test constrained optimization strategy
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

import logging
from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_penalty.log"):
        os.remove("./logfiles/test_penalty.log")
    logging.basicConfig(filename="./logfiles/test_penalty.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 4
    maxeval = 500
    penalty = 1e6
    nsamples = nthreads

    data = Keane(dim=10)
    print(data.info)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyPenalty(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval, dim=data.dim),
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim),
            penalty=penalty)

    # Launch the threads
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Use penalty based merit
    def feasible_merit(record):
        xx = np.zeros((1, record.params[0].shape[0]))
        xx[0, :] = record.params[0]
        return record.value + controller.strategy.penalty_fun(xx)[0, 0]

    result = controller.run(merit=feasible_merit)
    best, xbest = result.value, result.params[0]

    print('Best value: {0}'.format(best))
    print('Best solution: {0}'.format(
        np.array_str(xbest, max_line_width=np.inf,
                     precision=5, suppress_small=True)))
    print('Feasible: {0}\n'.format(np.max(data.eval_ineq_constraints(xbest)) <= 0.0))

if __name__ == '__main__':
    main()
