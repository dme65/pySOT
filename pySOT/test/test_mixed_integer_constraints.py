"""
.. module:: test_mixed_integer_constraints
  :synopsis: Test Mixed integer problem with constraints
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
    if os.path.exists("./logfiles/test_mixed_integer_constraints.log"):
        os.remove("./logfiles/test_mixed_integer_constraints.log")
    logging.basicConfig(filename="./logfiles/test_mixed_integer_constraints.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 200")
    print("Sampling methods: CandidateDYCORS, CandidateDYCORS_INT"
          ", CandidateDYCORS_CONT, CandidateUniform")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 4
    maxeval = 500
    penalty = 1e6
    nsamples = nthreads

    data = LinearMI()
    print(data.info)

    exp_design = SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1))
    response_surface = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval, dim=data.dim)

    # Use a multi-search strategy for candidate points
    sampling_method = MultiSampling(
        [CandidateDYCORS(data=data, numcand=100*data.dim),
         CandidateUniform(data=data, numcand=100*data.dim),
         CandidateDYCORS_INT(data=data, numcand=100*data.dim),
         CandidateDYCORS_CONT(data=data, numcand=100*data.dim)],
        [0, 1, 2, 3])

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyPenalty(
            worker_id=0, data=data,
            response_surface=response_surface,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=exp_design,
            sampling_method=sampling_method,
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
    print('Feasible: {0}\n'.format(np.max(data.eval_ineq_constraints(np.atleast_2d(xbest))) <= 0.0))


if __name__ == '__main__':
    main()
