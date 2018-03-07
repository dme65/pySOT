"""
.. module:: test_penalty
  :synopsis: Test constrained optimization strategy
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.adaptive_sampling import CandidateDYCORS
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Keane

from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging


def test_penalty():
    print("This is currently broken")
    return

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
        SRBFStrategy(
            worker_id=0, opt_prob=data, maxeval=maxeval, batch_size=nsamples,
            surrogate=RBFInterpolant(dim=data.dim, kernel=CubicKernel(),
                                     tail=LinearTail(data.dim), maxpts=maxeval),
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim))
            #,penalty=penalty)

    # Launch the threads
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.eval)
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
    print('Feasible: {0}\n'.format(np.max(data.eval_cheap(xbest)) <= 0.0))


if __name__ == '__main__':
    test_penalty()
