"""
.. module:: test_extra_vals
  :synopsis: Test extra values
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.adaptive_sampling import CandidateDYCORS
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Ackley

from poap.controller import ThreadController, BasicWorkerThread, EvalRecord
import numpy as np
import os.path
import logging


def test_extra_vals():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_extra_vals.log"):
        os.remove("./logfiles/test_extra_vals.log")
    logging.basicConfig(filename="./logfiles/test_extra_vals.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 4
    maxeval = 500
    nsamples = nthreads

    opt_prob = Ackley(dim=10)
    print(opt_prob.info)

    nextra = 10
    extra = np.random.uniform(opt_prob.lb, opt_prob.ub, (nextra, opt_prob.dim))
    extra_vals = np.nan * np.ones((nextra, 1))
    for i in range(nextra):  # Evaluate every second point
        if i % 2 == 0:
            extra_vals[i] = opt_prob.eval(extra[i, :])

    surrogate = RBFInterpolant(opt_prob.dim, kernel=CubicKernel(),
                               tail=LinearTail(opt_prob.dim), maxpts=maxeval + nextra)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SRBFStrategy(worker_id=0, maxeval=maxeval, opt_prob=opt_prob,
                     exp_design=SymmetricLatinHypercube(dim=opt_prob.dim, npts=2 * (opt_prob.dim + 1)),
                     surrogate=surrogate, sampling_method=CandidateDYCORS(data=opt_prob, numcand=100*opt_prob.dim),
                     batch_size=nsamples, async=True, extra=extra, extra_vals=extra_vals)

    # Append the known function values to the POAP database since POAP won't evaluate these points
    for i in range(len(extra_vals)):
        if not np.isnan(extra_vals[i]):
            record = EvalRecord(params=(np.ravel(extra[i, :]),), status='completed')
            record.value = extra_vals[i]
            record.feasible = True
            controller.fevals.append(record)

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, opt_prob.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    test_extra_vals()
