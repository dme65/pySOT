"""
.. module:: test_extra_vals
  :synopsis: Test extra values
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import Ackley, SyncStrategyNoConstraints, \
    SymmetricLatinHypercube, RBFInterpolant, CubicKernel, \
    LinearTail, CandidateDYCORS
from poap.controller import ThreadController, BasicWorkerThread, EvalRecord
import numpy as np
import os.path
import logging


def main():
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

    data = Ackley(dim=10)
    print(data.info)

    nextra = 10
    extra = np.random.uniform(data.xlow, data.xup, (nextra, data.dim))
    extra_vals = np.nan * np.ones((nextra, 1))
    for i in range(nextra):  # Evaluate every second point
        if i % 2 == 0:
            extra_vals[i] = data.objfunction(extra[i, :])

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail,
                                            maxp=maxeval),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim),
            extra=extra, extra_vals=extra_vals)

    # Append the known function values to the POAP database since POAP won't evaluate these points
    for i in range(len(extra_vals)):
        if not np.isnan(extra_vals[i]):
            record = EvalRecord(params=(np.ravel(extra[i, :]),), status='completed')
            record.value = extra_vals[i]
            record.feasible = True
            controller.fevals.append(record)

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))

if __name__ == '__main__':
    main()
