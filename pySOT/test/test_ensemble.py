"""
.. module:: test_ensemble
  :synopsis: Test Ensemble surrogates
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
    if os.path.exists("./logfiles/test_ensemble.log"):
        os.remove("./logfiles/test_ensemble.log")
    logging.basicConfig(filename="./logfiles/test_ensemble.log",
                        level=logging.INFO)

    print("\nNumber of threads: 5")
    print("Maximum number of evaluations: 250")
    print("Sampling method: CandidateSRBF")
    print("Experimental design: Symmetric Latin Hypercube + point [1,1,...,1]")
    print("Ensemble Surrogate: Cubic RBF, PolyReg")

    nthreads = 5
    maxeval = 250
    nsamples = nthreads

    data = Ackley(dim=5)
    print(data.info)

    # Use RBF + PolyReg
    bounds = np.vstack((data.xlow, data.xup)).T
    basisp = basis_TD(data.dim, 2)  # use order 2 and no cross-terms

    models = [
        RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval, dim=data.dim),
        PolyRegression(bounds, basisp)
    ]
    response_surface = EnsembleSurrogate(model_list=models, maxp=maxeval)

    # Add an additional point to the experimental design. If a good
    # solution is already known you can add this point to the
    # experimental design
    extra = np.ones((1, data.dim))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            response_surface=response_surface,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            sampling_method=CandidateSRBF(data=data, numcand=100*data.dim),
            extra=extra)

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    response_surface.compute_weights()
    print('Final weights: {0}'.format(
        np.array_str(response_surface.weights, max_line_width=np.inf,
                     precision=5, suppress_small=True)))

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    main()
