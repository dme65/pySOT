"""
.. module:: test_projection
  :synopsis: Test Projection strategy
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path


class AckleyUnit:
    def __init__(self, dim=10):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Ackley function on the unit sphere \n" +\
                             "Global optimum: f(1,0,...,0) = ... = f(0,0,...,1) = " +\
                             str(np.round(20*(1-np.exp(-0.2/np.sqrt(dim))), 3))
        self.min = 20*(1 - np.exp(-0.2/np.sqrt(dim)))
        self.integer = []
        self.continuous = np.arange(0, dim)
        check_opt_prob(self)

    def objfunction(self, x):
        n = float(len(x))
        return -20.0 * np.exp(-0.2*np.sqrt(np.sum(x**2)/n)) - \
            np.exp(np.sum(np.cos(2.0*np.pi*x))/n) + 20 + np.exp(1)

    def eval_eq_constraints(self, x):
        return np.linalg.norm(x) - 1


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_projection.log"):
        os.remove("./logfiles/test_projection.log")
    logging.basicConfig(filename="./logfiles/test_projection.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 1000")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 4
    maxeval = 1000
    nsamples = nthreads

    data = AckleyUnit(dim=10)
    print(data.info)

    def projection(x):
        return x / np.linalg.norm(x)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyProjection(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=LatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval, dim=data.dim),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim),
            proj_fun=projection
        )

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))
    print('||x||_2 = {0}\n'.format(np.linalg.norm(result.params[0])))


if __name__ == '__main__':
    main()
