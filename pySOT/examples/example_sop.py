"""
.. module:: example_sop
  :synopsis: Example SOP
.. moduleauthor:: Taimoor Akhtar <erita@nus.edu.sg>
"""

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SOPStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, \
    LinearTail, SurrogateUnitBox
from pySOT.optimization_problems import Ackley

from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging


def example_sop():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/example_simple.log"):
        os.remove("./logfiles/example_simple.log")
    logging.basicConfig(filename="./logfiles/example_simple.log",
                        level=logging.INFO)

    print("\nNumber of threads: 8")
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    num_threads = 8
    max_evals = 500

    ackley = Ackley(dim=10)
    rbf = SurrogateUnitBox(RBFInterpolant(dim=ackley.dim, kernel=CubicKernel(),
                                          tail=LinearTail(ackley.dim)),
                           lb=ackley.lb, ub=ackley.ub)
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = SOPStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=rbf, asynchronous=False, ncenters=num_threads,
        batch_size=num_threads)

    print("Number of threads: {}".format(num_threads))
    print("Maximum number of evaluations: {}".format(max_evals))
    print("Strategy: {}".format(controller.strategy.__class__.__name__))
    print("Experimental design: {}".format(slhd.__class__.__name__))
    print("Surrogate: {}".format(rbf.__class__.__name__))

    # Launch the threads and give them access to the objective function
    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    example_sop()
