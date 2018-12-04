"""
.. module:: example_mars
  :synopsis: Example MARS
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.optimization_problems import Ackley

from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging

# Try to import MARS
try:
    from pySOT.surrogate import MARSInterpolant
except Exception as err:
    print("\nERROR: Failed to import MARS. This is likely "
          "because py-earth is not installed. Aborting.....\n")
    exit()


def example_mars():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/example_mars.log"):
        os.remove("./logfiles/example_mars.log")
    logging.basicConfig(filename="./logfiles/example_mars.log",
                        level=logging.INFO)

    num_threads = 4
    max_evals = 200

    ackley = Ackley(dim=5)
    try:
        mars = MARSInterpolant(dim=ackley.dim)
    except Exception as e:
        print(str(e))
        return
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=mars, asynchronous=True, batch_size=num_threads)

    print("Number of threads: {}".format(num_threads))
    print("Maximum number of evaluations: {}".format(max_evals))
    print("Strategy: {}".format(controller.strategy.__class__.__name__))
    print("Experimental design: {}".format(slhd.__class__.__name__))
    print("Surrogate: {}".format(mars.__class__.__name__))

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
    example_mars()
