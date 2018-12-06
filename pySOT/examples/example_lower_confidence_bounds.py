"""
.. module:: example_lower_confidence_bounds
  :synopsis: Example Lower Confidence Bounds
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import LCBStrategy
from pySOT.surrogate import GPRegressor
from pySOT.optimization_problems import Hartman6

from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging


def example_lower_confidence_bounds():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/example_lower_confidence_bounds.log"):
        os.remove("./logfiles/example_lower_confidence_bounds.log")
    logging.basicConfig(
        filename="./logfiles/example_lower_confidence_bounds.log",
        level=logging.INFO)

    num_threads = 4
    max_evals = 100

    hart6 = Hartman6()
    gp = GPRegressor(dim=hart6.dim)
    slhd = SymmetricLatinHypercube(
        dim=hart6.dim, num_pts=2*(hart6.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = LCBStrategy(
        max_evals=max_evals, opt_prob=hart6, exp_design=slhd,
        surrogate=gp, asynchronous=True)

    print("Number of threads: {}".format(num_threads))
    print("Maximum number of evaluations: {}".format(max_evals))
    print("Strategy: {}".format(controller.strategy.__class__.__name__))
    print("Experimental design: {}".format(slhd.__class__.__name__))
    print("Surrogate: {}".format(gp.__class__.__name__))

    # Launch the threads and give them access to the objective function
    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, hart6.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))

if __name__ == '__main__':
    example_lower_confidence_bounds()
