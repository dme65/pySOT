"""
.. module:: example_matlab_engine
  :synopsis: Example with MATLAB objective function
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Ackley

from poap.controller import ThreadController, ProcessWorkerThread
import numpy as np
import os.path
import logging

# Try to import the matlab_wrapper module
try:
    import matlab.engine
except Exception as err:
    print("\nERROR: Failed to import the matlab engine\n")
    pass


class MatlabWorker(ProcessWorkerThread):
    def handle_eval(self, record):
        try:
            x = matlab.double(record.params[0].tolist())
            val = self.matlab.ackley(x)
            if np.isnan(val):
                raise ValueError()
            self.finish_success(record, val)
        finally:
            logging.info("WARNING: Incorrect output or crashed evaluation")
            self.finish_cancelled(record)


def example_matlab_engine():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/example_matlab_engine.log"):
        os.remove("./logfiles/example_matlab_engine.log")
    logging.basicConfig(filename="./logfiles/example_matlab_engine.log",
                        level=logging.INFO)

    num_threads = 4
    max_evals = 500

    ackley = Ackley(dim=10)
    rbf = RBFInterpolant(
        dim=ackley.dim, kernel=CubicKernel(),
        tail=LinearTail(ackley.dim))
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Use the serial controller (uses only one thread)
    controller = ThreadController()
    controller.strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=rbf, asynchronous=True, batch_size=num_threads)

    print("Number of threads: {}".format(num_threads))
    print("Maximum number of evaluations: {}".format(max_evals))
    print("Strategy: {}".format(controller.strategy.__class__.__name__))
    print("Experimental design: {}".format(slhd.__class__.__name__))
    print("Surrogate: {}".format(rbf.__class__.__name__))

    # Launch the threads
    for _ in range(num_threads):
        try:
            worker = MatlabWorker(controller)
            worker.matlab = matlab.engine.start_matlab()
            controller.launch_worker(worker)
        except Exception as e:
            print("\nERROR: Failed to initialize a MATLAB session.\n")
            print(str(e))
            return

    # Run the optimization strategy
    result = controller.run()

    # Print the final result
    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    example_matlab_engine()
