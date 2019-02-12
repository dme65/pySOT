"""
.. module:: example_subprocess_files
  :synopsis: Example of an external objective function with input text files
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, TPSKernel, LinearTail
from pySOT.optimization_problems import Sphere

from poap.controller import ThreadController, ProcessWorkerThread
import numpy as np
import os.path
import logging
from subprocess import Popen, PIPE


def array2str(x):
    return ",".join(np.char.mod('%f', x))


# Find path of the executable
path = os.path.dirname(os.path.abspath(__file__)) + "/sphere_ext_files"


class CppSim(ProcessWorkerThread):
    def handle_eval(self, record):
        try:
            # Print to the input file
            f = open(self.my_filename, 'w')
            f.write(array2str(record.params[0]))
            f.close()

            self.process = Popen([path, self.my_filename], stdout=PIPE,
                                 bufsize=1, universal_newlines=True)
            val = self.process.communicate()[0]

            self.finish_success(record, float(val))
            os.remove(self.my_filename)  # Remove input file
        except ValueError:
            logging.info("WARNING: Incorrect output or crashed evaluation")
            os.remove(self.my_filename)  # Remove input file
            self.finish_cancelled(record)


def example_subprocess_files():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/example_subprocess_files.log"):
        os.remove("./logfiles/example_subprocess_files.log")
    logging.basicConfig(filename="./logfiles/example_subprocess_files.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 200")
    print("Sampling method: Candidate DYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    assert os.path.isfile(path), "You need to build sphere_ext_files"

    num_threads = 4
    max_evals = 200

    sphere = Sphere(dim=10)
    rbf = RBFInterpolant(dim=sphere.dim, kernel=TPSKernel(),
                         tail=LinearTail(sphere.dim))
    slhd = SymmetricLatinHypercube(
        dim=sphere.dim, num_pts=2*(sphere.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=sphere, exp_design=slhd,
        surrogate=rbf, asynchronous=False, batch_size=num_threads)

    print("Number of threads: {}".format(num_threads))
    print("Maximum number of evaluations: {}".format(max_evals))
    print("Strategy: {}".format(controller.strategy.__class__.__name__))
    print("Experimental design: {}".format(slhd.__class__.__name__))
    print("Surrogate: {}".format(rbf.__class__.__name__))

    # Launch the threads and give them access to the objective function
    for i in range(num_threads):
        worker = CppSim(controller)
        worker.my_filename = str(i) + ".txt"
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    example_subprocess_files()
