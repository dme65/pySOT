"""
.. module:: test_subprocess_files
  :synopsis: Test an external objective function
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

import logging
from pySOT import *
from poap.controller import ThreadController, ProcessWorkerThread
import numpy as np
from subprocess import Popen, PIPE
import os.path
import threading


def array2str(x):
    return ",".join(np.char.mod('%f', x))


class FilenameGenerator:
    """
    Generates an integer that can be used as a file name or directory name.
    This object is both thread safe, i.e., it will never fail because two
    threads are trying to obtain a file name at the same time. It also guarantees
    that no threads ever recieves a file name that has been assigned since the
    object was initialized. It assigns integers 0, 1, 2, ... to the threads in the
    order they request a new filename.
    """
    def __init__(self):
        self.next_name = 0
        self.lock = threading.Lock()

    def next_filename(self):
        self.lock.acquire()
        new_name = self.next_name
        self.next_name += 1
        self.lock.release()
        return new_name


class SphereExtFiles:
    def __init__(self, dim=10):
        self.xlow = -15 * np.ones(dim)
        self.xup = 20 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Sphere function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)

# This object is responsible for generating the new filename in a thread safe fashion
my_gen = FilenameGenerator()


class DummySim(ProcessWorkerThread):

    def handle_eval(self, record):
        # This gives a file name / directory name that no other thread can use
        my_unique_filename = my_gen.next_filename()
        my_unique_filename = str(my_unique_filename) + ".txt"

        # Print to the input file
        f = open(my_unique_filename, 'w')
        f.write(array2str(record.params[0]))
        f.close()

        # Run the objective function and pass the filename of the input file
        self.process = Popen(['./sphere_ext_files', my_unique_filename], stdout=PIPE)
        out = self.process.communicate()[0]

        # Parse the output
        try:
            val = float(out)  # This raises ValueError if out is not a float
            self.finish_success(record, val)
            os.remove(my_unique_filename)  # Remove input file
        except ValueError:
            logging.warning("Function evaluation crashed/failed")
            self.finish_failure(record)
            os.remove(my_unique_filename)  # Remove input file


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    logging.basicConfig(filename="./logfiles/test_subprocess_files.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 200")
    print("Search strategy: Candidate DyCORS")
    print("Experimental design: Latin Hypercube")
    print("Ensemble surrogates: Cubic RBF")

    assert os.path.isfile("./sphere_ext_files"), "You need to build sphere_ext"

    nthreads = 4
    maxeval = 200
    nsamples = nthreads

    data = SphereExtFiles(dim=10)
    print(data.info)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=LatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            search_procedure=CandidateDYCORS(data=data, numcand=100*data.dim),
            response_surface=RBFInterpolant(surftype=CubicRBFSurface, maxp=maxeval))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        controller.launch_worker(DummySim(controller))

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))

if __name__ == '__main__':
    main()
