"""
.. module:: mpiexample_subprocess_mpi
  :synopsis: Example of an external objective function with MPI
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Sphere

from poap.mpiserve import MPIController, MPIProcessWorker
import numpy as np
import sys
import os.path
import logging
from subprocess import Popen, PIPE

# Try to import mpi4py
try:
    from mpi4py import MPI
except Exception as err:
    print("ERROR: You need mpi4py to use the POAP MPI controller")
    exit()


def array2str(x):
    return ",".join(np.char.mod('%f', x))


# Find path of the executable
path = os.path.dirname(os.path.abspath(__file__)) + "/sphere_ext"


class CppSim(MPIProcessWorker):
    def eval(self, record_id, params, extra_args=None):
        try:
            self.process = Popen([
                path, array2str(params[0])], stdout=PIPE,
                bufsize=1, universal_newlines=True)
            val = self.process.communicate()[0]
            self.finish_success(record_id, float(val))
        except ValueError:
            logging.info("WARNING: Incorrect output or crashed evaluation")
            self.finish_cancel(record_id)


def main_worker():
    logging.basicConfig(filename="./logfiles/test_subprocess_mpi.log",
                        level=logging.INFO)
    CppSim().run()


def main_master(num_workers):
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_subprocess_mpi.log"):
        os.remove("./logfiles/test_subprocess_mpi.log")
    logging.basicConfig(filename="./logfiles/test_subprocess_mpi.log",
                        level=logging.INFO)

    print("\nTesting the POAP MPI controller with {0} workers".format(
        num_workers))
    print("Maximum number of evaluations: 200")
    print("Search strategy: Candidate DYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    assert os.path.isfile(path), "You need to build sphere_ext"

    max_evals = 200

    sphere = Sphere(dim=10)
    rbf = RBFInterpolant(dim=sphere.dim, kernel=CubicKernel(),
                         tail=LinearTail(sphere.dim))
    slhd = SymmetricLatinHypercube(
        dim=sphere.dim, num_pts=2*(sphere.dim+1))

    # Create a strategy and a controller
    strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=sphere, exp_design=slhd,
        surrogate=rbf, asynchronous=True, batch_size=num_workers)
    controller = MPIController(strategy)

    print("Number of threads: {}".format(num_workers))
    print("Maximum number of evaluations: {}".format(max_evals))
    print("Strategy: {}".format(controller.strategy.__class__.__name__))
    print("Experimental design: {}".format(slhd.__class__.__name__))
    print("Surrogate: {}".format(rbf.__class__.__name__))

    # Run the optimization strategy
    result = controller.run()
    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


def mpiexample_subprocess_mpi():
    # Extract the rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if rank == 0:
        main_master(nprocs)
    else:
        main_worker()


if __name__ == '__main__':
    mpiexample_subprocess_mpi()
