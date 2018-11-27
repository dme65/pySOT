"""
.. module:: mpiexample_simple_mpi
  :synopsis: Simple MPI example
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Ackley

from poap.mpiserve import MPIController, MPISimpleWorker
import numpy as np
import os.path
import logging

# Try to import mpi4py
try:
    from mpi4py import MPI
except Exception as err:
    print("ERROR: You need mpi4py to use the POAP MPI controller.")
    exit()


def main_worker(objfunction):
    MPISimpleWorker(objfunction).run()


def main_master(opt_prob, nworkers):
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/mpiexample_mpi.log"):
        os.remove("./logfiles/mpiexample_mpi.log")
    logging.basicConfig(filename="./logfiles/mpiexample_mpi.log",
                        level=logging.INFO)

    print("\nTesting the POAP MPI controller with {0} workers".format(nworkers))
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    max_evals = 500
    print(opt_prob.info)

    rbf = RBFInterpolant(
        dim=opt_prob.dim, kernel=CubicKernel(), tail=LinearTail(opt_prob.dim))
    slhd = SymmetricLatinHypercube(
        dim=opt_prob.dim, num_pts=2*(opt_prob.dim+1))

    # Create a strategy and a controller
    strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=opt_prob, exp_design=slhd, 
        surrogate=rbf, asynchronous=True, batch_size=nworkers)
    controller = MPIController(strategy)

    result = controller.run()
    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


def mpiexample_simple():
    # Optimization problem
    ackley = Ackley(dim=10)

    # Extract the rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if rank == 0:
        main_master(ackley, nprocs)
    else:
        main_worker(ackley.eval)


if __name__ == '__main__':
    mpiexample_simple()
