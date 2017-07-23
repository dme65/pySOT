"""
.. module:: test_simple_mpi
  :synopsis: Test Simple MPI
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import Ackley, SyncStrategyNoConstraints, \
    SymmetricLatinHypercube, RBFInterpolant, CubicKernel, \
    LinearTail, CandidateDYCORS
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


def main_master(data, nworkers):
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_simple_mpi.log"):
        os.remove("./logfiles/test_simple_mpi.log")
    logging.basicConfig(filename="./logfiles/test_simple_mpi.log",
                        level=logging.INFO)

    print("\nTesting the POAP MPI controller with {0} workers".format(nworkers))
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    maxeval = 500
    print(data.info)

    # Create a strategy and a controller
    strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nworkers,
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail,
                                            maxp=maxeval),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim))
    controller = MPIController(strategy)

    result = controller.run()
    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    # Optimization problem
    data = Ackley(dim=10)

    # Extract the rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if rank == 0:
        main_master(data, nprocs)
    else:
        main_worker(data.objfunction)
