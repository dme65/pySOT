"""
.. module:: test_simple_mpi
  :synopsis: Test Simple MPI
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from mpi4py import MPI
from pySOT import *
from poap.mpiserve import MPIController, MPISimpleWorker
import numpy as np
import os.path


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
    print("Sampling method: CandidateDYCORS, with weight 0.5")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF, domain scaled to unit box")

    maxeval = 500
    print(data.info)

    # Create a strategy and a controller
    strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nworkers,
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail,
                                            maxp=maxeval, dim=data.dim),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim, weights=[0.5]))
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
