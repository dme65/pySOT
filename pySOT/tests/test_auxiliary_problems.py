from pySOT.auxiliary_problems import weighted_distance_merit, ei_merit, \
    candidate_dycors, candidate_srbf, candidate_uniform, expected_improvement_ga, \
    expected_improvement_uniform
from pySOT.surrogate import GPRegressor
from pySOT.optimization_problems import Ackley
import numpy as np


def test_ei():
    np.random.seed(0)
    ackley = Ackley(dim=1)
    X = np.expand_dims([-15, -10, 0, 1, 20], axis=1)
    fX = np.array([ackley.eval(x) for x in X])

    gp = GPRegressor(dim=1)
    gp.add_points(X, fX)
        
    # Find the global optimizer of EI
    x_true = -3.0556
    x_best = expected_improvement_ga(
        X=X, Xpend=None, dtol=0.0, ei_tol=0, 
        fX=fX, num_pts=1, opt_prob=ackley, surrogate=gp)
    assert np.isclose(x_best, x_true, atol=1e-2)

    x_best = expected_improvement_uniform(
        X=X, Xpend=None, dtol=0.0, ei_tol=0, 
        fX=fX, num_pts=1, opt_prob=ackley, surrogate=gp,
        num_cand=10000)
    assert np.isclose(x_best, x_true, atol=1e-2)

    # Find the optimizer at least distance 5 from other points
    x_true = 11.14
    x_best = expected_improvement_ga(
        X=X, Xpend=None, dtol=5.0, ei_tol=0, 
        fX=fX, num_pts=1, opt_prob=ackley, surrogate=gp)
    assert np.isclose(x_best, x_true, atol=1e-2)

    x_best = expected_improvement_uniform(
        X=X, Xpend=None, dtol=5.0, ei_tol=0, 
        fX=fX, num_pts=1, opt_prob=ackley, surrogate=gp,
        num_cand=10000)
    assert np.isclose(x_best, x_true, atol=1e-2)    


if __name__ == '__main__':
    test_ei()