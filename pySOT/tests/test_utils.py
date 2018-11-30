import numpy as np
from pySOT.utils import unit_rescale, from_unit_box, \
    to_unit_box, round_vars, GeneticAlgorithm


def test_unit_box_map():
    X = np.random.rand(5, 3)
    lb = -1 * np.ones((3,))
    ub = 2 * np.ones((3,))

    X1 = to_unit_box(X, lb, ub)
    np.testing.assert_equal(X.shape, X1.shape)
    np.testing.assert_almost_equal(X1, to_unit_box(X, np.atleast_2d(lb), np.atleast_2d(ub)))
    assert(X.max() <= 1.0 and X.min() >= 0)

    # Try to map back to what we started with
    X2 = from_unit_box(X1, lb, ub)
    np.testing.assert_equal(X.shape, X2.shape)
    np.testing.assert_almost_equal(X2, from_unit_box(X1, np.atleast_2d(lb), np.atleast_2d(ub)))
    np.testing.assert_almost_equal(X2, X)


def test_unit_rescale():
    X = np.random.rand(5, 3)
    X1 = unit_rescale(X)
    np.testing.assert_equal(X.shape, X1.shape)
    np.testing.assert_almost_equal(X1.max(), 1.0)
    np.testing.assert_almost_equal(X1.min(), 0.0)

    X = X.flatten()  # Test for 1D array as well
    X1 = unit_rescale(X)
    np.testing.assert_equal(X.shape, X1.shape)
    np.testing.assert_almost_equal(X1.max(), 1.0)
    np.testing.assert_almost_equal(X1.min(), 0.0)

    X = 0.5 * np.ones((5, 3))
    X1 = unit_rescale(X)
    np.testing.assert_equal(X.shape, X1.shape)
    np.testing.assert_almost_equal(X1.max(), 1.0)
    np.testing.assert_almost_equal(X1.min(), 1.0)


def test_round_vars():
    X = np.random.rand(5, 4)
    cont_var = np.array([1, 3])
    int_var = np.array([0, 2])
    lb = np.zeros((3,))
    ub = np.ones((3,))
    X1 = round_vars(X, int_var, lb, ub)
    np.testing.assert_equal(X.shape, X1.shape)
    np.testing.assert_almost_equal(X1[:, int_var], np.round(X[:, int_var]))
    np.testing.assert_almost_equal(X1[:, cont_var], X[:, cont_var])


def test_ga():
    dim = 10

    # Vectorized Ackley function in dim dimensions
    def obj_function(x):
        return -20.0*np.exp(-0.2*np.sqrt(np.sum(x**2, axis=1)/dim)) - \
            np.exp(np.sum(np.cos(2.0*np.pi*x), axis=1)/dim) + 20 + np.exp(1)

    ga = GeneticAlgorithm(obj_function, dim, -15*np.ones(dim), 20*np.ones(dim),
                          pop_size=100, num_gen=100, start="SLHD")
    x_best, f_best = ga.optimize()

    ga = GeneticAlgorithm(obj_function, dim, -15*np.ones(dim), 20*np.ones(dim), np.array([0]),
                          pop_size=100, num_gen=100, start="SLHD")
    x_best, f_best = ga.optimize()
    np.testing.assert_almost_equal(x_best[0], np.round(x_best[0]))

if __name__ == '__main__':
    test_ga()
    test_round_vars()
    test_unit_box_map()
    test_unit_rescale()