import numpy as np
from pySOT.utils import unit_rescale, from_unit_box, \
    to_unit_box, round_vars, GeneticAlgorithm,\
    nd_sorting, nd_front, check_radius_rule, POSITIVE_INFINITY


def test_unit_box_map():
    X = np.random.rand(5, 3)
    lb = -1 * np.ones((3,))
    ub = 2 * np.ones((3,))

    X1 = to_unit_box(X, lb, ub)
    np.testing.assert_equal(X.shape, X1.shape)
    np.testing.assert_almost_equal(
        X1, to_unit_box(X, np.atleast_2d(lb), np.atleast_2d(ub)))
    assert(X.max() <= 1.0 and X.min() >= 0)

    # Try to map back to what we started with
    X2 = from_unit_box(X1, lb, ub)
    np.testing.assert_equal(X.shape, X2.shape)
    np.testing.assert_almost_equal(
        X2, from_unit_box(X1, np.atleast_2d(lb), np.atleast_2d(ub)))
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

    ga = GeneticAlgorithm(
        obj_function, dim, -15*np.ones(dim), 20*np.ones(dim),
        pop_size=100, num_gen=100, start="SLHD")
    x_best, f_best = ga.optimize()

    ga = GeneticAlgorithm(
        obj_function, dim, -15*np.ones(dim), 20*np.ones(dim), np.array([0]),
        pop_size=100, num_gen=100, start="SLHD")
    x_best, f_best = ga.optimize()
    np.testing.assert_almost_equal(x_best[0], np.round(x_best[0]))


def test_nd_front():
    npts = 100
    nobj = 2
    F = np.random.rand(nobj, npts)
    (nd_index, d_index) = nd_front(F)

    # check sum of indices equals the number of pts
    assert(len(nd_index)+len(d_index) == npts)

    # check if all index refereces to non-dom and dom pts are unique
    assert(len(nd_index) == len(set(nd_index)))
    assert(len(d_index) == len(set(d_index)))

    # check if a better point is added to set it dominates all others
    new_p = np.asarray([-0.1, -0.5])
    F_new = np.vstack((F.transpose(), new_p))
    (nd_index, d_index) = nd_front(F_new.transpose())
    assert(len(nd_index) == 1 and nd_index[0] == npts)

    # check if a worst point is added to set it is dominated
    (nd_index, d_index) = nd_front(F)
    new_p = np.asarray([1.1, 1.4])
    npts_nd = len(nd_index)
    F_new = np.vstack((F[:, nd_index].transpose(), new_p))
    (nd_index, d_index) = nd_front(F_new.transpose())
    assert(len(d_index) == 1 and d_index[0] == npts_nd)


def test_nd_sorting():
    npts = 100
    nmax = npts
    nobj = 2
    F = np.random.rand(nobj, npts)
    ranks = nd_sorting(F, nmax)
    # make sure that every point has a rank
    assert(len(ranks) == npts)
    # make sure that number of ranks = maximum rank
    assert(len(set(ranks)) == int(max(ranks)))

    # check if nmax < npts, then atleast nmax points are ranked
    npts = 200
    nmax = 150
    nobj = 2
    F = np.random.rand(nobj, npts)
    ranks = nd_sorting(F, nmax)
    assert(list(ranks).count(POSITIVE_INFINITY) <= npts - nmax)


def test_radius_rules():
    dim = 2
    nc = 32
    sigma = 0.2
    X_c = np.zeros((nc, dim+5))
    X_c[:, 0:dim] = np.random.rand(nc, dim)
    X = np.random.rand(1, dim)
    d_thresh = 0.7

    # ensure that radius rule functions return a 0 or 1
    flag = check_radius_rule(X, X_c, sigma, dim, nc, d_thresh)
    assert(flag == 0 or flag == 1)

    # ensure that flag is 1 if d_thresh = 0
    d_thresh = 0.0
    flag = check_radius_rule(X, X_c, sigma, dim, nc, d_thresh)
    assert(flag == 1)

    # ensure that flag is 1 if X is in X_c (i.e., via radius rule)
    d_thresh = 0.7
    X = X_c[15, 0:dim]
    flag = check_radius_rule(X, X_c, sigma, dim, nc, d_thresh)
    assert(flag == 0)


if __name__ == '__main__':
    test_ga()
    test_round_vars()
    test_unit_box_map()
    test_unit_rescale()
    test_nd_front()
    test_nd_sorting()
    test_radius_rules()
