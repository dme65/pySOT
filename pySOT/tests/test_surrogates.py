from pySOT.surrogate import *
import inspect
import sys


def f(x):
    """Test function"""
    fx = x[:, 1] * np.sin(x[:, 0]) + x[:, 0] * np.cos(x[:, 1])
    return fx


def df(x):
    """Derivative of test function"""
    dfx = np.zeros((x.shape[0], 2))
    dfx[:, 0] = x[:, 1] * np.cos(x[:, 0]) + np.cos(x[:, 1])
    dfx[:, 1] = np.sin(x[:, 0]) - x[:, 0] * np.sin(x[:, 1])
    return dfx


def test_rbf():
    kernel = CubicKernel()
    tail = LinearTail(2)
    fhat = RBFInterpolant(2, kernel, tail, 500)
    xx = np.random.rand(120, 2)
    fx = f(xx)
    fhat.add_points(xx[:100, :], fx[:100])
    fhat.eval(xx)  # Trigger initial fit
    fhat.add_points(xx[100:, :], fx[100:])

    xs = np.random.rand(20, 2)
    fhx = fhat.eval(xs)
    fx = f(xs)
    dfx = df(xs)
    for i in range(20):
        dfhx = fhat.deriv(xs[i, :])
        print("Err (interp): %e : %e" % (abs(fx[i] - fhx[i]) / abs(fx[i]),
                                         la.norm(dfx[i, :] - dfhx) / la.norm(dfx[i, :])))

if __name__ == "__main__":
    test_rbf()
