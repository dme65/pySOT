from pySOT.experimental_design import ExperimentalDesign, \
        SymmetricLatinHypercube, LatinHypercube, TwoFactorial
import numpy as np
import pytest


def test_lhd():
    lhd = LatinHypercube(dim=4, num_pts=10, criterion='c')
    X = lhd.generate_points()
    assert (isinstance(lhd, ExperimentalDesign))
    assert (np.all(X.shape == (10, 4)))
    assert (lhd.num_pts == 10)
    assert (lhd.dim == 4)


def test_slhd():
    for i in range(10, 12):  # To test even and odd
        print(i)
        slhd = SymmetricLatinHypercube(dim=3, num_pts=i)
        X = slhd.generate_points()
        assert(isinstance(slhd, ExperimentalDesign))
        assert(np.all(X.shape == (i, 3)))
        assert (slhd.num_pts == i)
        assert (slhd.dim == 3)


def test_full_factorial():
    ff = TwoFactorial(dim=3)
    X = ff.generate_points()
    assert (isinstance(ff, ExperimentalDesign))
    assert (np.all(X.shape == (8, 3)))
    assert (ff.num_pts == 8)
    assert (ff.dim == 3)

    with pytest.raises(ValueError):  # This should raise an exception
        TwoFactorial(20)

if __name__ == '__main__':
    test_full_factorial()
    test_lhd()
    test_slhd()
