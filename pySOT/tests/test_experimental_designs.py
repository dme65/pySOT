from pySOT.experimental_design import *
import inspect
import sys


def test_lhd():
    lhd = LatinHypercube(4, 10, criterion='c')
    X = lhd.generate_points()
    assert (isinstance(lhd, ExperimentalDesign))
    assert (np.all(X.shape == (10, 4)))
    assert (lhd.npts() == 10)
    assert (lhd.dim() == 4)


def test_slhd():
    for i in range(10, 12):  # To test even and odd
        print(i)
        slhd = SymmetricLatinHypercube(3, i)
        X = slhd.generate_points()
        assert(isinstance(slhd, ExperimentalDesign))
        assert(np.all(X.shape == (i, 3)))
        assert (slhd.npts() == i)
        assert (slhd.dim() == 3)


def test_box_behnken():
    bb = BoxBehnken(3)
    X = bb.generate_points()
    assert(isinstance(bb, ExperimentalDesign))
    assert(np.all(X.shape == (13, 3)))
    assert (bb.npts() == 13)
    assert (bb.dim() == 3)

    try:



def test_full_factorial():
    ff = TwoFactorial(3)
    X = ff.generate_points()
    assert (isinstance(ff, ExperimentalDesign))
    assert (np.all(X.shape == (8, 3)))
    assert (ff.npts() == 8)
    assert (ff.dim() == 3)
