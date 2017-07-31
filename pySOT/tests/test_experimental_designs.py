from pySOT.experimental_design import *
import inspect
import sys


def test_lhd():
    lhd = LatinHypercube(4, 10, criterion='c')
    X = lhd.generate_points()


def test_slhd():
    slhd = SymmetricLatinHypercube(3, 10)
    X = slhd.generate_points()


if __name__ == "__main__":
    test_lhd()
    test_slhd()
