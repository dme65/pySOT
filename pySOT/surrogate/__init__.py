#!/usr/bin/env python3
from .gp import GPRegressor
from .kernels import CubicKernel, Kernel, LinearKernel, TPSKernel
from .mars import MARSInterpolant
from .output_transformations import identity, median_capping
from .poly import PolyRegressor
from .rbf import RBFInterpolant
from .surrogate import Surrogate
from .tails import ConstantTail, LinearTail, Tail

__all__ = [
    "Surrogate",
    "GPRegressor",
    "MARSInterpolant",
    "PolyRegressor",
    "RBFInterpolant",
    #
    "Kernel",
    "CubicKernel",
    "LinearKernel",
    "TPSKernel",
    #
    "Tail",
    "ConstantTail",
    "LinearTail",
    #
    "identity",
    "median_capping",
]
