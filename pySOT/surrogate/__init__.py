from .gp import GPRegressor
from .kernels import CubicKernel, Kernel, LinearKernel, TPSKernel
from .mars import MARSInterpolant
from .poly import PolyRegressor
from .rbf import RBFInterpolant
from .surrogate import Surrogate
from .surrogate_capped import SurrogateCapped
from .surrogate_unit_box import SurrogateUnitBox
from .tails import ConstantTail, LinearTail, Tail

__all__ = [
    "Surrogate",
    "GPRegressor",
    "MARSInterpolant",
    "PolyRegressor",
    "RBFInterpolant",
    #
    "SurrogateCapped",
    "SurrogateUnitBox",
    #
    "Kernel",
    "CubicKernel",
    "LinearKernel",
    "TPSKernel",
    #
    "Tail",
    "ConstantTail",
    "LinearTail",
]
