#!/usr/bin/env python3
from . import auxiliary_problems, controller, experimental_design, optimization_problems, strategy, surrogate

__version__ = "0.3.3"
__author__ = "David Eriksson, David Bindel, Christine Shoemaker"


__all__ = [
    "auxiliary_problems",
    "controller",
    "experimental_design",
    "optimization_problems",
    "strategy",
    "surrogate",
    # Other
    "__author__",
    "__version__",
]
