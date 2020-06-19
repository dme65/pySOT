#!/usr/bin/env python3
from .constant_tail import ConstantTail
from .linear_tail import LinearTail
from .tail import Tail

__all__ = [
    "Tail",
    "ConstantTail",
    "LinearTail",
]
