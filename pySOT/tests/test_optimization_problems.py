import numpy as np
import pySOT.optimization_problems
from pySOT.optimization_problems import OptimizationProblem
import inspect
import pytest
import importlib
from pySOT.utils import check_opt_prob


def test_all():
    module = importlib.import_module("pySOT.optimization_problems")
    for name, obj in inspect.getmembers(pySOT.optimization_problems):
        if inspect.isclass(obj) and name != "OptimizationProblem":
            opt = getattr(module, name)
            opt = opt()
            assert (isinstance(opt, OptimizationProblem))

            if hasattr(opt, 'minimum'):
                val = opt.eval(opt.minimum)
                assert(abs(val - opt.min) < 1e-3)
            else:
                val = opt.eval(np.zeros(opt.dim))
            with pytest.raises(ValueError):  # This should raise an exception
                opt.eval(np.zeros(opt.dim + 1))

            # Sanity check all methods
            check_opt_prob(opt)


if __name__ == '__main__':
    test_all()
