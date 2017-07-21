try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from pySOT.experimental_design import *
from pySOT.rbf import *
from pySOT.rs_wrappers import RSCapped, RSUnitbox
from pySOT.adaptive_sampling import *
from pySOT.test_problems import *
from pySOT.sot_sync_strategies import *
from pySOT.ensemble_surrogate import EnsembleSurrogate
from pySOT.poly_regression import *
from pySOT.merit_functions import *
from pySOT.utils import *

try:
    from pySOT.mars_interpolant import MARSInterpolant
    __with_mars__ = True
except ImportError as err:
    __with_mars__ = False

try:
    from pySOT.gp_regression import GPRegression
    __with_gp__ = True
except ImportError as err:
    __with_gp__ = False

__version__ = '0.1.36'
__author__ = 'David Eriksson, David Bindel, Christine Shoemaker'
