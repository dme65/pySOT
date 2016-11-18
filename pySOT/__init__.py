try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from experimental_design import *
from rbf import *
from rs_wrappers import RSCapped, RSUnitbox
from adaptive_sampling import *
from test_problems import *
from sot_sync_strategies import *
from ensemble_surrogate import EnsembleSurrogate
from poly_regression import *
from merit_functions import *
from utils import *

try:
    from gui import GUI
    __with_gui__ = True
except ImportError:
    __with_gui__ = False

try:
    from mars_interpolant import MARSInterpolant
    __with_mars__ = True
except ImportError:
    __with_mars__ = False

__version__ = '0.1.30'
__author__ = 'David Eriksson, David Bindel, Christine Shoemaker'
