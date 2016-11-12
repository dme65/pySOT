try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from experimental_design import *
from rbf import *
from rs_capped import RSCapped, RSUnitbox
from sampling_methods import *
from test_problems import *
from sot_sync_strategies import *
from ensemble_surrogate import EnsembleSurrogate
from poly_regression import *
from merit_functions import *

try:
    from GUI import GUI
    __with_gui__ = True
except ImportError:
    __with_gui__ = False

try:
    from mars_interpolant import MARSInterpolant
    __with_mars__ = True
except ImportError:
    __with_mars__ = False

__version__ = '0.1.29'
__author__ = 'David Eriksson, David Bindel'
