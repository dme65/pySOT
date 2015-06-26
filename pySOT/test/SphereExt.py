from poap.controller import ProcessWorkerThread
import numpy as np
from subprocess import Popen, PIPE
import os

def array2str(x):
    return ",".join(np.char.mod('%f', x))

class SphereExt:
    def __init__(self, dim=10):
        self.xlow = -15 * np.ones(dim)
        self.xup = 20 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Sphere function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        assert os.path.isfile("./test/sphere_ext"), "You need to build sphere_ext" \
            " or specify another path and/or filename"


class objfunction(ProcessWorkerThread):

    def handle_eval(self, record):
        self.process = Popen(['./test/sphere_ext', array2str(record.params[0])],
                             stdout=PIPE)
        out = self.process.communicate()[0]
        try:
            val = float(out)  # This raises ValueError if out is not a float
            self.finish_success(record, val)
        except ValueError:
            self.finish_failure(record)