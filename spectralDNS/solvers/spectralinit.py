__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unused-argument,redefined-outer-name,unused-import

import sys
import cProfile
import numpy as np
from mpi4py import MPI
from shenfun import CachedArrayDict as work_arrays
from spectralDNS import config
from spectralDNS.utilities import create_profile, MemoryUsage, Timer, reset_profile
from spectralDNS.h5io import HDF5File
from spectralDNS.optimization import optimizer
from spectralDNS.maths import cross1, cross2, project, getintegrator

comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
params = config.params
profiler = cProfile.Profile()

def datatypes(precision):
    """Return datatypes associated with precision."""
    assert precision in ("single", "double")
    return {"single": (np.float32, np.complex64, MPI.C_FLOAT_COMPLEX),
            "double": (np.float64, np.complex128, MPI.C_DOUBLE_COMPLEX)}[precision]

def regression_test(context):
    """Optional function called at the end"""
    pass

def update(context):
    """Optional function called every time step"""
    pass

def additional_callback(context):
    """Function used by some integrators"""
    pass

def solve_linear(context):
    """Function used by implicit solvers"""
    pass

def conv(*args):
    """Function used to compute convective term"""
    raise NotImplementedError

def set_source(Source, **context):
    """Return the source term"""
    Source[:] = 0
    return Source

def end_of_tstep(context):
    """Function called at end of time step.

    If returning True, the while-loop in time breaks free. Used by adaptive solvers
    to modify the time stepsize.
    """
    return False
