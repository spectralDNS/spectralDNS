__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpi4py import MPI
import sys, cProfile
import numpy as np
# possibly byte-aligned zeros/empty
from mpiFFT4py import Slab_R2C, Pencil_R2C, Line_R2C, empty, zeros, \
     work_arrays, datatypes    
from spectralDNS import config
from spectralDNS.utilities import create_profile, MemoryUsage, Timer, reset_profile
from spectralDNS.h5io import HDF5Writer
from spectralDNS.optimization import optimizer
from spectralDNS.maths import cross1, cross2, project, getintegrator

comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
params = config.params

def get_FFT(params):
    """Return instance of class for performing transformations"""
    if params.decomposition == 'slab':
        assert len(params.N) == 3
        assert len(params.L) == 3
        FFT = Slab_R2C(params.N, params.L, MPI, params.precision, 
                       communication=params.communication, 
                       threads=params.threads,
                       planner_effort=params.planner_effort)
        
    elif params.decomposition == 'pencil':
        assert len(params.N) == 3
        assert len(params.L) == 3
        FFT = Pencil_R2C(params.N, params.L, MPI, params.precision, P1=params.Pencil_P1, 
                         communication=params.communication, threads=params.threads,
                         alignment=params.Pencil_alignment,
                         planner_effort=params.planner_effort)
            
    elif params.decomposition == 'line':
        assert len(params.N) == 2
        assert len(params.L) == 2
        FFT = Line_R2C(params.N, params.L, MPI, params.precision,
                       threads=params.threads,
                       planner_effort=params.planner_effort)
    return FFT

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

class SolverBase(object):
    """Assemble and solve rhs of spectral Navier Stokes equations"""
    
    def __init__(self):
        self._conv = None

    @staticmethod
    def _getConvection(conv_type):
        pass

    def nonlinear(self, *args):
        """Compute contribution to rhs from nonlinear term

        Since there may be many different ways of computing the nonlinear
        term, the actual method is here chosen using a parameter set 
        externally:

            config.params.convection

        To avoid costly if tests, the function to use is collected
        dynamically the first time nonlinear is called. Overload only
        required for _getConvection in subclasses.
        """
        try:
            return self._conv(*args)

        except TypeError:
            self._conv = self._getConvection(params.convection)
            return self._conv(*args)

    @staticmethod
    def add_linear(rhs, u_hat, *args):
        """Add contributions from linear terms to the rhs
        
        args:
            rhs         The right hand side to be returned
            u_hat       The solution at current time. May differ from the primary
                        variable (see setup) since it is set by the integrator

        """
        return rhs

    def __call__(self, rhs, u_hat, **context):
        """Return right hand side of Navier Stokes
        
        args:
            rhs         The right hand side to be returned
            u_hat       The FFT of the velocity at current time. May differ from
                        context.U_hat since it is set by the integrator

            **context   The solvers context
        """
        rhs = self.nonlinear(rhs, u_hat, **context)
        rhs = self.add_linear(rhs, u_hat, **context)
        return rhs
