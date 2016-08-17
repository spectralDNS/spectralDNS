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

def set_source(Source, **context):
    Source[:] = 0
    return Source

def solve(solver, context):
    """Solve Navier Stokes equations

    args:
        context      The solver's context from setup()

    global args:
        params       Dictionary (config.params) of parameters
                     that control the integration.
                     See spectralDNS.config for details
    """

    global integrate, profiler, timer
    
    # Link to calling solver (NS or VV) module
    #solver = config.solver

    timer = Timer()
    params.t = 0.0
    params.tstep = 0
    
    integrate = solver.getintegrator(solver.ComputeRHS, 
                                     context.dU,
                                     context.u, # primary variable
                                     params,
                                     context,
                                     solver.additional_callback)

    solver.conv = solver.getConvection(params.convection)

    if params.make_profile: profiler = cProfile.Profile()

    dt_in = params.dt

    while params.t + params.dt <= params.T+1e-15:

        u, params.dt, dt_took = integrate()

        params.t += dt_took
        params.tstep += 1

        solver.update(context)

        context.hdf5file.update(params, **context)

        timer()

        if params.tstep == 1 and params.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()

        #Make sure that the last step hits T exactly.
        if params.t + params.dt >= params.T:
            params.dt = params.T - params.t
            if params.dt <= 1.e-14:
                break

    params.dt = dt_in

    dU = solver.ComputeRHS(context.dU, u, **context)

    solver.additional_callback(fU_hat=dU, **context)

    timer.final(MPI, rank)

    if params.make_profile:
        results = create_profile(**globals())

    solver.regression_test(context)

    context.hdf5file.close()

