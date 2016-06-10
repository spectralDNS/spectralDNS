__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-01-02"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
Velocity-vorticity formulation
"""
from NS import *

# Get and update the global namespace of the NS solver (to avoid having two namespaces filled with arrays)
# Overload just a few routines

context = solve.func_globals
context.update(setup['VV'](**vars()))
vars().update(context)

hdf5file = HDF5Writer(FFT, float, {"U":U[0], "V":U[1], "W":U[2], "P":P}, "VV.h5")

def Curl(a, c, dealias=None):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    F_tmp = work[(a, 0)]
    F_tmp = cross2(F_tmp, K_over_K2, a)
    c[0] = FFT.ifftn(F_tmp[0], c[0], dealias)
    c[1] = FFT.ifftn(F_tmp[1], c[1], dealias)
    c[2] = FFT.ifftn(F_tmp[2], c[2], dealias)    
    return c

#@profile
def ComputeRHS(dU, W_hat):
    U_dealiased = work[((3,)+FFT.work_shape(params.dealias), float, 0)]
    W_dealiased = work[((3,)+FFT.work_shape(params.dealias), float, 1)]
    F_tmp = work[(dU, 0)]
    
    U_dealiased[:] = Curl(W_hat, U_dealiased, params.dealias)
    for i in range(3):
        W_dealiased[i] = FFT.ifftn(W_hat[i], W_dealiased[i], params.dealias)
    F_tmp[:] = Cross(U_dealiased, W_dealiased, F_tmp, params.dealias)
    dU = cross2(dU, K, F_tmp)    
    dU -= params.nu*K2*W_hat    
    dU += Source    
    return dU

def solve():
    global dU, W, W_hat, conv, integrate, profiler
    
    timer = Timer()
    params.t = 0.0
    params.tstep = 0
    # Set up function to perform temporal integration (using params.integrator parameter)
    integrate = getintegrator(**globals())
    conv = getConvection(params.convection)

    if params.make_profile: profiler = cProfile.Profile()

    dt_in = params.dt
    
    while params.t + params.dt <= params.T+1e-15:
        
        W_hat, params.dt, dt_took = integrate()

        for i in range(3):
            W[i] = FFT.ifftn(W_hat[i], W[i])

        params.t += dt_took
        params.tstep += 1
                 
        update(**globals())
        
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
    
    dU = ComputeRHS(dU, W_hat)
    
    additional_callback(fU_hat=dU, **globals())

    timer.final(MPI, rank)
    
    if params.make_profile:
        results = create_profile(**globals())
        
    regression_test(**globals())
        
    hdf5file.close()
