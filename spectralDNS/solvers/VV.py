__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-01-02"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
Velocity-vorticity formulation
"""
from spectralinit import *
from spectralDNS.mesh.triplyperiodic import setup

vars().update(setup['VV'](**vars()))

hdf5file = HDF5Writer(FFT, float, {"U":U[0], "V":U[1], "W":U[2], "P":P}, "VV.h5")

def standardConvection(c, U_dealiased, U_hat, dealias=None):
    """c_i = u_j du_i/dx_j"""
    gradUi = work[(U_dealiased, 2)]
    for i in range(3):
        for j in range(3):
            gradUi[j] = FFT.ifftn(1j*K[j]*U_hat[i], gradUi[j], dealias)
        c[i] = FFT.fftn(sum(U_dealiased*gradUi, 0), c[i], dealias)
    return c

def divergenceConvection(c, U_dealiased, dealias=None, add=False):
    """c_i = div(u_i u_j)"""
    if not add: c.fill(0)
    UUi_hat = work[(c, 0)]
    for i in range(3):
        UUi_hat[i] = FFT.fftn(U_dealiased[0]*U_dealiased[i], UUi_hat[i], dealias)
    c[0] += 1j*sum(K*UUi_hat, 0)
    c[1] += 1j*K[0]*UUi_hat[1]
    c[2] += 1j*K[0]*UUi_hat[2]
    UUi_hat[0] = FFT.fftn(U_dealiased[1]*U_dealiased[1], UUi_hat[0], dealias)
    UUi_hat[1] = FFT.fftn(U_dealiased[1]*U_dealiased[2], UUi_hat[1], dealias)
    UUi_hat[2] = FFT.fftn(U_dealiased[2]*U_dealiased[2], UUi_hat[2], dealias)
    c[1] += (1j*K[1]*UUi_hat[0] + 1j*K[2]*UUi_hat[1])
    c[2] += (1j*K[1]*UUi_hat[1] + 1j*K[2]*UUi_hat[2])
    return c

#@profile
def Cross(a, b, c, dealias=None):
    """c_k = F_k(a x b)"""
    Uc = work[(a, 2)]
    Uc = cross1(Uc, a, b)
    c[0] = FFT.fftn(Uc[0], c[0], dealias)
    c[1] = FFT.fftn(Uc[1], c[1], dealias)
    c[2] = FFT.fftn(Uc[2], c[2], dealias)
    return c

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
    global dU, W, W_hat
    
    timer = Timer()
    params.t = 0.0
    params.tstep = 0
    # Set up function to perform temporal integration (using params.integrator parameter)
    integrate = getintegrator(**globals())

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
