__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *
from spectralDNS.mesh.triplyperiodic import setup

vars().update(setup['MHD'](**vars()))

hdf5file = HDF5Writer(FFT, float, {"U":U[0], "V":U[1], "W":U[2], "P":P, 
                                   "Bx": B[0], "By": B[1], "Bz":B[2]}, "MHD.h5")

eta = float(params.eta)

def set_Elsasser(c, F_tmp, K):
    c[:3] = -1j*(K[0]*(F_tmp[:, 0] + F_tmp[0, :])
                +K[1]*(F_tmp[:, 1] + F_tmp[1, :])
                +K[2]*(F_tmp[:, 2] + F_tmp[2, :]))/2.0

    c[3:] =  1j*(K[0]*(F_tmp[0, :] - F_tmp[:, 0])
                +K[1]*(F_tmp[1, :] - F_tmp[:, 1])
                +K[2]*(F_tmp[2, :] - F_tmp[:, 2]))/2.0
    return c

def divergenceConvection(z0, z1, c, dealias=None):
    """Divergence convection using Elsasser variables
    z0=U+B
    z1=U-B
    """
    F_tmp = work[((3, 3) + FFT.complex_shape(), complex, 0)]
    for i in range(3):
        for j in range(3):
            F_tmp[i, j] = FFT.fftn(z0[i]*z1[j], F_tmp[i, j], dealias)
            
    c = set_Elsasser(c, F_tmp, K)
    #c[:3] = -1j*(K[0]*(F_tmp[:, 0] + F_tmp[0, :])
                #+K[1]*(F_tmp[:, 1] + F_tmp[1, :])
                #+K[2]*(F_tmp[:, 2] + F_tmp[2, :]))/2.0

    #c[3:] =  1j*(K[0]*(F_tmp[0, :] - F_tmp[:, 0])
                #+K[1]*(F_tmp[1, :] - F_tmp[:, 1])
                #+K[2]*(F_tmp[2, :] - F_tmp[:, 2]))/2.0

    return c    
    
def ComputeRHS(dU, UB_hat):
    """Compute and return entire rhs contribution"""
    UB_dealiased = work[((6,)+FFT.work_shape(params.dealias), float, 0)]
    for i in range(6):
        UB_dealiased[i] = FFT.ifftn(UB_hat[i], UB_dealiased[i], params.dealias)
    
    U_dealiased = UB_dealiased[:3]
    B_dealiased = UB_dealiased[3:]
    # Compute convective term and place in dU
    dU = divergenceConvection(U_dealiased+B_dealiased, U_dealiased-B_dealiased, dU, params.dealias)
    
    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat[:] = sum(dU[:3]*K_over_K2, 0)
        
    # Add pressure gradient
    dU[:3] -= P_hat*K

    # Add contribution from diffusion
    dU[:3] -= params.nu*K2*U_hat
    dU[3:] -= params.eta*K2*B_hat
    
    return dU

def regression_test(**kw):
    pass

def solve():
    global dU, UB, UB_hat
    
    timer = Timer()
    params.t = 0.0
    params.tstep = 0
    # Set up function to perform temporal integration (using params.integrator parameter)
    integrate = getintegrator(**globals())

    if params.make_profile: profiler = cProfile.Profile()
    
    dt_in = params.dt
    
    while params.t + params.dt <= params.T+1e-15:
        
        UB_hat, params.dt, dt_took = integrate()

        for i in range(6):
            UB[i] = FFT.ifftn(UB_hat[i], UB[i])
                 
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
    
    dU = ComputeRHS(dU, UB_hat)
    
    additional_callback(fU_hat=dU, **globals())

    timer.final(MPI, rank)
    
    if params.make_profile:
        results = create_profile(**vars())
        
    regression_test(**globals())
    
    hdf5file.close()
