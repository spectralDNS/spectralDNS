__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *
from spectralDNS.mesh.doublyperiodic import setup

vars().update(setup['NS2D'](**vars()))

hdf5file = HDF5Writer(FFT, float, {"U":U[0], "V":U[1], "P":P}, config.solver+".h5")
assert config.decomposition == 'line'

def add_pressure_diffusion(dU, P_hat, U_hat, K, K2, K_over_K2, nu):
    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat[:] = sum(dU*K_over_K2, 0, out=P_hat)

    # Add pressure gradient
    dU -= P_hat*K

    # Add contribution from diffusion
    dU -= nu*K2*U_hat
    
    return dU

def ComputeRHS(dU, rk):
    curl_hat = work[(FFT.complex_shape(), complex, 0)]    
    U_dealiased = work[((2,)+FFT.work_shape(config.dealias), float, 0)]
    curl_dealiased = work[(FFT.work_shape(config.dealias), float, 0)]
    
    curl_hat = cross2(curl_hat, K, U_hat)    
    curl_dealiased = FFT.ifft2(curl_hat, curl_dealiased, config.dealias)
    U_dealiased[0] = FFT.ifft2(U_hat[0], U_dealiased[0], config.dealias)
    U_dealiased[1] = FFT.ifft2(U_hat[1], U_dealiased[1], config.dealias)
    dU[0] = FFT.fft2(U_dealiased[1]*curl_dealiased, dU[0], config.dealias)
    dU[1] = FFT.fft2(-U_dealiased[0]*curl_dealiased, dU[1], config.dealias)
    dU = add_pressure_diffusion(dU, P_hat, U_hat, K, K2, K_over_K2, config.nu)    
    return dU

def regression_test(**kw):
    pass

def solve():
    timer = Timer()
    config.t = 0.0
    config.tstep = 0
    integrate = getintegrator(**globals())   

    while config.t < config.T:        
        config.t += config.dt
        config.tstep += 1

        U_hat[:] = integrate()

        for i in range(2): 
            U[i] = FFT.ifft2(U_hat[i], U[i])

        update(**globals())

        timer()
        
        if config.tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
                
    timer.final(MPI, rank)

    if config.make_profile:
        results = create_profile(**globals())
    
    regression_test(**globals())
    
    hdf5file.close()
