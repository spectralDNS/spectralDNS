__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *

hdf5file = HDF5Writer(comm, float, {"U":U[0], "V":U[1], "P":P}, config.solver+".h5")
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
    F_tmp[0] = cross2(F_tmp[0], K, U_hat)
    curl[:] = FFT.ifft2(F_tmp[0], curl, config.dealias)
    U_dealiased = FFT.get_workarray(((2,)+FFT.real_shape(), float), 0)
    U_dealiased[0] = FFT.ifft2(U_hat[0], U_dealiased[0], config.dealias)
    U_dealiased[1] = FFT.ifft2(U_hat[1], U_dealiased[1], config.dealias)
    dU[0] = FFT.fft2(U_dealiased[1]*curl, dU[0], config.dealias)
    dU[1] = FFT.fft2(-U_dealiased[0]*curl, dU[1], config.dealias)
    dU = add_pressure_diffusion(dU, P_hat, U_hat, K, K2, K_over_K2, nu)    
    return dU

integrate = getintegrator(**vars())   

def regression_test(t, tstep, **kw):
    pass

def solve():
    timer = Timer()
    t = 0.0
    tstep = 0
    while t < config.T:        
        t += dt
        tstep += 1

        U_hat[:] = integrate(t, tstep, dt)

        for i in range(2): 
            U[i] = FFT.ifft2(U_hat[i], U[i])

        update(t, tstep, **globals())

        timer()
        
        if tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
                
    timer.final(MPI, rank)

    if config.make_profile:
        results = create_profile(**globals())
    
    regression_test(t, tstep, **globals())
    
    hdf5file.close()
