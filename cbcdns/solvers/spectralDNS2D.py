__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *

hdf5file = HDF5Writer(comm, N, float, {"U":U[0], "V":U[1], "P":P}, config.solver+".h5")
assert config.decomposition == 'line'

def ComputeRHS(dU, rk):
    if rk > 0: # For rk=0 the correct values are already in U, V, W
        U[0] = ifft2_mpi(U_hat[0], U[0])
        U[1] = ifft2_mpi(U_hat[1], U[1])

    curl[:] = ifft2_mpi(1j*(K[0]*U_hat[1] - K[1]*U_hat[0]), curl)
    dU[0] = fft2_mpi(U[1]*curl, dU[0])
    dU[1] = fft2_mpi(-U[0]*curl, dU[1])

    # Dealias the nonlinear convection
    dU *= dealias

    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat[:] = sum(dU*K_over_K2, 0, out=P_hat)

    # Add pressure gradient
    dU -= P_hat*K

    # Add contribution from diffusion
    dU -= nu*K2*U_hat
    
    return dU

integrate = getintegrator(**vars())   

def regression_test(**kw):
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
            U[i] = ifft2_mpi(U_hat[i], U[i])

        update(t, tstep, **globals())

        timer()
        
        if tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
                
    timer.final(MPI, rank)

    if config.make_profile:
        results = create_profile(**globals())
    
    regression_test(**globals())
    
    hdf5file.close()

