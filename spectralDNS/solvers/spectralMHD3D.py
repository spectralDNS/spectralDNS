__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *
            
hdf5file = HDF5Writer(comm, float, {"U":U[0], "V":U[1], "W":U[2], "P":P, 
                                    "Bx": B[0], "By": B[1], "Bz":B[2]}, config.solver+".h5")

eta = float(config.eta)

def set_Elsasser(c, F_tmp, K):
    c[:3] = -1j*(K[0]*(F_tmp[:, 0] + F_tmp[0, :])
                +K[1]*(F_tmp[:, 1] + F_tmp[1, :])
                +K[2]*(F_tmp[:, 2] + F_tmp[2, :]))/2.0

    c[3:] =  1j*(K[0]*(F_tmp[0, :] - F_tmp[:, 0])
                +K[1]*(F_tmp[1, :] - F_tmp[:, 1])
                +K[2]*(F_tmp[2, :] - F_tmp[:, 2]))/2.0
    return c

def divergenceConvection(z0, z1, c):
    """Divergence convection using Elsasser variables
    z0=U+B
    z1=U-B
    """
    for i in range(3):
        for j in range(3):
            F_tmp[i, j] = FFT.fftn(z0[i]*z1[j], F_tmp[i, j])
            
    c = set_Elsasser(c, F_tmp, K)
    #c[:3] = -1j*(K[0]*(F_tmp[:, 0] + F_tmp[0, :])
                #+K[1]*(F_tmp[:, 1] + F_tmp[1, :])
                #+K[2]*(F_tmp[:, 2] + F_tmp[2, :]))/2.0

    #c[3:] =  1j*(K[0]*(F_tmp[0, :] - F_tmp[:, 0])
                #+K[1]*(F_tmp[1, :] - F_tmp[:, 1])
                #+K[2]*(F_tmp[2, :] - F_tmp[:, 2]))/2.0

    return c    
    
def ComputeRHS(dU, rk):
    if rk > 0: # For rk=0 the correct values are already in U, B
        for i in range(6):
            UB[i] = FFT.ifftn(UB_hat[i], UB[i])
    
    # Compute convective term and place in dU
    dU = divergenceConvection(U+B, U-B, dU)
    
    # Dealias the nonlinear convection
    dU[:] *= dealias
    
    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat[:] = sum(dU[:3]*K_over_K2, 0)
        
    # Add pressure gradient
    dU[:3] -= P_hat*K

    # Add contribution from diffusion
    dU[:3] -= nu*K2*U_hat
    dU[3:] -= eta*K2*B_hat
    
    return dU

def regression_test(t, tstep, **kw):
    pass

# Set up function to perform temporal integration (using config.integrator parameter)
integrate = getintegrator(**vars())

def solve():
    timer = Timer()
    t = 0.0
    tstep = 0
    while t < config.T-1e-8:
        t += dt 
        tstep += 1
        
        UB_hat[:] = integrate(t, tstep, dt)

        for i in range(6):
            UB[i] = FFT.ifftn(UB_hat[i], UB[i])
                 
        update(t, tstep, **globals())
        
        timer()
        
        if tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()

    timer.final(MPI, rank)
    
    if config.make_profile:
        results = create_profile(**vars())
        
    regression_test(t, tstep, **globals())
    
    hdf5file.close()
