__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-01-02"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
Velocity-vorticity formulation
"""
from spectralinit import *
from spectralDNS import Cross, hdf5file

# Rename variable since we are working with a vorticity formulation
W = U.copy()               # W is vorticity
W_hat = U_hat              # U_hat is used in subroutines, rename here for convenience
Source = U_hat.copy()*0    # Possible source term initialized to zero

def Curl(a, c):
    """c = curl(a) = F_inv(F(curl(a))/K2) = F_inv(1j*(K x a)/K2)"""
    F_tmp[:] = cross2(F_tmp, K_over_K2, a)
    c[0] = ifftn_mpi(F_tmp[0], c[0])
    c[1] = ifftn_mpi(F_tmp[1], c[1])
    c[2] = ifftn_mpi(F_tmp[2], c[2])    
    return c

#@profile
def ComputeRHS(dU, rk):
    if rk > 0:
        for i in range(3):
            W[i] = ifftn_mpi(W_hat[i], W[i])
            
    U[:] = Curl(W_hat, U)
    F_tmp[:] = Cross(U, W, F_tmp)
    dU = cross2(dU, K, F_tmp)    
    dU = dealias_rhs(dU, dealias)
    dU -= nu*K2*W_hat    
    dU += Source    
    return dU

# Set up function to perform temporal integration (using config.integrator)
integrate = getintegrator(**vars())

def solve():
    timer = Timer()
    t = 0.0
    tstep = 0
    while t < config.T-1e-8:
        t += dt
        tstep += 1
        
        W_hat = integrate(t, tstep, dt)

        for i in range(3):
            W[i] = ifftn_mpi(W_hat[i], W[i])
                        
        update(t, tstep, **globals())
        
        timer()
            
        if tstep == 1 and config.make_profile:
            profiler.enable()  #Enable profiling after first step is finished

    timer.final(MPI, rank)
    
    if config.make_profile:
        results = create_profile(**vars())
        
    hdf5file.close()
