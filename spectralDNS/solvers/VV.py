__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-01-02"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
Velocity-vorticity formulation
"""
from spectralinit import *
from NS import Cross, hdf5file, regression_test

# Rename variable since we are working with a vorticity formulation
W = U.copy()               # W is vorticity
W_hat = U_hat              # U is used in setup, rename here for convenience
W_dealiased = U_dealiased.copy()
Source = U_hat.copy()*0    # Possible source term initialized to zero

def Curl(a, c, dealiasing=True):
    """c = curl(a) = F_inv(F(curl(a))/K2) = F_inv(1j*(K x a)/K2)"""
    F_tmp[:] = cross2(F_tmp, K_over_K2, a)
    if dealiasing:
        F_tmp[:] = dealias_rhs(F_tmp, dealias)
    c[0] = FFT.ifftn(F_tmp[0], c[0])
    c[1] = FFT.ifftn(F_tmp[1], c[1])
    c[2] = FFT.ifftn(F_tmp[2], c[2])    
    return c

#@profile
def ComputeRHS(dU, rk):
    if rk > 0:
        for i in range(3):
            W[i] = FFT.ifftn(W_hat[i], W[i])
            
    U_dealiased[:] = Curl(W_hat, U_dealiased)
    for i in range(3):
        W_dealiased[i] = FFT.ifftn(W_hat[i]*dealias, W_dealiased[i])
    F_tmp[:] = Cross(U_dealiased, W_dealiased, F_tmp)
    dU = cross2(dU, K, F_tmp)    
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
            W[i] = FFT.ifftn(W_hat[i], W[i])
                        
        update(t, tstep, **globals())
        
        timer()
            
        if tstep == 1 and config.make_profile:
            profiler.enable()  #Enable profiling after first step is finished

    timer.final(MPI, rank)
    
    if config.make_profile:
        results = create_profile(**vars())

    regression_test(t, tstep, **globals())
        
    hdf5file.close()
