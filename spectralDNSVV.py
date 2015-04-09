__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-01-02"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
Velocity-vorticity formulation
"""
from MPI_knee import mpi_import
with mpi_import():
    from init import *

# Rename variable since we are working with a vorticity formulation
W = U.copy()               # W is vorticity
W_hat = U_hat              # U_hat is used in subroutines, rename here for convenience
Source = U_hat.copy()

def Cross(a, b, c):
    """c_k = F_k(a x b)"""
    U_tmp[:] = cross1(U_tmp, a, b)
    c[0] = fftn_mpi(U_tmp[0], c[0])
    c[1] = fftn_mpi(U_tmp[1], c[1])
    c[2] = fftn_mpi(U_tmp[2], c[2])
    return c

def Curl(a, c):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    F_tmp[:] = cross3(F_tmp, K_over_K2, a)
    c[0] = ifftn_mpi(F_tmp[0], c[0])
    c[1] = ifftn_mpi(F_tmp[1], c[1])
    c[2] = ifftn_mpi(F_tmp[2], c[2])    
    return c

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

Source = set_source(**vars())

# Set up function to perform temporal integration (using config.integrator parameter)
integrate = getintegrator(**vars())

def solve():
    t = 0.0
    tstep = 0
    fastest_time = 1e8
    slowest_time = 0.0
    tic = t0 = time.time()
    while t < config.T-1e-8:
        t += dt; tstep += 1
        
        W_hat = integrate(t, tstep, dt)

        for i in range(3):
            W[i] = ifftn_mpi(W_hat[i], W[i])
                        
        globals().update(locals())                
        update(**globals())
        
        tt = time.time()-t0
        t0 = time.time()
        if tstep > 1:
            fastest_time = min(tt, fastest_time)
            slowest_time = max(tt, slowest_time)
            
        if tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()

    toc = time.time()-tic

    # Get min/max of fastest and slowest process
    fast = (comm.reduce(fastest_time, op=MPI.MIN, root=0),
            comm.reduce(slowest_time, op=MPI.MIN, root=0))
    slow = (comm.reduce(fastest_time, op=MPI.MAX, root=0),
            comm.reduce(slowest_time, op=MPI.MAX, root=0))

    if rank == 0:
        print "Time = ", toc
        print "Fastest = ", fast
        print "Slowest = ", slow
        
    if config.make_profile:
        results = create_profile(**vars())
        
    hdf5file.close()
