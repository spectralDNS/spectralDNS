__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-03-07"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..TwoD import *

__all__ = ["initialize", "update", "regression_test", "parameters"]

# Set some (any) problem dependent parameters by overloading default parameters
parameters.update(
    {
    'nu': 0.01,
    'dt': 0.01}
    )

def initialize(U, X, sin, cos, **kwargs):
    U[0] = sin(X[0])*cos(X[1])
    U[1] =-cos(X[0])*sin(X[1])
    return U

def update(comm, rank, tstep, time, debug, U, dx, L, t0, float64, sum, **kwargs):
    # Compute energy with double precision
    kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx*dx/L**2/2) 
    if rank == 0 and debug == True:
        print "%d %2.8f %2.8f"%(tstep, time.time()-t0[0], kk)
        t0[0] = time.time()

def regression_test(comm, U, X, dx, L, nu, t, sin, cos, sum, float64, exp, 
                    rank, **kwargs):
    # Check accuracy
    kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx*dx/L**2/2)
    u0 = sin(X[0])*cos(X[1])*exp(-2.*nu*t)
    u1 =-sin(X[1])*cos(X[0])*exp(-2.*nu*t)
    k1 = comm.reduce(sum(u0*u0+u1*u1)*dx*dx/L**2/2) # Compute energy with double precision)
    if rank == 0:
        print "Energy (exact, numeric, error)  = (%2.6f, %2.6f, %2.4e) " %(k1, kk, k1-kk)
        assert abs(k1-kk)<1.e-10

