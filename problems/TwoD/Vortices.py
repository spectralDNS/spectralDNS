__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-03-07"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..TwoD import *

__all__ = ["initialize", "update", "regression_test", "parameters"]

# Set some (any) problem dependent parameters by overloading default parameters
parameters.update(
    {
      'nu': 0.001,
      'dt': 0.005,
      'T': 50,
      'debug': False}
    )

def initialize(U, X, U_hat, exp, pi, irfft2_mpi, rfft2_mpi, K_over_K2, **kwargs):
    w =     exp(-((X[0]-pi)**2+(X[1]-pi+pi/4)**2)/(0.2)) \
       +    exp(-((X[0]-pi)**2+(X[1]-pi-pi/4)**2)/(0.2)) \
       -0.5*exp(-((X[0]-pi-pi/4)**2+(X[1]-pi-pi/4)**2)/(0.4))
    w_hat = U_hat[0].copy()
    w_hat = rfft2_mpi(w, w_hat)
    U[0] = irfft2_mpi(1j*K_over_K2[1]*w_hat, U[0])
    U[1] = irfft2_mpi(-1j*K_over_K2[0]*w_hat, U[1])
    return U

def regression_test(U, num_processes, loadtxt, allclose, **kwargs):
    if num_processes > 1:
        return True
    U_ref = loadtxt('vortices.txt')
    assert allclose(U[0], U_ref)
