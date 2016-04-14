__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-12-30"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpiFFT4py import *
from spectralDNS import config
from ..optimization import optimizer
from numpy import array, sum, meshgrid, mgrid, where, abs, pi, uint8, conj

__all__ = ['setup']

def setupNS(float, complex, FFT, **kwargs):
        
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()

    # Solution array and Fourier coefficients
    U     = empty((2,) + FFT.real_shape(), dtype=float)
    U_hat = empty((2,) + FFT.complex_shape(), dtype=complex)
    P     = empty(FFT.real_shape(), dtype=float)
    P_hat = empty(FFT.complex_shape(), dtype=complex)
    curl  = empty(FFT.real_shape(), dtype=float)
    F_tmp = empty((2,) + FFT.complex_shape(), dtype=complex)
    dU     = empty((2,) + FFT.complex_shape(), dtype=complex)
    
    K2 = sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)    

    del kwargs
    return locals()

def setupBoussinesq(float, complex, FFT, **kwargs):
    
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()

    Ur     = empty((3,) + FFT.real_shape(), dtype=float)
    Ur_hat = empty((3,) + FFT.complex_shape(), dtype=complex)
    P      = empty(FFT.real_shape(), dtype=float)
    P_hat  = empty(FFT.complex_shape(), dtype=complex)
    curl   = empty(FFT.real_shape(), dtype=float)
    dU     = empty((3,) + FFT.complex_shape(), dtype=complex)
     
    # Create views into large data structures
    rho     = Ur[2]
    rho_hat = Ur_hat[2]
    U       = Ur[:2] 
    U_hat   = Ur_hat[:2]

    F_tmp = empty((2,) + FFT.complex_shape(), dtype=complex)
    
    K2 = sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)    

    del kwargs
    return locals()

setup = {"NS2D": setupNS,
         "Bq2D": setupBoussinesq}[config.solver]
