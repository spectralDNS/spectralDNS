__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-12-30"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"


from mpiFFT4py import *

from numpy import array, sum, meshgrid, mgrid, where, abs, pi, uint8, conj

__all__ = ['setup']

def setupNS(context): 
    float = context.types["float"]
    complex = context.types["complex"]
    FFT = context.FFT
        
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()

    # Solution array and Fourier coefficients
    U     = empty((2,) + FFT.real_shape(), dtype=float)
    U_hat = empty((2,) + FFT.complex_shape(), dtype=complex)
    P     = empty(FFT.real_shape(), dtype=float)
    P_hat = empty(FFT.complex_shape(), dtype=complex)
    curl  = empty(FFT.real_shape(), dtype=float)
    dU     = empty((2,) + FFT.complex_shape(), dtype=complex)
    
    K2 = sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)    

    to_return = locals()
    del to_return["context"]
    del to_return["float"]
    del to_return["complex"]
    del to_return["FFT"]
    return locals() # Lazy (need only return what is needed)

def setupBoussinesq(context):

    float = context.types["float"]
    complex = context.types["complex"]
    FFT = context.FFT
    
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

    K2 = sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)    

    to_return = locals()
    del to_return["context"]
    del to_return["float"]
    del to_return["complex"]
    del to_return["FFT"]
    return locals() # Lazy (need only return what is needed)

def setup(solver,**kwargs):
        return { "NS2D": setupNS, "Bq2D":  setupBoussinesq}[solver](**kwargs)
