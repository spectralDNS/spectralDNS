__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-12-30"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from numpy import sum,where
from mpiFFT4py import *

_all__ = ['setup']

def setupDNS(context):
   
    float = context.types["float"]
    complex = context.types["complex"]
    FFT = context.FFT

    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()
    K2 = sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)    
    
    U     = empty((3,) + FFT.real_shape(), dtype=float)  
    U_hat = empty((3,) + FFT.complex_shape(), dtype=complex)
    P     = empty(FFT.real_shape(), dtype=float)
    P_hat = empty(FFT.complex_shape(), dtype=complex)

    # RHS array
    dU     = empty((3,) + FFT.complex_shape(), dtype=complex)

    # 
    curl   = empty((3,) + FFT.real_shape(), dtype=float)   
    Source = None
    
    to_return = locals()
    del to_return["context"]
    del to_return["float"]
    del to_return["complex"]
    del to_return["FFT"]
    return locals() # Lazy (need only return what is needed)

def setupDNS_Boussinesq(context):
   
    float = context.types["float"]
    complex = context.types["complex"]
    FFT = context.FFT

    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()
    K2 = sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)    
    
    Ur     = empty((4,) + FFT.real_shape(), dtype=float)  
    Ur_hat = empty((4,) + FFT.complex_shape(), dtype=complex)
    P     = empty(FFT.real_shape(), dtype=float)
    P_hat = empty(FFT.complex_shape(), dtype=complex)
    dUr     = empty((4,) + FFT.complex_shape(), dtype=complex)

    # Create views into large data structures
    rho     = Ur[3]
    rho_hat = Ur_hat[3]
    U       = Ur[:3] 
    U_hat   = Ur_hat[:3]

    # 
    curl   = empty((3,) + FFT.real_shape(), dtype=float)   
    Source = None
    
    to_return = locals()
    del to_return["context"]
    del to_return["float"]
    del to_return["complex"]
    del to_return["FFT"]
    return locals() # Lazy (need only return what is needed)

def setupMHD(context):

    float = context.types["float"]
    complex = context.types["complex"]
    FFT = context.FFT
    
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()

    K2 = sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)

    UB     = empty((6,) + FFT.real_shape(), dtype=float)  
    UB_hat = empty((6,) + FFT.complex_shape(), dtype=complex)
    P      = empty(FFT.real_shape(), dtype=float)
    P_hat  = empty(FFT.complex_shape(), dtype=complex)
    
    # Create views into large data structures
    U     = UB[:3] 
    U_hat = UB_hat[:3]
    B     = UB[3:]
    B_hat = UB_hat[3:]

    # RHS array
    dU = empty((6,) + FFT.complex_shape(), dtype=complex)

    # 
    curl   = empty((3,) + FFT.real_shape(), dtype=float)   
    Source = None
    
    to_return = locals()
    del to_return["context"]
    del to_return["float"]
    del to_return["complex"]
    del to_return["FFT"]
 
    return locals() # Lazy (need only return what is needed)

def setup(solver,**kwargs):
        return {"MHD": setupMHD,
         "NS":  setupDNS,
         "VV":  setupDNS,
         "Bq3D":setupDNS_Boussinesq}[solver](**kwargs)
