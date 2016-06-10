__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-12-30"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpiFFT4py import work_arrays, datatypes, zeros, empty

__all__ = ['setup']

def setupDNS(sum, where, config, get_FFT, **kwargs):
    
    params = config.params
    L = params.L
    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)
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
    
    work = work_arrays()
    
    del kwargs
    return locals() # Lazy (need only return what is needed)

def setupMHD(sum, where, config, get_FFT, **kwargs):
    
    params = config.params
    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)
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

    work = work_arrays()
    
    del kwargs
    return locals() # Lazy (need only return what is needed)

def setupVV(sum, where, config, get_FFT, **kwargs):
    d = setupDNS(sum, where, config, get_FFT, **kwargs)
    # Rename variable since we are working with a vorticity formulation
    d['W'] = empty((3,) + d['FFT'].real_shape(), dtype=d['float'])  # W is vorticity
    d['W_hat'] = d['U_hat']                               # U is used in setupDNS, rename here for convenience
    d['Source'] = zeros((3,) + d['FFT'].complex_shape(), dtype=d['complex']) # Possible source term initialized to zero
    return d
        
setup = {"MHD": setupMHD,
         "NS":  setupDNS,
         "VV":  setupVV}
