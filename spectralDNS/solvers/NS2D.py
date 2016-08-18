__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from .spectralinit import *

def setup():
    """Set up context for NS2D solver"""

    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)
    
    # Mesh variables
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()
    K2 = np.sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)

    # Solution variables
    U     = empty((2,) + FFT.real_shape(), dtype=float)
    U_hat = empty((2,) + FFT.complex_shape(), dtype=complex)
    P     = empty(FFT.real_shape(), dtype=float)
    P_hat = empty(FFT.complex_shape(), dtype=complex)
    curl  = empty(FFT.real_shape(), dtype=float)
    
    # Primary variable
    u = U_hat

    # RHS and work arrays
    dU = empty((2,) + FFT.complex_shape(), dtype=complex)
    work = work_arrays()

    hdf5file = NS2DWriter({"U":U[0], "V":U[1], "P":P}, 
                          filename=params.solver+".h5",
                          chkpoint={'current':{'U':U, 'P':P}, 'previous':{}})

    return config.ParamsBase(locals())

class NS2DWriter(HDF5Writer):
    def update_components(self, **context):
        """Transform to real data before storing the solution"""
        U = get_velocity(**context)
        P = get_pressure(**context)

def get_curl(curl, U_hat, work, FFT, K, **context):
    curl_hat = work[(FFT.complex_shape(), complex, 0)]
    curl_hat = cross2(curl_hat, K, U_hat)
    curl = FFT.ifft2(curl_hat, curl)
    return curl
            
def get_velocity(U, U_hat, FFT, **context):
    """Compute velocity from context"""
    for i in range(2):
        U[i] = FFT.ifft2(U_hat[i], U[i])
    return U

def get_pressure(P, P_hat, FFT, **context):
    """Compute pressure from context"""
    P = FFT.ifft2(1j*P_hat, P)
    return P

def add_pressure_diffusion(rhs, u_hat, P_hat, K, K2, K_over_K2, nu):
    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat[:] = np.sum(rhs*K_over_K2, 0, out=P_hat)

    # Add pressure gradient
    rhs -= P_hat*K

    # Add contribution from diffusion
    rhs -= nu*K2*u_hat
    
    return rhs

def getConvection(convection):
    """Return function used to compute nonlinear term"""
    if convection in ("Standard", "Divergence", "Skewed"):

        raise NotImplementedError

    elif convection == "Vortex":

        def Conv(rhs, u_hat, work, FFT, K):
            curl_hat = work[(FFT.complex_shape(), complex, 0)]
            u_dealias = work[((2,)+FFT.work_shape(params.dealias), float, 0)]
            curl_dealias = work[(FFT.work_shape(params.dealias), float, 0)]
            
            curl_hat = cross2(curl_hat, K, u_hat)
            curl_dealias = FFT.ifft2(curl_hat, curl_dealias, params.dealias)
            u_dealias[0] = FFT.ifft2(u_hat[0], u_dealias[0], params.dealias)
            u_dealias[1] = FFT.ifft2(u_hat[1], u_dealias[1], params.dealias)
            rhs[0] = FFT.fft2(u_dealias[1]*curl_dealias, rhs[0], params.dealias)
            rhs[1] = FFT.fft2(-u_dealias[0]*curl_dealias, rhs[1], params.dealias)
            return rhs

    return Conv

def ComputeRHS(rhs, u_hat, work, FFT, K, K2, K_over_K2, P_hat, **context):
    """Compute rhs of spectral Navier Stokes in 2D
    
    args:
        rhs        The right hand side to be returned
        u_hat      The FFT of the velocity at current time. May differ from
                   context.U_hat since it is set by the integrator

    Remaining args extracted from context:
        work       Work arrays
        FFT        Transform class from mpiFFT4py
        K          Scaled wavenumber mesh
        K2         K[0]*K[0] + K[1]*K[1] + K[2]*K[2]
        K_over_K2  K / K2
        P_hat      Transfomred pressure
    
    global functions:
        add_pressure_diffusion
                   Adds pressure and diffusion terms
        conv       Method returned from getConvection. Must be present in
                   global namespace prior to calling ComputeRHS
    """

    rhs = conv(rhs, u_hat, work, FFT, K)
    
    rhs = add_pressure_diffusion(rhs, u_hat, P_hat, K, K2, K_over_K2, params.nu)    
    
    return rhs
