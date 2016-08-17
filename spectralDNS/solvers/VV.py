__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-01-02"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
Velocity-vorticity formulation

Derived by taking the curl of the momentum equation and solving for the curl.
The velocity that is required in the convective term is computed from the curl
in the computeU method.

This solver inherits most features from the NS solver. 
Overloading just a few routines.
"""
from NS import *

def setup():
    """Set up context for solver"""
    
    # FFT class performs the 3D parallel transforms
    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)
    
    # Mesh variables
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()    
    K2 = np.sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)    
    
    # Velocity and pressure
    U     = empty((3,) + FFT.real_shape(), dtype=float)  
    curl  = empty((3,) + FFT.real_shape(), dtype=float)   
    W_hat = empty((3,) + FFT.complex_shape(), dtype=complex) # curl transformed

    # Primary variable
    u = W_hat

    # RHS array
    dU     = empty((3,) + FFT.complex_shape(), dtype=complex)
    Source = zeros((3,) + FFT.complex_shape(), dtype=complex) # Possible source term initialized to zero
    work = work_arrays()
        
    hdf5file = VVWriter({'U':U[0], 'V':U[1], 'W':U[2],
                         'curlx':curl[0], 'curly':curl[1], 'curlz':curl[2]},
                        chkpoint={'current':{'U':U, 'curl':curl}, 'previous':{}},
                        filename=params.solver+'.h5')

    return config.ParamsBase(locals())

class VVWriter(HDF5Writer):
    """Subclass HDF5Writer for appropriate updating of real components"""
    def update_components(self, **context):
        """Transform to real data when storing the solution"""
        U = get_velocity(**context)
        curl = get_curl(**context)

def computeU(c, a, work, FFT, K_over_K2, dealias=None):
    """Compute u from curl(u)
    
    Follows from
    w = [curl(u)=] \nabla \times u
    curl(w) = \nabla^2(u) (since div(u)=0)
    FFT(curl(w)) = FFT(\nabla^2(u))
    ik \times w_hat = k^2 u_hat
    u_hat = (ik \times w_hat) / k^2
    u = iFFT(u_hat)
    """
    v_hat = work[(a, 0)]
    v_hat = cross2(v_hat, K_over_K2, a)
    c[0] = FFT.ifftn(v_hat[0], c[0], dealias)
    c[1] = FFT.ifftn(v_hat[1], c[1], dealias)
    c[2] = FFT.ifftn(v_hat[2], c[2], dealias)    
    return c

def get_velocity(W_hat, U, work, FFT, K_over_K2, **context):
    """Compute velocity from context"""
    U = computeU(U, W_hat, work, FFT, K_over_K2)
    return U

def get_curl(curl, W_hat, FFT, **context):
    """Compute curl from context"""
    for i in range(3):
        curl[i] = FFT.ifftn(W_hat[i], curl[i])
    return curl

#@profile
def ComputeRHS(rhs, w_hat, work, FFT, K, K2, K_over_K2, Source, **context):
    """Compute rhs of spectral Navier Stokes in velocity-vorticity formulation
    
    args:
        rhs        The right hand side to be returned
        w_hat      The FFT of the curl at current time. May differ from
                   context.W_hat since it is set by the integrator

    Remaining args extracted from context:
        work       Work arrays
        FFT        Transform class from mpiFFT4py
        K          Scaled wavenumber mesh
        K2         K[0]*K[0] + K[1]*K[1] + K[2]*K[2]
        K_over_K2  K / K2
        Source     Source term to rhs
    """
    u_dealias = work[((3,)+FFT.work_shape(params.dealias), float, 0)]
    w_dealias = work[((3,)+FFT.work_shape(params.dealias), float, 1)]
    v_hat = work[(rhs, 0)]
    
    u_dealias = computeU(u_dealias, w_hat, work, FFT, K_over_K2, params.dealias)
    for i in range(3):
        w_dealias[i] = FFT.ifftn(w_hat[i], w_dealias[i], params.dealias)
    v_hat = Cross(v_hat, u_dealias, w_dealias, work, FFT, params.dealias) # v_hat = F_k(u_dealias x w_dealias)
    rhs = cross2(rhs, K, v_hat)  # rhs = 1j*(K x v_hat)
    rhs -= params.nu*K2*w_hat    
    rhs += Source    
    return rhs
