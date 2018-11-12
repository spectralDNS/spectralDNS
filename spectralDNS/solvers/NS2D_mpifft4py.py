__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unused-variable,unused-argument,function-redefined

from .NS_mpifft4py import *

# Reuses most of NS.py module

def get_context():
    """Set up context for NS2D solver"""

    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)

    # Mesh variables
    X = FFT.get_local_mesh()
    K = FFT.get_local_wavenumbermesh(scaled=True)
    K2 = K[0]*K[0] + K[1]*K[1]

    # Set Nyquist frequency to zero on K that is used for odd derivatives
    Kx = FFT.get_local_wavenumbermesh(scaled=True, eliminate_highest_freq=True)
    K_over_K2 = zeros((2,) + FFT.complex_shape())
    for i in range(2):
        K_over_K2[i] = K[i] / np.where(K2 == 0, 1, K2)

    # Solution variables
    U = empty((2,) + FFT.real_shape(), dtype=float)
    U_hat = empty((2,) + FFT.complex_shape(), dtype=complex)
    P = empty(FFT.real_shape(), dtype=float)
    P_hat = empty(FFT.complex_shape(), dtype=complex)
    curl = empty(FFT.real_shape(), dtype=float)

    # Primary variable
    u = U_hat

    Source = None

    # RHS and work arrays
    dU = empty((2,) + FFT.complex_shape(), dtype=complex)
    work = work_arrays()

    hdf5file = NS2DWriter({"U":U[0], "V":U[1], "P":P},
                          filename=params.h5filename+".h5",
                          chkpoint={'current':{'U':U, 'P':P}, 'previous':{}})

    return config.AttributeDict(locals())

class NS2DWriter(HDF5Writer):
    def update_components(self, **context):
        """Transform to real data before storing the solution"""
        get_velocity(**context)
        get_pressure(**context)

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

def set_velocity(U, U_hat, FFT, **context):
    """Compute transformed velocity from context"""
    for i in range(U.shape[0]):
        U_hat[i] = FFT.fft2(U[i], U_hat[i])
    return U_hat

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

    Conv.convection = convection
    return Conv
