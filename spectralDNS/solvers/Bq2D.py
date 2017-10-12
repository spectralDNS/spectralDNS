__author__ = "Mikael Mortensen <mikaem@math.uio.no> and Diako Darian <diako.darian@mn.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from .spectralinit import *
from .NS2D import get_curl, get_velocity, get_pressure, end_of_tstep

def get_context():
    """Set up context for Bq2D solver"""

    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)

    # Mesh variables
    X = FFT.get_local_mesh()
    K = FFT.get_local_wavenumbermesh(scaled=True)
    K2 = K[0]*K[0] + K[1]*K[1]

    # Set Nyquist frequency to zero on K that is used for odd derivatives
    K = FFT.get_local_wavenumbermesh(scaled=True, eliminate_highest_freq=True)
    K_over_K2 = zeros((2,) + FFT.complex_shape())
    for i in range(2):
        K_over_K2[i] = K[i] / np.where(K2==0, 1, K2)

    # Solution variables
    Ur     = empty((3,) + FFT.real_shape(), dtype=float)
    Ur_hat = empty((3,) + FFT.complex_shape(), dtype=complex)
    P      = empty(FFT.real_shape(), dtype=float)
    P_hat  = empty(FFT.complex_shape(), dtype=complex)
    curl   = empty(FFT.real_shape(), dtype=float)

    # Create views into large data structures
    rho     = Ur[2]
    rho_hat = Ur_hat[2]
    U       = Ur[:2]
    U_hat   = Ur_hat[:2]

    # Primary variable
    u = Ur_hat

    # RHS and work arrays
    dU = empty((3,) + FFT.complex_shape(), dtype=complex)
    work = work_arrays()

    hdf5file = Bq2DWriter({'U':U[0], 'V':U[1], 'rho':rho, 'P':P},
                          chkpoint={'current':{'U':Ur, 'P':P}, 'previous':{}},
                          filename=params.h5filename+'.h5')

    return config.AttributeDict(locals())

class Bq2DWriter(HDF5Writer):
    def update_components(self, Ur, Ur_hat, P, P_hat, FFT, **context):
        """Transform to real data before storing the solution"""
        for i in range(3):
            Ur[i] = FFT.ifft2(Ur_hat[i], Ur[i])
        P = FFT.ifft2(P_hat, P)

def get_Ur(Ur, Ur_hat, FFT, **context):
    """Compute and return Ur from context"""
    for i in range(3):
        Ur[i] = FFT.ifft2(Ur_hat[i], Ur[i])
    return Ur

def get_rho(Ur, Ur_hat, FFT, **context):
    """Compute and return rho from context"""
    Ur[2] = FFT.ifft2(Ur_hat[2], Ur[2])
    return Ur[2]

def get_velocity(Ur, Ur_hat, FFT, **context):
    """Compute and return velocity from context"""
    Ur[0] = FFT.ifft2(Ur_hat[0], Ur[0])
    Ur[1] = FFT.ifft2(Ur_hat[1], Ur[1])
    return Ur[:2]

def getConvection(convection):
    """Return function used to compute nonlinear term"""
    if convection in ("Standard", "Divergence", "Skewed"):
        raise NotImplementedError

    elif convection == "Vortex":

        def Conv(rhs, ur_hat, work, FFT, K):
            ur_dealias = work[((3,)+FFT.work_shape(params.dealias), float, 0)]
            curl_dealias = work[(FFT.work_shape(params.dealias), float, 0)]
            F_tmp = work[(rhs, 0)]

            for i in range(3):
                ur_dealias[i] = FFT.ifft2(ur_hat[i], ur_dealias[i], params.dealias)

            u_dealias = ur_dealias[:2]
            rho_dealiased = ur_dealias[2]

            F_tmp[0] = cross2(F_tmp[0], K, ur_hat[:2])
            curl_dealias = FFT.ifft2(F_tmp[0], curl_dealias, params.dealias)
            rhs[0] = FFT.fft2(u_dealias[1]*curl_dealias, rhs[0], params.dealias)
            rhs[1] = FFT.fft2(-u_dealias[0]*curl_dealias, rhs[1], params.dealias)

            F_tmp[0] = FFT.fft2(u_dealias[0]*rho_dealiased, F_tmp[0], params.dealias)
            F_tmp[1] = FFT.fft2(u_dealias[1]*rho_dealiased, F_tmp[1], params.dealias)
            rhs[2] = -1j*(K[0]*F_tmp[0]+K[1]*F_tmp[1])

            return rhs

    Conv.convection = convection
    return Conv

@optimizer
def add_pressure_diffusion(rhs, ur_hat, P_hat, K_over_K2, K, K2, nu, Ri, Pr):
    u_hat = ur_hat[:2]
    rho_hat = ur_hat[2]

    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat = np.sum(rhs[:2]*K_over_K2, 0, out=P_hat)
    P_hat -= Ri*rho_hat*K_over_K2[1]

    # Add pressure gradient
    for i in range(2):
        rhs[i] -= P_hat*K[i]

    # Add contribution from diffusion
    rhs[0] -= nu*K2*u_hat[0]
    rhs[1] -= (nu*K2*u_hat[1] + Ri*rho_hat)
    rhs[2] -= nu*K2*rho_hat/Pr
    return rhs

def ComputeRHS(rhs, ur_hat, solver, work, FFT, K, K2, K_over_K2, P_hat, **context):
    """Compute and return right hand side of 2D Navier Stokes equations
    on Boussinesq form

    args:
        rhs         The right hand side to be returned
        ur_hat      The FFT of the velocity and density at current time.
                    May differ from context.Ur_hat since it is set by the
                    integrator
        solver      The solver module. Included for possible inheritance
                    and flexibility of integrators.

    Remaining args may be extracted from context:
        work        Work arrays
        FFT         Transform class from mpiFFT4py
        K           Scaled wavenumber mesh
        K2          K[0]*K[0] + K[1]*K[1] + K[2]*K[2]
        K_over_K2   K / K2
        P_hat       Transformed pressure

    """
    rhs = solver.conv(rhs, ur_hat, work, FFT, K)
    rhs = solver.add_pressure_diffusion(rhs, ur_hat, P_hat, K_over_K2, K, K2,
                                        params.nu, params.Ri, params.Pr)
    return rhs
