__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *

def setup():
    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()
    K2 = np.sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)

    UB = empty((6,) + FFT.real_shape(), dtype=float)
    P  = empty(FFT.real_shape(), dtype=float)
    curl = empty((3,) + FFT.real_shape(), dtype=float)
    UB_hat = empty((6,) + FFT.complex_shape(), dtype=complex)
    P_hat = empty(FFT.complex_shape(), dtype=complex)
    dU = empty((6,) + FFT.complex_shape(), dtype=complex)
    Source = None

    # Create views into large data structures
    U     = UB[:3]
    U_hat = UB_hat[:3]
    B     = UB[3:]
    B_hat = UB_hat[3:]
    
    # Primary variable
    u = UB_hat

    work = work_arrays()
    
    hdf5file = MHDWriter({'U':U[0], 'V':U[1], 'W':U[2], 'P':P,
                         'Bx':B[0], 'By':B[1], 'Bz':B[2]},
                         chkpoint={'current':{'UB':UB, 'P':P}, 'previous':{}},
                         filename=params.h5filename+'.h5')

    return config.ParamsBase(locals())

class MHDWriter(HDF5Writer):
    def update_components(self, UB, UB_hat, P, P_hat, FFT, **kw):
        """Transform to real data when storing the solution"""
        for i in range(6):
            UB[i] = FFT.ifftn(UB_hat[i], UB[i])
        P = FFT.ifftn(P_hat, P)

def backward_transform(u, u_hat, FFT):
    for i in range(u.shape[0]):
        u[i] = FFT.ifftn(u_hat[i], u[i])
    return u

def forward_transform(u_hat, u, FFT):
    for i in range(u.shape[0]):
        u_hat[i] = FFT.fftn(u[i], u_hat[i])
    return u_hat

def get_UB(UB, UB_hat, FFT, **context):
    """Compute U and B from context"""
    UB = backward_transform(UB, UB_hat, FFT)
    return UB

def set_Elsasser(c, F_tmp, K):
    c[:3] = -1j*(K[0]*(F_tmp[:, 0] + F_tmp[0, :])
                +K[1]*(F_tmp[:, 1] + F_tmp[1, :])
                +K[2]*(F_tmp[:, 2] + F_tmp[2, :]))/2.0

    c[3:] =  1j*(K[0]*(F_tmp[0, :] - F_tmp[:, 0])
                +K[1]*(F_tmp[1, :] - F_tmp[:, 1])
                +K[2]*(F_tmp[2, :] - F_tmp[:, 2]))/2.0
    return c

def divergenceConvection(c, z0, z1, work, FFT, K, dealias=None):
    """Divergence convection using Elsasser variables
    z0=U+B
    z1=U-B
    """
    F_tmp = work[((3, 3) + FFT.complex_shape(), complex, 0)]
    for i in range(3):
        for j in range(3):
            F_tmp[i, j] = FFT.fftn(z0[i]*z1[j], F_tmp[i, j], dealias)

    c = _set_Elsasser(c, F_tmp, K)
    return c

def getConvection(convection):

    if convection in ("Standard", "Vortex", "Skewed"):
        raise NotImplementedError

    elif convection == "Divergence":

        def Conv(rhs, ub_hat, work, FFT, K):
            ub_dealias = work[((6,)+FFT.work_shape(params.dealias), float, 0)]
            for i in range(6):
                ub_dealias[i] = FFT.ifftn(ub_hat[i], ub_dealias[i], params.dealias)

            u_dealias = ub_dealias[:3]
            b_dealias = ub_dealias[3:]
            # Compute convective term and place in dU
            rhs = _divergenceConvection(rhs, u_dealias+b_dealias, u_dealias-b_dealias,
                                        work, FFT, K, params.dealias)
            return rhs
    
    Conv.convection = convection
    return Conv

@optimizer
def add_pressure_diffusion(rhs, ub_hat, nu, eta, K2, K, P_hat, K_over_K2):
    """Add contributions from pressure and diffusion to the rhs"""

    u_hat = ub_hat[:3]
    b_hat = ub_hat[3:]
    
    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat = np.sum(rhs[:3]*K_over_K2, 0, out=P_hat)

    # Add pressure gradient
    rhs[:3] -= P_hat*K

    # Add contribution from diffusion
    rhs[:3] -= nu*K2*u_hat
    rhs[3:] -= eta*K2*b_hat
    return rhs

def ComputeRHS(rhs, ub_hat, solver, work, FFT, K, K2, K_over_K2, P_hat, **context):
    """Return right hand side of Navier Stokes
    
    args:
        rhs         The right hand side to be returned
        ub_hat      The FFT of the velocity and magnetic fields at current
                    time. May differ from context.UB_hat since it is set by
                    the integrator
        solver      The solver module. Included for possible inheritance.

    Remaining args may be extracted from context:
        work        Work arrays
        FFT         Transform class from mpiFFT4py
        K           Scaled wavenumber mesh
        K2          sum_i K[i]*K[i]
        K_over_K2   K / K2
        P_hat       Transfomred pressure
    
    """
    # Get and evaluate the convection method
    try:
        rhs = ComputeRHS._conv(rhs, ub_hat, work, FFT, K)
        assert ComputeRHS._conv.convection == params.convection

    except (AttributeError, AssertionError):
        ComputeRHS._conv = solver.getConvection(params.convection)
        rhs = ComputeRHS._conv(rhs, ub_hat, work, FFT, K)

    rhs = solver.add_pressure_diffusion(rhs, ub_hat, params.nu, params.eta, K2,
                                        K, P_hat, K_over_K2)
    return rhs
