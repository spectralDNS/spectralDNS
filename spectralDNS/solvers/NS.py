__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from .spectralinit import *

def get_context():
    """Set up context for classical (NS) solver"""

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
    U_hat = empty((3,) + FFT.complex_shape(), dtype=complex)
    P     = empty(FFT.real_shape(), dtype=float)
    P_hat = empty(FFT.complex_shape(), dtype=complex)

    # Primary variable
    u = U_hat

    # RHS array
    dU     = empty((3,) + FFT.complex_shape(), dtype=complex)
    curl   = empty((3,) + FFT.real_shape(), dtype=float)   
    Source = None
    work = work_arrays()
        
    hdf5file = NSWriter({'U':U[0], 'V':U[1], 'W':U[2], 'P':P},
                        chkpoint={'current':{'U':U, 'P':P}, 'previous':{}},
                        filename=params.h5filename+'.h5')

    return config.AttributeDict(locals())

class NSWriter(HDF5Writer):
    """Subclass HDF5Writer for appropriate updating of real components

    The method 'update_components' is used to transform all variables
    that are to be stored. If more variables than U and P are
    wanted, then subclass HDF5Writer in the application.
    """
    def update_components(self, **context):
        """Transform to real data before storing the solution"""
        U = get_velocity(**context)
        P = get_pressure(**context)

def get_curl(curl, U_hat, work, FFT, K, **context):
    """Compute curl from context"""
    curl = compute_curl(curl, U_hat, work, FFT, K)
    return curl

def get_velocity(U, U_hat, FFT, **context):
    """Compute velocity from context"""
    for i in range(3):
        U[i] = FFT.ifftn(U_hat[i], U[i])
    return U

def get_pressure(P, P_hat, FFT, **context):
    """Compute pressure from context"""
    P = FFT.ifftn(1j*P_hat, P)

def forward_transform(a, a_hat, FFT):
    """A common method for transforming forward """
    for i in range(3):
        a_hat[i] = FFT.fftn(a[i], a_hat[i])
    return a_hat

def backward_transform(a_hat, a, FFT):
    """A common method for transforming backward"""
    for i in range(3):
        a[i] = FFT.ifftn(a_hat[i], a[i])
    return a

def compute_curl(c, a, work, FFT, K, dealias=None):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    curl_hat = work[(a, 0, False)]
    curl_hat = cross2(curl_hat, K, a)
    c[0] = FFT.ifftn(curl_hat[0], c[0], dealias)
    c[1] = FFT.ifftn(curl_hat[1], c[1], dealias)
    c[2] = FFT.ifftn(curl_hat[2], c[2], dealias)
    return c

def Cross(c, a, b, work, FFT, dealias=None):
    """c_k = F_k(a x b)"""
    Uc = work[(a, 2, False)]
    Uc = cross1(Uc, a, b)
    c[0] = FFT.fftn(Uc[0], c[0], dealias)
    c[1] = FFT.fftn(Uc[1], c[1], dealias)
    c[2] = FFT.fftn(Uc[2], c[2], dealias)
    return c

def standard_convection(rhs, u_dealias, U_hat, work, FFT, K, dealias=None):
    """rhs_i = u_j du_i/dx_j"""
    gradUi = work[(u_dealias, 2, False)]
    for i in range(3):
        for j in range(3):
            gradUi[j] = FFT.ifftn(1j*K[j]*U_hat[i], gradUi[j], dealias)
        rhs[i] = FFT.fftn(np.sum(u_dealias*gradUi, 0), rhs[i], dealias)
    return rhs

def divergence_convection(rhs, u_dealias, work, FFT, K, dealias=None, add=False):
    """rhs_i = div(u_i u_j)"""
    if not add: rhs.fill(0)
    UUi_hat = work[(rhs, 0, False)]
    for i in range(3):
        UUi_hat[i] = FFT.fftn(u_dealias[0]*u_dealias[i], UUi_hat[i], dealias)
    rhs[0] += 1j*np.sum(K*UUi_hat, 0)
    rhs[1] += 1j*K[0]*UUi_hat[1]
    rhs[2] += 1j*K[0]*UUi_hat[2]
    UUi_hat[0] = FFT.fftn(u_dealias[1]*u_dealias[1], UUi_hat[0], dealias)
    UUi_hat[1] = FFT.fftn(u_dealias[1]*u_dealias[2], UUi_hat[1], dealias)
    UUi_hat[2] = FFT.fftn(u_dealias[2]*u_dealias[2], UUi_hat[2], dealias)
    rhs[1] += (1j*K[1]*UUi_hat[0] + 1j*K[2]*UUi_hat[1])
    rhs[2] += (1j*K[1]*UUi_hat[1] + 1j*K[2]*UUi_hat[2])
    return rhs

def getConvection(convection):
    
    if convection == "Standard":

        def Conv(rhs, u_hat, work, FFT, K):
            u_dealias = work[((3,)+FFT.work_shape(params.dealias),
                            float, 0, False)]
            for i in range(3):
                u_dealias[i] = FFT.ifftn(u_hat[i], u_dealias[i], params.dealias)
            rhs = standard_convection(rhs, u_dealias, u_hat, work, FFT, K, params.dealias)
            rhs[:] *= -1
            return rhs

    elif convection == "Divergence":

        def Conv(rhs, u_hat, work, FFT, K):
            u_dealias = work[((3,)+FFT.work_shape(params.dealias),
                            float, 0, False)]
            for i in range(3):
                u_dealias[i] = FFT.ifftn(u_hat[i], u_dealias[i], params.dealias)
            rhs = divergence_convection(rhs, u_dealias, work, FFT, K, params.dealias, False)
            rhs[:] *= -1
            return rhs

    elif convection == "Skewed":

        def Conv(rhs, u_hat, work, FFT, K):
            u_dealias = work[((3,)+FFT.work_shape(params.dealias),
                            float, 0, False)]
            for i in range(3):
                u_dealias[i] = FFT.ifftn(u_hat[i], u_dealias[i], params.dealias)
            rhs = standard_convection(rhs, u_dealias, u_hat, work, FFT, K, params.dealias)
            rhs = divergence_convection(rhs, u_dealias, work, FFT, K, params.dealias, True)
            rhs *= -0.5
            return rhs

    elif convection == "Vortex":

        #@profile
        def Conv(rhs, u_hat, work, FFT, K):
            u_dealias = work[((3,)+FFT.work_shape(params.dealias),
                            float, 0, False)]
            curl_dealias = work[((3,)+FFT.work_shape(params.dealias),
                                float, 1, False)]
            for i in range(3):
                u_dealias[i] = FFT.ifftn(u_hat[i], u_dealias[i], params.dealias)

            curl_dealias = compute_curl(curl_dealias, u_hat, work, FFT, K, params.dealias)
            rhs = Cross(rhs, u_dealias, curl_dealias, work, FFT, params.dealias)
            return rhs

    Conv.convection = convection
    return Conv

@optimizer
def add_pressure_diffusion(rhs, u_hat, nu, K2, K, P_hat, K_over_K2):
    """Add contributions from pressure and diffusion to the rhs"""

    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat = np.sum(rhs*K_over_K2, 0, out=P_hat)

    # Subtract pressure gradient
    rhs -= P_hat*K

    # Subtract contribution from diffusion
    rhs -= nu*K2*u_hat

    return rhs

def ComputeRHS(rhs, u_hat, solver, work, FFT, P_hat, K, K2, K_over_K2, **context):
    """Compute right hand side of Navier Stokes
    
    args:
        rhs         The right hand side to be returned
        u_hat       The FFT of the velocity at current time. May differ from
                    context.U_hat since it is set by the integrator
        solver      The solver module. Included for possible inheritance.

    Remaining args extracted from context:
        work        Work arrays
        FFT         Transform class from mpiFFT4py
        P_hat       Transformed pressure
        K           Scaled wavenumber mesh
        K2          sum_i K[i]*K[i]
        K_over_K2   K / K2
    
    """
    rhs = solver.conv(rhs, u_hat, work, FFT, K)
    rhs = solver.add_pressure_diffusion(rhs, u_hat, params.nu, K2, K, P_hat,
                                        K_over_K2)
    return rhs
