__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unused-variable,unused-argument,function-redefined

from shenfun import Basis, TensorProductSpace, VectorTensorProductSpace, \
    Array, Function
from .spectralinit import *

def get_context():
    """Set up context for classical (NS) solver"""
    float, complex, mpitype = datatypes(params.precision)
    collapse_fourier = False if params.dealias == '3/2-rule' else True
    dim = len(params.N)
    dtype = lambda d: float if d == dim-1 else complex
    V = [Basis(params.N[i], 'F', domain=(0, params.L[i]),
               dtype=dtype(i)) for i in range(dim)]

    kw0 = {'threads': params.threads,
           'planner_effort': params.planner_effort['fft']}
    T = TensorProductSpace(comm, V, dtype=float,
                           slab=(params.decomposition == 'slab'),
                           collapse_fourier=collapse_fourier, **kw0)
    VT = VectorTensorProductSpace(T)

    kw = {'padding_factor': 1.5 if params.dealias == '3/2-rule' else 1,
          'dealias_direct': params.dealias == '2/3-rule'}

    Vp = [Basis(params.N[i], 'F', domain=(0, params.L[i]),
                dtype=dtype(i), **kw) for i in range(dim)]

    Tp = TensorProductSpace(comm, Vp, dtype=float,
                            slab=(params.decomposition == 'slab'),
                            collapse_fourier=collapse_fourier, **kw0)
    VTp = VectorTensorProductSpace(Tp)

    # Mesh variables
    X = T.local_mesh(True)
    K = T.local_wavenumbers(scaled=True)
    for i in range(dim):
        X[i] = X[i].astype(float)
        K[i] = K[i].astype(float)
    K2 = np.zeros(T.local_shape(True), dtype=float)
    for i in range(dim):
        K2 += K[i]*K[i]

    # Set Nyquist frequency to zero on K that is, from now on, used for odd derivatives
    Kx = T.local_wavenumbers(scaled=True, eliminate_highest_freq=True)
    for i in range(dim):
        Kx[i] = Kx[i].astype(float)

    K_over_K2 = np.zeros(VT.local_shape(), dtype=float)
    for i in range(dim):
        K_over_K2[i] = K[i] / np.where(K2 == 0, 1, K2)

    # Velocity and pressure
    U = Array(VT)
    U_hat = Function(VT)
    P = Array(T)
    P_hat = Function(T)

    # Primary variable
    u = U_hat

    # RHS array
    dU = Function(VT)
    curl = Array(VT)
    Source = Function(VT) # Possible source term initialized to zero
    work = work_arrays()

    hdf5file = NSFile(config.params.solver,
                      checkpoint={'space': VT,
                                  'data': {'0': {'U': [U_hat]}}},
                      results={'space': VT,
                               'data': {'U': [U], 'P': [P]}})

    return config.AttributeDict(locals())

class NSFile(HDF5File):
    """Subclass HDF5File for appropriate updating of real components

    The method 'update_components' is used to transform all variables
    that are to be stored. If more variables than U and P are
    wanted, then subclass HDF5Writer in the application.
    """
    def update_components(self, **context):
        """Transform to real data before storing the solution"""
        U = get_velocity(**context)
        P = get_pressure(**context)

def get_curl(curl, U_hat, work, VT, VTp, K, **context):
    """Compute curl from context"""
    curl = compute_curl(curl, U_hat, work, VT, VTp, K)
    return curl

def get_velocity(U, U_hat, **context):
    """Compute velocity from context"""
    U = U_hat.backward(U)
    return U

def get_pressure(P, P_hat, T, **context):
    """Compute pressure from context"""
    P = T.backward(1j*P_hat, P)
    return P

def set_velocity(U, U_hat, **context):
    """Compute velocity from context"""
    U_hat = U.forward(U_hat)
    return U_hat

def get_divergence(T, K, U_hat, **context):
    div_u = Array(T)
    div_u = T.backward(1j*(K[0]*U_hat[0]+K[1]*U_hat[1]+K[2]*U_hat[2]), div_u)
    return div_u

def end_of_tstep(context):
    # Make sure that the last step hits T exactly.
    # Used by adaptive solvers
    if abs(params.t - params.T) < 1e-12:
        return True

    if (abs(params.t + params.dt - params.T) < 1e-12 or
            params.t + params.dt >= params.T + 1e-12):
        params.dt = params.T - params.t

    return False

def compute_curl(c, a, work, VT, VTp, K, dealias=None):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    curl_hat = work[(a, 0, False)]
    curl_hat = cross2(curl_hat, K, a)
    V = VT if dealias is None else VTp
    c = V.backward(curl_hat, c)
    return c

def Cross(c, a, b, work, VT, VTp, dealias=None):
    """c_k = F_k(a x b)"""
    V = VT if dealias is None else VTp
    Uc = work[(a, 2, False)]
    Uc = cross1(Uc, a, b)
    c = V.forward(Uc, c)
    return c

def standard_convection(rhs, u_dealias, U_hat, work, T, Tp, K, dealias=None):
    """rhs_i = u_j du_i/dx_j"""
    gradUi = work[(u_dealias, 2, False)]
    T = T if dealias is None else Tp
    for i in range(3):
        for j in range(3):
            gradUi[j] = T.backward(1j*K[j]*U_hat[i], gradUi[j])
        rhs[i] = T.forward(np.sum(u_dealias*gradUi, 0), rhs[i])
    return rhs

def divergence_convection(rhs, u_dealias, work, T, Tp, K, dealias=None, add=False):
    """rhs_i = div(u_i u_j)"""
    if not add:
        rhs.fill(0)
    UUi_hat = work[(rhs, 0, False)]
    T = T if dealias is None else Tp
    for i in range(3):
        UUi_hat[i] = T.forward(u_dealias[0]*u_dealias[i], UUi_hat[i])
    rhs[0] += 1j*(K[0]*UUi_hat[0] + K[1]*UUi_hat[1] + K[2]*UUi_hat[2])
    rhs[1] += 1j*K[0]*UUi_hat[1]
    rhs[2] += 1j*K[0]*UUi_hat[2]
    UUi_hat[0] = T.forward(u_dealias[1]*u_dealias[1], UUi_hat[0])
    UUi_hat[1] = T.forward(u_dealias[1]*u_dealias[2], UUi_hat[1])
    UUi_hat[2] = T.forward(u_dealias[2]*u_dealias[2], UUi_hat[2])
    rhs[1] += (1j*K[1]*UUi_hat[0] + 1j*K[2]*UUi_hat[1])
    rhs[2] += (1j*K[1]*UUi_hat[1] + 1j*K[2]*UUi_hat[2])
    return rhs

def getConvection(convection):

    if convection == "Standard":

        def Conv(rhs, u_hat, work, T, Tp, VT, VTp, K):
            u_dealias = work[((3,)+Tp.backward.output_array.shape,
                              Tp.backward.output_array.dtype, 0, False)]
            u_dealias = VTp.backward(u_hat, u_dealias)
            rhs = standard_convection(rhs, u_dealias, u_hat, work, T, Tp, K, params.dealias)
            rhs[:] *= -1
            return rhs

    elif convection == "Divergence":

        def Conv(rhs, u_hat, work, T, Tp, VT, VTp, K):
            u_dealias = work[((3,)+Tp.backward.output_array.shape,
                              Tp.backward.output_array.dtype, 0, False)]
            u_dealias = VTp.backward(u_hat, u_dealias)
            rhs = divergence_convection(rhs, u_dealias, work, T, Tp, K, params.dealias, False)
            rhs[:] *= -1
            return rhs

    elif convection == "Skewed":

        def Conv(rhs, u_hat, work, T, Tp, VT, VTp, K):
            u_dealias = work[((3,)+Tp.backward.output_array.shape,
                              Tp.backward.output_array.dtype, 0, False)]
            u_dealias = VTp.backward(u_hat, u_dealias)
            rhs = standard_convection(rhs, u_dealias, u_hat, work, T, Tp, K, params.dealias)
            rhs = divergence_convection(rhs, u_dealias, work, T, Tp, K, params.dealias, True)
            rhs *= -0.5
            return rhs

    elif convection == "Vortex":

        def Conv(rhs, u_hat, work, T, Tp, VT, VTp, K):
            u_dealias = work[(VTp.local_shape(False),
                              VTp.backward.output_array.dtype, 0, False)]
            curl_dealias = work[(VTp.local_shape(False),
                                 VTp.backward.output_array.dtype, 1, False)]
            u_dealias = VTp.backward(u_hat, u_dealias)
            curl_dealias = compute_curl(curl_dealias, u_hat, work, VT, VTp, K, params.dealias)
            rhs = Cross(rhs, u_dealias, curl_dealias, work, VT, VTp, params.dealias)
            return rhs

    Conv.convection = convection
    return Conv

@optimizer
def add_pressure_diffusion(rhs, u_hat, nu, K2, K, P_hat, K_over_K2):
    """Add contributions from pressure and diffusion to the rhs"""

    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat = np.sum(rhs*K_over_K2, 0, out=P_hat)

    # Subtract pressure gradient
    for i in range(rhs.shape[0]):
        rhs[i] -= P_hat*K[i]

    # Subtract contribution from diffusion
    rhs -= nu*K2*u_hat

    return rhs

def ComputeRHS(rhs, u_hat, solver, work, T, Tp, VT, VTp, P_hat, K, Kx, K2,
               K_over_K2, Source, **context):
    """Compute right hand side of Navier Stokes

    Parameters
    ----------
        rhs : array
            The right hand side to be returned
        u_hat : array
            The FFT of the velocity at current time. May differ from
            context.U_hat since it is set by the integrator
        solver : module
            The solver module. Included for possible inheritance
            and flexibility of integrators.

    Other Parameters
    ----------------
        work : dict
            Work arrays
        T : TensorProductSpace
        Tp : TensorProductSpace
            for padded transforms
        VT : VectorTensorProductSpace
        VTp : VectorTensorProductSpace
            for padded transforms
        P_hat : array
            Transformed pressure
        K : list of arrays
            Scaled wavenumber mesh
        Kx : list of arrays
            Scaled wavenumber mesh with Nyquist eliminated
        K2 : array
            sum_i K[i]*K[i]
        K_over_K2 : array
            K / K2

    """
    rhs = solver.conv(rhs, u_hat, work, T, Tp, VT, VTp, Kx)
    rhs = solver.add_pressure_diffusion(rhs, u_hat, params.nu, K2, K, P_hat,
                                        K_over_K2)
    rhs += Source
    return rhs
