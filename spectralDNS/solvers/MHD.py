__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unused-variable,unused-argument

from shenfun import Basis, TensorProductSpace, VectorTensorProductSpace, \
    Array, Function, MixedTensorProductSpace
from .spectralinit import *
from .NS import end_of_tstep

def get_context():
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
    VM = MixedTensorProductSpace([T]*2*dim)

    mask = T.mask_nyquist() if params.mask_nyquist else None

    kw = {'padding_factor': 1.5 if params.dealias == '3/2-rule' else 1,
          'dealias_direct': params.dealias == '2/3-rule'}

    Vp = [Basis(params.N[i], 'F', domain=(0, params.L[i]),
                dtype=dtype(i), **kw) for i in range(dim)]

    Tp = TensorProductSpace(comm, Vp, dtype=float,
                            slab=(params.decomposition == 'slab'),
                            collapse_fourier=collapse_fourier, **kw0)
    VTp = VectorTensorProductSpace(Tp)
    VMp = MixedTensorProductSpace([Tp]*2*dim)

    # Mesh variables
    X = T.local_mesh(True)
    K = T.local_wavenumbers(scaled=True)
    for i in range(dim):
        X[i] = X[i].astype(float)
        K[i] = K[i].astype(float)
    K2 = np.zeros(T.shape(True), dtype=float)
    for i in range(dim):
        K2 += K[i]*K[i]

    # Set Nyquist frequency to zero on K that is, from now on, used for odd derivatives
    Kx = T.local_wavenumbers(scaled=True, eliminate_highest_freq=True)
    for i in range(dim):
        Kx[i] = Kx[i].astype(float)

    K_over_K2 = np.zeros(VT.shape(True), dtype=float)
    for i in range(dim):
        K_over_K2[i] = K[i] / np.where(K2 == 0, 1, K2)

    UB = Array(VM)
    P = Array(T)
    curl = Array(VT)
    UB_hat = Function(VM)
    P_hat = Function(T)
    dU = Function(VM)
    Source = Array(VM)
    ub_dealias = Array(VMp)
    ZZ_hat = np.zeros((3, 3) + Tp.shape(True), dtype=complex) # Work array

    # Create views into large data structures
    U = UB[:3]
    U_hat = UB_hat[:3]
    B = UB[3:]
    B_hat = UB_hat[3:]

    # Primary variable
    u = UB_hat

    hdf5file = MHDFile(config.params.solver,
                       checkpoint={'space': VM,
                                   'data': {'0': {'UB': [UB_hat]}}},
                       results={'space': VM,
                                'data': {'UB': [UB]}})

    return config.AttributeDict(locals())

class MHDFile(HDF5File):
    def update_components(self, UB, UB_hat, **kw):
        """Transform to real data when storing the solution"""
        UB = UB_hat.backward(UB)

def get_divergence(T, Kx, U_hat, **context):
    div_u = Array(T)
    div_u = T.backward(1j*(Kx[0]*U_hat[0]+Kx[1]*U_hat[1]+Kx[2]*U_hat[2]), div_u)
    return div_u

def set_Elsasser(c, ZZ, K):
    c[:3] = -1j*(K[0]*(ZZ[:, 0] + ZZ[0, :])
                 + K[1]*(ZZ[:, 1] + ZZ[1, :])
                 + K[2]*(ZZ[:, 2] + ZZ[2, :]))/2.0

    c[3:] = 1j*(K[0]*(ZZ[0, :] - ZZ[:, 0])
                + K[1]*(ZZ[1, :] - ZZ[:, 1])
                + K[2]*(ZZ[2, :] - ZZ[:, 2]))/2.0
    return c

def divergenceConvection(c, z0, z1, Tp, K, ZZ_hat):
    """Divergence convection using Elsasser variables
    z0=U+B
    z1=U-B
    """

    for i in range(3):
        for j in range(3):
            ZZ_hat[i, j] = Tp.forward(z0[i]*z1[j], ZZ_hat[i, j])

    c = set_Elsasser(c, ZZ_hat, K)
    return c

def getConvection(convection):

    if convection in ("Standard", "Vortex", "Skewed"):
        raise NotImplementedError

    elif convection == "Divergence":

        def Conv(rhs, ub_hat, Tp, VMp, K, ub_dealias, ZZ_hat):
            #ub_dealias = work[((6,)+Tp.shape(False), float, 0)]
            ub_dealias = VMp.backward(ub_hat, ub_dealias)
            u_dealias = ub_dealias[:3]
            b_dealias = ub_dealias[3:]
            # Compute convective term and place in dU
            rhs = divergenceConvection(rhs, u_dealias+b_dealias, u_dealias-b_dealias,
                                       Tp, K, ZZ_hat)
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
    for i in range(3):
        rhs[i] -= P_hat*K[i]

    # Add contribution from diffusion
    rhs[:3] -= nu*K2*u_hat
    rhs[3:] -= eta*K2*b_hat
    return rhs

def ComputeRHS(rhs, ub_hat, solver, Tp, VMp, K, Kx, K2, K_over_K2, P_hat,
               ub_dealias, ZZ_hat, mask, **context):
    """Return right hand side of Navier Stokes

    args:
        rhs         The right hand side to be returned
        ub_hat      The FFT of the velocity and magnetic fields at current
                    time. May differ from context.UB_hat since it is set by
                    the integrator
        solver      The solver module. Included for possible inheritance
                    and flexibility of integrators.

    Remaining args may be extracted from context:
        work        Work arrays
        K           Scaled wavenumber mesh
        Kx          Scaled wavenumber mesh with Nyquist eliminated
        K2          sum_i K[i]*K[i]
        K_over_K2   K / K2
        P_hat       Transfomred pressure

    """
    rhs = solver.conv(rhs, ub_hat, Tp, VMp, Kx, ub_dealias, ZZ_hat)
    if mask is not None:
        rhs *= mask
    rhs = solver.add_pressure_diffusion(rhs, ub_hat, params.nu, params.eta, K2,
                                        K, P_hat, K_over_K2)
    return rhs
