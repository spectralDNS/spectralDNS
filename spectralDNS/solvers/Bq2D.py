#pylint: disable=unused-variable,unused-argument,unused-import,function-redefined

__author__ = "Mikael Mortensen <mikaem@math.uio.no> and Diako Darian <diako.darian@mn.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from shenfun import MixedTensorProductSpace
from .NS2D import *

def get_context():
    """Set up context for Bq2D solver"""
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
    VM = MixedTensorProductSpace([T]*(dim+1))

    kw = {'padding_factor': 1.5 if params.dealias == '3/2-rule' else 1,
          'dealias_direct': params.dealias == '2/3-rule'}

    Vp = [Basis(params.N[i], 'F', domain=(0, params.L[i]),
                dtype=dtype(i), **kw) for i in range(dim)]

    Tp = TensorProductSpace(comm, Vp, dtype=float,
                            slab=(params.decomposition == 'slab'),
                            collapse_fourier=collapse_fourier, **kw0)
    VTp = VectorTensorProductSpace(Tp)
    VMp = MixedTensorProductSpace([Tp]*(dim+1))

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

    K_over_K2 = np.zeros(VT.shape(), dtype=float)
    for i in range(dim):
        K_over_K2[i] = K[i] / np.where(K2 == 0, 1, K2)

    # Solution variables
    Ur = Array(VM)
    Ur_hat = Function(VM)
    P = Array(T)
    P_hat = Function(T)
    curl = Array(T)
    W_hat = Function(T)
    ur_dealias = Array(VMp)

    # Create views into large data structures
    rho = Ur[2]
    rho_hat = Ur_hat[2]
    U = Ur[:2]
    U_hat = Ur_hat[:2]

    # Primary variable
    u = Ur_hat

    # RHS and work arrays
    dU = Function(VM)
    work = work_arrays()

    hdf5file = BqFile(config.params.solver,
                      checkpoint={'space': VM,
                                  'data': {'0': {'Ur': [Ur_hat]}}},
                      results={'space': VM,
                               'data': {'UR': [Ur]}})

    return config.AttributeDict(locals())

class BqFile(HDF5File):
    def update_components(self, Ur, Ur_hat, **context):
        """Transform to real data before storing the solution"""
        Ur = Ur_hat.backward(Ur)

def get_Ur(Ur, Ur_hat, **context):
    """Compute and return Ur from context"""
    Ur = Ur_hat.backward(Ur)
    return Ur

def get_rho(Ur, Ur_hat, **context):
    """Compute and return rho from context"""
    Ur[2] = Ur_hat[2].backward(Ur[2])
    return Ur[2]

def get_velocity(U, U_hat, **context):
    """Compute and return velocity from context"""
    U[0] = U_hat[0].backward(U[0])
    U[1] = U_hat[1].backward(U[1])
    return U

def getConvection(convection):
    """Return function used to compute nonlinear term"""
    if convection in ("Standard", "Divergence", "Skewed"):
        raise NotImplementedError

    elif convection == "Vortex":

        def Conv(rhs, ur_hat, work, T, Tp, VM, VMp, K, ur_dealias):
            curl_dealias = work[(ur_dealias[0], 0, False)]
            F_tmp = work[(rhs, 0, True)]

            ur_dealias = VMp.backward(ur_hat, ur_dealias)

            u_dealias = ur_dealias[:2]
            rho_dealiased = ur_dealias[2]

            F_tmp[0] = cross2(F_tmp[0], K, ur_hat[:2])
            curl_dealias = Tp.backward(F_tmp[0], curl_dealias)
            rhs[0] = Tp.forward(u_dealias[1]*curl_dealias, rhs[0])
            rhs[1] = Tp.forward(-u_dealias[0]*curl_dealias, rhs[1])

            F_tmp[0] = Tp.forward(u_dealias[0]*rho_dealiased, F_tmp[0])
            F_tmp[1] = Tp.forward(u_dealias[1]*rho_dealiased, F_tmp[1])
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

def ComputeRHS(rhs, ur_hat, solver, work, K, Kx, K2, K_over_K2, P_hat, T, Tp,
               VM, VMp, ur_dealias, **context):
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
        K           Scaled wavenumber mesh
        Kx          Scaled wavenumber mesh with Nyquist eliminated
        K2          K[0]*K[0] + K[1]*K[1] + K[2]*K[2]
        K_over_K2   K / K2
        P_hat       Transformed pressure

    """
    rhs = solver.conv(rhs, ur_hat, work, T, Tp, VM, VMp, Kx, ur_dealias)
    rhs = solver.add_pressure_diffusion(rhs, ur_hat, P_hat, K_over_K2, K, K2,
                                        params.nu, params.Ri, params.Pr)
    return rhs
