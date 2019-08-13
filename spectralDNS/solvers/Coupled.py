__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2019-6-14"
__copyright__ = "Copyright (C) 2019 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unbalanced-tuple-unpacking,unused-variable,function-redefined,unused-argument

from shenfun.spectralbase import inner_product
from shenfun.la import TDMA
from scipy.sparse.linalg import splu

from .spectralinit import *
from shenfun import TensorProductSpace, Array, TestFunction, TrialFunction, \
    MixedTensorProductSpace, div, grad, Dx, curl, inner, Function, Basis, \
    VectorTensorProductSpace, BlockMatrix, project
from ..shen.Matrices import HelmholtzCoeff


def get_context():
    """Set up context for solver"""

    # Get points and weights for Chebyshev weighted integrals
    assert params.Dquad == params.Bquad
    collapse_fourier = False if params.dealias == '3/2-rule' else True
    ST = Basis(params.N[0], 'C', bc=(0, 0), quad=params.Dquad)
    CT = Basis(params.N[0], 'C', quad=params.Dquad)
    CP = Basis(params.N[0], 'C', quad=params.Dquad)
    K0 = Basis(params.N[1], 'F', domain=(0, params.L[1]), dtype='D')
    K1 = Basis(params.N[2], 'F', domain=(0, params.L[2]), dtype='d')
    CP.slice = lambda: slice(0, CT.N)

    kw0 = {'threads': params.threads,
           'planner_effort': params.planner_effort["dct"],
           'slab': (params.decomposition == 'slab'),
           'collapse_fourier': collapse_fourier}
    FST = TensorProductSpace(comm, (ST, K0, K1), **kw0)    # Dirichlet
    FCT = TensorProductSpace(comm, (CT, K0, K1), **kw0)    # Regular Chebyshev N
    FCP = TensorProductSpace(comm, (CP, K0, K1), **kw0)    # Regular Chebyshev N-2
    VFS = VectorTensorProductSpace(FST)
    VCT = VectorTensorProductSpace(FCT)
    VQ = MixedTensorProductSpace([VFS, FCP])

    mask = FST.get_mask_nyquist() if params.mask_nyquist else None

    # Padded
    kw = {'padding_factor': 1.5 if params.dealias == '3/2-rule' else 1,
          'dealias_direct': params.dealias == '2/3-rule'}
    if params.dealias == '3/2-rule':
        # Requires new bases due to planning and transforms on different size arrays
        STp = Basis(params.N[0], 'C', bc=(0, 0), quad=params.Dquad)
        CTp = Basis(params.N[0], 'C', quad=params.Dquad)
    else:
        STp, CTp = ST, CT
    K0p = Basis(params.N[1], 'F', dtype='D', domain=(0, params.L[1]), **kw)
    K1p = Basis(params.N[2], 'F', dtype='d', domain=(0, params.L[2]), **kw)
    FSTp = TensorProductSpace(comm, (STp, K0p, K1p), **kw0)
    FCTp = TensorProductSpace(comm, (CTp, K0p, K1p), **kw0)
    VFSp = VectorTensorProductSpace(FSTp)
    VCp = MixedTensorProductSpace([FSTp, FCTp, FCTp])

    float, complex, mpitype = datatypes("double")

    constraints = ((3, 0, 0),
                   (3, params.N[0]-1, 0))

    # Mesh variables
    X = FST.local_mesh(True)
    x0, x1, x2 = FST.mesh()
    K = FST.local_wavenumbers(scaled=True)

    # Solution variables
    UP_hat = Function(VQ)
    UP_hat0 = Function(VQ)
    U_hat, P_hat = UP_hat
    U_hat0, P_hat0 = UP_hat0

    UP = Array(VQ)
    UP0 = Array(VQ)
    U, P = UP
    U0, P0 = UP0

    # primary variable
    u = UP_hat

    H_hat = Function(VFS)
    H_hat0 = Function(VFS)
    H_hat1 = Function(VFS)

    dU = Function(VQ)
    Source = Array(VFS) # Note - not using VQ. Only used for constant pressure gradient
    Sk = Function(VFS)

    K2 = K[1]*K[1]+K[2]*K[2]

    for i in range(3):
        K[i] = K[i].astype(float)

    work = work_arrays()
    u_dealias = Array(VFSp)
    curl_hat = Function(VCp)
    curl_dealias = Array(VCp)

    nu, dt, N = params.nu, params.dt, params.N

    up = TrialFunction(VQ)
    vq = TestFunction(VQ)

    ut, pt = up
    vt, qt = vq

    alfa = 2./nu/dt
    a0 = inner(vt, (2./nu/dt)*ut-div(grad(ut)))
    a1 = inner(vt, (2./nu)*grad(pt))
    a2 = inner(qt, (2./nu)*div(ut))

    M = BlockMatrix(a0+a1+a2)

    # Collect all matrices
    mat = config.AttributeDict(
        dict(CDD=inner_product((ST, 0), (ST, 1)),
             AB=HelmholtzCoeff(N[0], 1., alfa-K2, 0, ST.quad),))

    la = None

    hdf5file = CoupledFile(config.params.solver,
                        checkpoint={'space': VQ,
                                    'data': {'0': {'UP': [UP_hat]},
                                             '1': {'UP': [UP_hat0]}}},
                        results={'space': VFS,
                                 'data': {'U': [U]}})

    return config.AttributeDict(locals())

class CoupledFile(HDF5File):
    def update_components(self, U_hat, U, **context):
        """Transform to real data when storing the solution"""
        U = U_hat.backward(U)

assert params.precision == "double"

def end_of_tstep(context):
    """Function called at end of time step.

    If returning True, the while-loop in time breaks free. Used by adaptive
    solvers to modify the time stepsize. Used here to rotate solutions.
    """
    context.UP_hat0[:] = context.UP_hat
    context.H_hat1[:] = context.H_hat
    return False

def get_velocity(U, U_hat, **context):
    """Compute velocity from context"""
    U = U_hat.backward(U)
    return U

def set_velocity(U_hat, U, **context):
    """Set transformed velocity from context"""
    U_hat = U.forward(U_hat)
    return U_hat

def get_convection(H_hat, U_hat, K, VFSp, VCp, FSTp, FCTp, work, u_dealias, curl_dealias, curl_hat, mat, la, **context):
    """Compute convection from context"""
    conv_ = getConvection(params.convection)
    H_hat = conv_(H_hat, U_hat, K, VFSp, VCp, FSTp, FCTp, work, u_dealias, curl_dealias, curl_hat, mat, la)
    return H_hat

def get_divergence(U_hat, FST, mask, **context):
    div_hat = project(div(U_hat), FST)
    if mask is not None:
        div_hat.mask_nyquist(mask)
    div_ = Array(FST)
    div_ = div_hat.backward(div_)
    return div_

def Cross(c, a, b, FSTp, work):
    Uc = work[(a, 2, False)]
    Uc = cross1(Uc, a, b)
    c[0] = FSTp.forward(Uc[0], c[0])
    c[1] = FSTp.forward(Uc[1], c[1])
    c[2] = FSTp.forward(Uc[2], c[2])
    return c

def compute_curl(curl_dealias, u_hat, VCp, curl_hat, work, K):
    curl_hat[:] = 0
    curl_hat = project(curl(u_hat), VCp, output_array=curl_hat)
    curl_dealias = curl_hat.backward(curl_dealias)
    return curl_dealias

def standardConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp, FCTp, work,
                       mat, la):
    rhs[:] = 0
    U = u_dealias
    Uc = work[(U, 1, True)]
    Uc2 = work[(U, 2, True)]
    F_tmp = work[(rhs, 0, True)]

    # dudx = 0 on walls from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    dudx = project(Dx(u_hat[0], 0, 1), FSTp).backward()
    dvdx = project(Dx(u_hat[1], 0, 1), FCTp).backward()
    dwdx = project(Dx(u_hat[2], 0, 1), FCTp).backward()

    dudy = Uc2[0] = FSTp.backward(1j*K[1]*u_hat[0], Uc2[0])
    dudz = Uc2[1] = FSTp.backward(1j*K[2]*u_hat[0], Uc2[1])
    rhs[0] = FSTp.forward(U[0]*dudx + U[1]*dudy + U[2]*dudz, rhs[0])

    Uc2[:] = 0
    dvdy = Uc2[0] = FSTp.backward(1j*K[1]*u_hat[1], Uc2[0])
    dvdz = Uc2[1] = FSTp.backward(1j*K[2]*u_hat[1], Uc2[1])
    rhs[1] = FSTp.forward(U[0]*dvdx + U[1]*dvdy + U[2]*dvdz, rhs[1])

    Uc2[:] = 0
    dwdy = Uc2[0] = FSTp.backward(1j*K[1]*u_hat[2], Uc2[0])
    dwdz = Uc2[1] = FSTp.backward(1j*K[2]*u_hat[2], Uc2[1])

    rhs[2] = FSTp.forward(U[0]*dwdx + U[1]*dwdy + U[2]*dwdz, rhs[2])

    return rhs

def divergenceConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp, FCTp, work,
                         mat, la, add=False):
    """c_i = div(u_i u_j)"""
    if not add:
        rhs.fill(0)
    F_tmp = Function(VFSp, buffer=work[(rhs, 0, True)])
    F_tmp2 = Function(VFSp, buffer=work[(rhs, 1, True)])
    U = u_dealias

    F_tmp[0] = FSTp.forward(U[0]*U[0], F_tmp[0])
    F_tmp[1] = FSTp.forward(U[0]*U[1], F_tmp[1])
    F_tmp[2] = FSTp.forward(U[0]*U[2], F_tmp[2])

    F_tmp2 = project(Dx(F_tmp, 0, 1), VFSp, output_array=F_tmp2)
    rhs += F_tmp2

    F_tmp2[0] = FSTp.forward(U[0]*U[1], F_tmp2[0])
    F_tmp2[1] = FSTp.forward(U[0]*U[2], F_tmp2[1])
    rhs[0] += 1j*K[1]*F_tmp2[0] # duvdy
    rhs[0] += 1j*K[2]*F_tmp2[1] # duwdz

    F_tmp[0] = FSTp.forward(U[1]*U[1], F_tmp[0])
    F_tmp[1] = FSTp.forward(U[1]*U[2], F_tmp[1])
    F_tmp[2] = FSTp.forward(U[2]*U[2], F_tmp[2])
    rhs[1] += 1j*K[1]*F_tmp[0]  # dvvdy
    rhs[1] += 1j*K[2]*F_tmp[1]  # dvwdz
    rhs[2] += 1j*K[1]*F_tmp[1]  # dvwdy
    rhs[2] += 1j*K[2]*F_tmp[2]  # dwwdz

    return rhs

def getConvection(convection):

    if convection == "Standard":

        def Conv(rhs, u_hat, K, VFSp, VCp, FSTp, FCTp, work, u_dealias, curl_dealias, curl_hat, mat, la):
            u_dealias = VFSp.backward(u_hat, u_dealias)
            rhs = standardConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                     FCTp, work, mat, la)
            rhs *= -1
            return rhs

    elif convection == "Divergence":

        def Conv(rhs, u_hat, K, VFSp, VCp, FSTp, FCTp, work, u_dealias, curl_dealias, curl_hat, mat, la):
            u_dealias = VFSp.backward(u_hat, u_dealias)
            rhs = divergenceConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                       FCTp, work, mat, la, False)
            rhs *= -1
            return rhs

    elif convection == "Skew":

        def Conv(rhs, u_hat, K, VFSp, VCp, FSTp, FCTp, work, u_dealias, curl_dealias, curl_hat, mat, la):
            u_dealias = VFSp.backward(u_hat, u_dealias)
            rhs = standardConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                       FCTp, work, mat, la)
            rhs = divergenceConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                       FCTp, work, mat, la, True)
            rhs *= -0.5
            return rhs

    elif convection == "Vortex":

        def Conv(rhs, u_hat, K, VFSp, VCp, FSTp, FCTp, work, u_dealias, curl_dealias, curl_hat, mat, la):
            u_dealias = VFSp.backward(u_hat, u_dealias)
            curl_dealias = compute_curl(curl_dealias, u_hat, VCp, curl_hat, work, K)
            rhs = Cross(rhs, u_dealias, curl_dealias, FSTp, work)
            return rhs

    Conv.convection = convection
    return Conv

@optimizer
def assembleAB(H_hat0, H_hat, H_hat1):
    H_hat0[:] = 1.5*H_hat - 0.5*H_hat1
    return H_hat0

def ComputeRHS(rhs, u_hat, solver,
               H_hat, H_hat1, H_hat0, VFSp, FSTp, FCTp, VCp, work, K, K2,
               u_dealias, curl_dealias, curl_hat, mat, la, vt, Sk, mask, **context):
    """Compute right hand side of Navier Stokes

    Parameters
    ----------
        rhs : array
            The right hand side to be returned
        u_hat : array
            The velocity vector at current time
        solver : module
            The current solver module

    Remaining args are extracted from context

    """
    # Nonlinear convection term at current u_hat
    H_hat = solver.conv(H_hat, u_hat, K, VFSp, VCp, FSTp, FCTp, work,
                        u_dealias, curl_dealias, curl_hat, mat, la)

    # Assemble convection with Adams-Bashforth at time = n+1/2
    H_hat0 = solver.assembleAB(H_hat0, H_hat, H_hat1)

    if mask is not None:
        H_hat0.mask_nyquist(mask)

    # Assemble rhs
    rhs_u, rhs_p = rhs
    rhs_u[0] = mat.AB.matvec(u_hat[0], rhs_u[0])
    rhs_u[1] = mat.AB.matvec(u_hat[1], rhs_u[1])
    rhs_u[2] = mat.AB.matvec(u_hat[2], rhs_u[2])

    # Convection
    rhs_u[:] += 2./params.nu*inner(vt, H_hat0)

    # Source
    rhs_u[1] -= 2./params.nu*Sk[1]
    return rhs


def integrate(up_hat, rhs, dt, solver, context):
    """Regular implicit solver for KMM channel solver"""
    u_hat, p_hat = up_hat
    rhs[:] = 0
    rhs = solver.ComputeRHS(rhs, u_hat, solver, **context)
    up_hat = context.M.solve(rhs, u=up_hat, constraints=context.constraints)
    if rank == 0:
        u_hat[0, :, 0, 0] = 0

    return up_hat, dt, dt

def getintegrator(rhs, u0, solver, context):
    def func():
        return solver.integrate(u0, rhs, params.dt, solver, context)
    return func
