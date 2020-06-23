__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2018-10-23"
__copyright__ = "Copyright (C) 2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unbalanced-tuple-unpacking,unused-variable,function-redefined,unused-argument

from shenfun.spectralbase import inner_product
from shenfun.la import TDMA
from shenfun import TensorProductSpace, Array, TestFunction, TrialFunction, \
    MixedTensorProductSpace, div, grad, Dx, inner, Function, FunctionSpace
from shenfun.chebyshev.la import Helmholtz, Biharmonic

from .spectralinit import *
from ..shen.Matrices import BiharmonicCoeff, HelmholtzCoeff
from ..shen import LUsolve

def get_context():
    """Set up context for solver"""

    # Get points and weights for Chebyshev weighted integrals
    assert params.Dquad == params.Bquad
    collapse_fourier = False if params.dealias == '3/2-rule' else True
    ST = FunctionSpace(params.N[2], 'C', bc=(0, 0), quad=params.Dquad)
    SB = FunctionSpace(params.N[2], 'C', bc='Biharmonic', quad=params.Bquad)
    CT = FunctionSpace(params.N[2], 'C', quad=params.Dquad)
    ST0 = FunctionSpace(params.N[2], 'C', bc=(0, 0), quad=params.Dquad) # For 1D problem
    K0 = FunctionSpace(params.N[0], 'F', domain=(0, params.L[0]), dtype='D')
    K1 = FunctionSpace(params.N[1], 'F', domain=(0, params.L[1]), dtype='d')

    kw0 = {'threads':params.threads,
           'planner_effort':params.planner_effort["dct"],
           'slab': (params.decomposition == 'slab'),
           'collapse_fourier': collapse_fourier}
    FST = TensorProductSpace(comm, (K0, K1, ST), axes=(2, 0, 1), **kw0)    # Dirichlet
    FSB = TensorProductSpace(comm, (K0, K1, SB), axes=(2, 0, 1), **kw0)    # Biharmonic
    FCT = TensorProductSpace(comm, (K0, K1, CT), axes=(2, 0, 1), **kw0)    # Regular Chebyshev
    VFS = MixedTensorProductSpace([FST, FST, FSB])
    VFST = MixedTensorProductSpace([FST, FST, FST])
    VUG = MixedTensorProductSpace([FST, FSB])

    mask = FST.get_mask_nyquist() if params.mask_nyquist else None

    # Padded
    kw = {'padding_factor': 1.5 if params.dealias == '3/2-rule' else 1,
          'dealias_direct': params.dealias == '2/3-rule'}
    if params.dealias == '3/2-rule':
        # Requires new bases due to planning and transforms on different size arrays
        STp = FunctionSpace(params.N[2], 'C', bc=(0, 0), quad=params.Dquad)
        SBp = FunctionSpace(params.N[2], 'C', bc='Biharmonic', quad=params.Bquad)
        CTp = FunctionSpace(params.N[2], 'C', quad=params.Dquad)
    else:
        STp, SBp, CTp = ST, SB, CT
    K0p = FunctionSpace(params.N[0], 'F', dtype='D', domain=(0, params.L[0]), **kw)
    K1p = FunctionSpace(params.N[1], 'F', dtype='d', domain=(0, params.L[1]), **kw)
    FSTp = TensorProductSpace(comm, (K0p, K1p, STp), axes=(2, 0, 1), **kw0)
    FSBp = TensorProductSpace(comm, (K0p, K1p, SBp), axes=(2, 0, 1), **kw0)
    FCTp = TensorProductSpace(comm, (K0p, K1p, CTp), axes=(2, 0, 1), **kw0)
    VFSp = MixedTensorProductSpace([FSTp, FSTp, FSBp])

    float, complex, mpitype = datatypes("double")

    # Mesh variables
    X = FST.local_mesh(True)
    x0, x1, x2 = FST.mesh()
    K = FST.local_wavenumbers(scaled=True)

    # Solution variables
    U = Array(VFS)
    U0 = Array(VFS)
    U_hat = Function(VFS)
    U_hat0 = Function(VFS)
    g = Function(FST)

    # primary variable
    u = (U_hat, g)

    H_hat = Function(VFST)
    H_hat0 = Function(VFST)
    H_hat1 = Function(VFST)

    dU = Function(VUG)
    hv = Function(FST)
    hg = Function(FST)
    Source = Array(VFS)
    Sk = Function(VFS)

    K2 = K[0]*K[0]+K[1]*K[1]
    K4 = K2**2

    # Set Nyquist frequency to zero on K that is used for odd derivatives in nonlinear terms
    Kx = FST.local_wavenumbers(scaled=True, eliminate_highest_freq=True)
    K_over_K2 = np.zeros((2,)+g.shape)
    for i in range(2):
        K_over_K2[i] = K[i] / np.where(K2 == 0, 1, K2)

    for i in range(3):
        K[i] = K[i].astype(float)
        Kx[i] = Kx[i].astype(float)

    Kx2 = Kx[1]*Kx[1]+Kx[2]*Kx[2]
    work = work_arrays()
    u_dealias = Array(VFSp)
    u0_hat = np.zeros((2, params.N[2]), dtype=complex)
    h0_hat = np.zeros((2, params.N[2]), dtype=complex)
    w = np.zeros((params.N[2], ), dtype=complex)
    w1 = np.zeros((params.N[2], ), dtype=complex)

    nu, dt, N = params.nu, params.dt, params.N

    # Collect all matrices
    mat = config.AttributeDict(
        dict(CDD=inner_product((ST, 0), (ST, 1)),
             CTD=inner_product((CT, 0), (ST, 1)),
             BTT=inner_product((CT, 0), (CT, 0)),
             AB=HelmholtzCoeff(N[2], 1.0, -(K2 - 2.0/nu/dt), 2, ST.quad),
             AC=BiharmonicCoeff(N[2], nu*dt/2., (1. - nu*dt*K2), -(K2 - nu*dt/2.*K4), 2, SB.quad),
             # Matrices for biharmonic equation
             CBD=inner_product((SB, 0), (ST, 1)),
             ABB=inner_product((SB, 0), (SB, 2)),
             BBB=inner_product((SB, 0), (SB, 0)),
             SBB=inner_product((SB, 0), (SB, 4)),
             # Matrices for Helmholtz equation
             ADD=inner_product((ST, 0), (ST, 2)),
             BDD=inner_product((ST, 0), (ST, 0)),
             BBD=inner_product((SB, 0), (ST, 0)),
             CDB=inner_product((ST, 0), (SB, 1)),
             ADD0=inner_product((ST0, 0), (ST0, 2)),
             BDD0=inner_product((ST0, 0), (ST0, 0))))

    la = config.AttributeDict(
        dict(HelmholtzSolverG=Helmholtz(mat.ADD, mat.BDD, -np.ones((1, 1, 1)),
                                        (K2+2.0/nu/dt)),
             BiharmonicSolverU=Biharmonic(mat.SBB, mat.ABB, mat.BBB, -nu*dt/2.*np.ones((1, 1, 1)),
                                          (1.+nu*dt*K2),
                                          (-(K2 + nu*dt/2.*K4))),
             HelmholtzSolverU0=Helmholtz(mat.ADD0, mat.BDD0, np.array([-1.]), np.array([2./nu/dt])),
             TDMASolverD=TDMA(inner_product((ST, 0), (ST, 0)))))

    hdf5file = KMMFile(config.params.solver,
                       checkpoint={'space': VFS,
                                   'data': {'0': {'U': [U_hat]},
                                            '1': {'U': [U_hat0]}}},
                       results={'space': VFS,
                                'data': {'U': [U]}})

    return config.AttributeDict(locals())

class KMMFile(HDF5File):
    def update_components(self, U_hat, U, **context):
        """Transform to real data when storing the solution"""
        U = U_hat.backward(U)

assert params.precision == "double"

def end_of_tstep(context):
    """Function called at end of time step.

    If returning True, the while-loop in time breaks free. Used by adaptive
    solvers to modify the time stepsize. Used here to rotate solutions.
    """
    context.U_hat0[:] = context.U_hat
    context.H_hat1[:] = context.H_hat
    return False

def get_velocity(U, U_hat, VFS, **context):
    """Compute velocity from context"""
    U = VFS.backward(U_hat, U)
    return U

def set_velocity(U_hat, U, VFS, **context):
    """Set transformed velocity from context"""
    U_hat = VFS.forward(U, U_hat)
    return U_hat

def get_curl(curl, U_hat, g, work, FCTp, FSTp, FSBp, Kx, **context):
    """Compute curl from context"""
    curl = compute_curl(curl, U_hat, g, Kx, FCTp, FSTp, FSBp, work)
    return curl

def get_convection(H_hat, U_hat, g, Kx, VFSp, FSTp, FSBp, FCTp, work, mat, la, u_dealias, **context):
    """Compute convection from context"""
    conv_ = getConvection(params.convection)
    H_hat = conv_(H_hat, U_hat, g, Kx, VFSp, FSTp, FSBp, FCTp, work, mat, la, u_dealias)
    return H_hat

def get_pressure(context, solver):
    FCT = context.FCT
    FST = context.FST

    U = solver.get_velocity(**context)
    U0 = context.VFS.backward(context.U_hat0, context.U0)
    dt = solver.params.dt

    H_hat = solver.get_convection(**context)
    Hx = Array(FST)
    Hx = FST.backward(H_hat[0], Hx)

    v = TestFunction(FCT)
    p = TrialFunction(FCT)
    U = U.as_function()
    U0 = U0.as_function()

    rhs_hat = inner((0.5*context.nu)*div(grad(U[0]+U0[0])), v)
    Hx -= 1./dt*(U[0]-U0[0])
    rhs_hat += inner(Hx, v)

    CT = inner(Dx(p, 0), v)

    # Should implement fast solver. Just a backwards substitution
    A = CT.diags().toarray()*CT.scale[0]
    A[-1, 0] = 1
    a_i = np.linalg.inv(A)

    p_hat = Function(context.FCT)

    for j in range(p_hat.shape[1]):
        for k in range(p_hat.shape[2]):
            p_hat[:, j, k] = np.dot(a_i, rhs_hat[:, j, k])

    p = Array(FCT)
    p = FCT.backward(p_hat, p)

    uu = np.sum((0.5*(U+U0))**2, 0)
    uu *= 0.5

    return p-uu+3./16.

def get_divergence(U, U_hat, FST, K, Kx, work, la, mat, **context):
    Uc_hat = work[(U_hat[0], 0, True)]
    Uc = work[(U, 2, True)]
    Uc_hat = mat.CDB.matvec(U_hat[2], Uc_hat, axis=2)
    Uc_hat = la.TDMASolverD(Uc_hat, axis=2)

    dwdz = Uc[2] = FST.backward(Uc_hat, Uc[2])
    dudx_h = 1j*K[0]*U_hat[0]
    dudx = Uc[0] = FST.backward(dudx_h, Uc[0])
    dvdy_h = 1j*K[1]*U_hat[1]
    dvdy = Uc[1] = FST.backward(dvdy_h, Uc[1])
    return dudx+dvdy+dwdz

#@profile
def Cross(c, a, b, FSTp, work):
    Uc = work[(a, 2, False)]
    Uc = cross1(Uc, a, b)
    c[0] = FSTp.forward(Uc[0], c[0])
    c[1] = FSTp.forward(Uc[1], c[1])
    c[2] = FSTp.forward(Uc[2], c[2])
    return c

def compute_curl(c, u_hat, g, K, FCTp, FSTp, FSBp, work):
    F_tmp = work[(u_hat, 0, False)]
    F_tmp2 = work[(u_hat, 2, False)]
    Uc = work[(c, 2, False)]
    # Mult_CTD_3D is projection to T of d(u_hat)/dz (for components 0 and 1 of u_hat)
    # Corresponds to CTD.matvec(u_hat[0])/BTT.dd, CTD.matvec(u_hat[1])/BTT.dd
    #LUsolve.Mult_CTD_3D_n(params.N[2], u_hat[0], u_hat[1], F_tmp[0], F_tmp[1], 2)
    LUsolve.Mult_CTD_3D_ptr(params.N[2], u_hat[0], u_hat[1], F_tmp[0], F_tmp[1], 2)

    dudz = Uc[0] = FCTp.backward(F_tmp[0], Uc[0])
    dvdz = Uc[1] = FCTp.backward(F_tmp[1], Uc[1])
    c[2] = FSTp.backward(g, c[2])
    dwdy = F_tmp2[0] = 1j*K[1]*u_hat[2]
    dwdx = F_tmp2[1] = 1j*K[0]*u_hat[2]

    c[0] = FSBp.backward(F_tmp2[0], c[0])
    c[0] -= dvdz
    c[1] = FSBp.backward(-F_tmp2[1], c[1])
    c[1] += dudz
    return c

def compute_derivatives(U, U_hat, FST, FCT, FSB, K, la, mat, work, **context):
    duidxj = np.zeros((3, 3)+U.shape[1:])
    F_tmp = work[(U_hat, 0, True)]
    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = mat.CDB.matvec(U_hat[0], F_tmp[0])
    F_tmp[0] = la.TDMASolverD(F_tmp[0])
    duidxj[0, 0] = FST.backward(F_tmp[0], duidxj[0, 0])
    LUsolve.Mult_CTD_3D_n(params.N[0], U_hat[1], U_hat[2], F_tmp[1], F_tmp[2])
    duidxj[1, 0] = dvdx = FCT.backward(F_tmp[1], duidxj[1, 0])  # proj to Cheb
    duidxj[2, 0] = dwdx = FCT.backward(F_tmp[2], duidxj[2, 0])  # proj to Cheb
    duidxj[0, 1] = dudy = FSB.backward(1j*K[1]*U_hat[0], duidxj[0, 1]) # ShenB
    duidxj[0, 2] = dudz = FSB.backward(1j*K[2]*U_hat[0], duidxj[0, 2])
    duidxj[1, 1] = dvdy = FST.backward(1j*K[1]*U_hat[1], duidxj[1, 1])
    duidxj[1, 2] = dvdz = FST.backward(1j*K[2]*U_hat[1], duidxj[1, 2])
    duidxj[2, 1] = dwdy = FST.backward(1j*K[1]*U_hat[2], duidxj[2, 1])
    duidxj[2, 2] = dwdz = FST.backward(1j*K[2]*U_hat[2], duidxj[2, 2])
    return duidxj

def standardConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp, FSBp, FCTp, work,
                       mat, la):
    rhs[:] = 0
    U = u_dealias
    Uc = work[(U, 1, True)]
    Uc2 = work[(U, 2, True)]
    F_tmp = work[(rhs, 0, True)]

    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = mat.CDB.matvec(u_hat[0], F_tmp[0])
    F_tmp[0] = la.TDMASolverD(F_tmp[0])
    dudx = Uc[0] = FSTp.backward(F_tmp[0], Uc[0])

    LUsolve.Mult_CTD_3D_n(params.N[0], u_hat[1], u_hat[2], F_tmp[1], F_tmp[2])
    dvdx = Uc[1] = FCTp.backward(F_tmp[1], Uc[1])
    dwdx = Uc[2] = FCTp.backward(F_tmp[2], Uc[2])

    dudy = Uc2[0] = FSBp.backward(1j*K[1]*u_hat[0], Uc2[0])
    dudz = Uc2[1] = FSBp.backward(1j*K[2]*u_hat[0], Uc2[1])
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

def divergenceConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp, FSBp, FCTp, work,
                         mat, la, add=False):
    """c_i = div(u_i u_j)"""
    if not add:
        rhs.fill(0)
    F_tmp = work[(rhs, 0, True)]
    F_tmp2 = work[(rhs, 1, True)]
    U = u_dealias

    F_tmp[0] = FSTp.forward(U[0]*U[0], F_tmp[0])
    F_tmp[1] = FSTp.forward(U[0]*U[1], F_tmp[1])
    F_tmp[2] = FSTp.forward(U[0]*U[2], F_tmp[2])

    F_tmp2[0] = mat.CDD.matvec(F_tmp[0], F_tmp2[0])
    F_tmp2[1] = mat.CDD.matvec(F_tmp[1], F_tmp2[1])
    F_tmp2[2] = mat.CDD.matvec(F_tmp[2], F_tmp2[2])
    F_tmp2[0] = la.TDMASolverD(F_tmp2[0])
    F_tmp2[1] = la.TDMASolverD(F_tmp2[1])
    F_tmp2[2] = la.TDMASolverD(F_tmp2[2])
    rhs[0] += F_tmp2[0]
    rhs[1] += F_tmp2[1]
    rhs[2] += F_tmp2[2]

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

        def Conv(rhs, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la, u_dealias):
            u_dealias = VFSp.backward(u_hat, u_dealias)
            rhs = standardConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                     FSBp, FCTp, work, mat, la)
            rhs[:] *= -1
            return rhs

    elif convection == "Divergence":

        def Conv(rhs, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la, u_dealias):
            u_dealias = VFSp.backward(u_hat, u_dealias)
            rhs = divergenceConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                       FSBp, FCTp, work, mat, la, False)
            rhs[:] *= -1
            return rhs

    elif convection == "Skew":

        def Conv(rhs, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la, u_dealias):
            u_dealias = VFSp.backward(u_hat, u_dealias)
            rhs = standardConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                     FSBp, FCTp, work, mat, la)
            rhs = divergenceConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                       FSBp, FCTp, work, mat, la, True)
            rhs *= -0.5
            return rhs

    elif convection == "Vortex":
        def Conv(rhs, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la, u_dealias):
            curl_dealias = work[(u_dealias, 1, False)]
            u_dealias = VFSp.backward(u_hat, u_dealias)
            curl_dealias = compute_curl(curl_dealias, u_hat, g_hat, K, FCTp, FSTp, FSBp, work)
            rhs = Cross(rhs, u_dealias, curl_dealias, FSTp, work)
            return rhs

    Conv.convection = convection
    return Conv

@optimizer
def assembleAB(H_hat0, H_hat, H_hat1):
    H_hat0[:] = 1.5*H_hat - 0.5*H_hat1
    return H_hat0

@optimizer
def add_linear(rhs, u, g, work, AB, AC, SBB, ABB, BBB, nu, dt, K2, K4):
    diff_u = work[(g, 0, False)]
    diff_g = work[(g, 1, False)]
    u0 = work[(g, 2, False)]

    # Compute diffusion for g-equation
    diff_g = AB.matvec(g, diff_g)

    # Compute diffusion++ for u-equation
    diff_u = AC.matvec(u, diff_u)
    #diff_u[:] = nu*dt/2.*SBB.matvec(u, u0)
    #diff_u += (1. - nu*dt*K2)*ABB.matvec(u, u0)
    #diff_u -= (K2 - nu*dt/2.*K4)*BBB.matvec(u, u0)

    rhs[0] += diff_u
    rhs[1] += diff_g
    return rhs

#@profile
def ComputeRHS(rhs, u_hat, g_hat, solver,
               H_hat, H_hat1, H_hat0, VFSp, FSTp, FSBp, FCTp, work, Kx, K2, Kx2,
               K4, hv, hg, mat, la, u_dealias, mask, **context):
    """Compute right hand side of Navier Stokes

    args:
        rhs         The right hand side to be returned
        u_hat       The FST of the velocity at current time
        g_hat       The FST of the curl in wall normal direction
        solver      The current solver module

    Remaining args are extracted from context

    """
    # Nonlinear convection term at current u_hat
    H_hat = solver.conv(H_hat, u_hat, g_hat, Kx, VFSp, FSTp, FSBp, FCTp, work, mat, la, u_dealias)

    # Assemble convection with Adams-Bashforth at time = n+1/2
    H_hat0 = solver.assembleAB(H_hat0, H_hat, H_hat1)

    if mask is not None:
        H_hat0.mask_nyquist(mask)

    # Assemble hv, hg and remaining rhs
    w0 = work[(hv, 0, False)]
    w1 = work[(hv, 1, False)]
    hv[:] = -1j*Kx[0]*mat.CBD.matvec(H_hat0[0], w0, axis=2)
    hv -= 1j*Kx[1]*mat.CBD.matvec(H_hat0[1], w0, axis=2)
    hv -= K2*mat.BBD.matvec(H_hat0[2], w0, axis=2)
    hg[:] = 1j*Kx[0]*mat.BDD.matvec(H_hat0[1], w0, axis=2) - 1j*Kx[1]*mat.BDD.matvec(H_hat0[0], w1, axis=2)

    rhs[0] = hv*params.dt
    rhs[1] = hg*2./params.nu

    rhs = solver.add_linear(rhs, u_hat[2], g_hat, work, mat.AB, mat.AC, mat.SBB,
                            mat.ABB, mat.BBB, params.nu, params.dt, K2, K4)

    return rhs

def compute_vw(u_hat, f_hat, g_hat, K_over_K2):
    u_hat[0] = -1j*(K_over_K2[0]*f_hat - K_over_K2[1]*g_hat)
    u_hat[1] = -1j*(K_over_K2[1]*f_hat + K_over_K2[0]*g_hat)
    return u_hat

#@profile
def solve_linear(u_hat, g_hat, rhs,
                 work, la, mat, K_over_K2, H_hat0, U_hat0, Sk, u0_hat, h0_hat,
                 w, w1, **context):
    """Solve final linear algebra systems"""
    f_hat = work[(u_hat[2], 0, True)]
    w0 = work[(u_hat[2], 1, False)]

    u_hat[2] = la.BiharmonicSolverU(u_hat[2], rhs[0])
    g_hat = la.HelmholtzSolverG(g_hat, rhs[1])

    # Compute v_hat and w_hat from u_hat and g_hat
    f_hat -= mat.CDB.matvec(u_hat[2], w0, axis=2)
    f_hat = la.TDMASolverD(f_hat, axis=2)
    u_hat = compute_vw(u_hat, f_hat, g_hat, K_over_K2)

    # Remains to fix wavenumber 0
    if rank == 0:
        h0_hat[0] = H_hat0[0, 0, 0]
        h0_hat[1] = H_hat0[1, 0, 0]
        u0_hat[0] = U_hat0[0, 0, 0]
        u0_hat[1] = U_hat0[1, 0, 0]

        w = mat.BDD0.matvec(2./params.nu*h0_hat[0], w)
        w -= 2./params.nu * Sk[0, 0, 0]
        w1 = mat.ADD0.matvec(u0_hat[0], w1)
        w += w1
        w += 2./params.nu/params.dt * mat.BDD0.matvec(u0_hat[0], w1)
        u0_hat[0] = la.HelmholtzSolverU0(u0_hat[0], w)

        w = mat.BDD0.matvec(2./params.nu*h0_hat[1], w)
        w += mat.ADD0.matvec(u0_hat[1], w1)
        w += mat.BDD0.matvec(2./params.nu/params.dt*u0_hat[1], w1)
        u0_hat[1] = la.HelmholtzSolverU0(u0_hat[1], w)

        u_hat[0, 0, 0] = u0_hat[0]
        u_hat[1, 0, 0] = u0_hat[1]
        u_hat[2, 0, 0] = 0           # This required for continuity

    return u_hat, g_hat

def integrate(u_hat, g_hat, rhs, dt, solver, context):
    """Regular implicit solver for KMM channel solver"""
    rhs[:] = 0
    rhs = solver.ComputeRHS(rhs, u_hat, g_hat, solver, **context)
    u_hat, g_hat = solver.solve_linear(u_hat, g_hat, rhs, **context)
    return (u_hat, g_hat), dt, dt

def getintegrator(rhs, u0, solver, context):
    u_hat, g_hat = u0
    def func():
        return solver.integrate(u_hat, g_hat, rhs, params.dt, solver, context)
    return func
