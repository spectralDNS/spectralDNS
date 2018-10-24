__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unbalanced-tuple-unpacking,unused-variable,function-redefined,unused-argument

from shenfun.spectralbase import inner_product
from shenfun.la import TDMA
from shenfun import TensorProductSpace, Array, TestFunction, TrialFunction, \
    MixedTensorProductSpace, div, grad, Dx, inner, Function, Basis
from shenfun.chebyshev.la import Helmholtz, Biharmonic

from .spectralinit import *
from ..shen.Matrices import BiharmonicCoeff, HelmholtzCoeff
from ..shen import LUsolve

def get_context():
    """Set up context for solver"""

    # Get points and weights for Chebyshev weighted integrals
    assert params.Dquad == params.Bquad
    collapse_fourier = False if params.dealias == '3/2-rule' else True
    ST = Basis(params.N[0], 'C', bc=(0, 0), quad=params.Dquad)
    SB = Basis(params.N[0], 'C', bc='Biharmonic', quad=params.Bquad)
    CT = Basis(params.N[0], 'C', quad=params.Dquad)
    ST0 = Basis(params.N[0], 'C', bc=(0, 0), quad=params.Dquad) # For 1D problem
    K0 = Basis(params.N[1], 'F', domain=(0, params.L[1]), dtype='D')
    K1 = Basis(params.N[2], 'F', domain=(0, params.L[2]), dtype='d')

    kw0 = {'threads': params.threads,
           'planner_effort': params.planner_effort["dct"],
           'slab': (params.decomposition == 'slab'),
           'collapse_fourier': collapse_fourier}
    FST = TensorProductSpace(comm, (ST, K0, K1), **kw0)    # Dirichlet
    FSB = TensorProductSpace(comm, (SB, K0, K1), **kw0)    # Biharmonic
    FCT = TensorProductSpace(comm, (CT, K0, K1), **kw0)    # Regular Chebyshev
    VFS = MixedTensorProductSpace([FSB, FST, FST])
    VFST = MixedTensorProductSpace([FST, FST, FST])
    VUG = MixedTensorProductSpace([FSB, FST])

    # Padded
    kw = {'padding_factor': 1.5 if params.dealias == '3/2-rule' else 1,
          'dealias_direct': params.dealias == '2/3-rule'}
    if params.dealias == '3/2-rule':
        # Requires new bases due to planning and transforms on different size arrays
        STp = Basis(params.N[0], 'C', bc=(0, 0), quad=params.Dquad)
        SBp = Basis(params.N[0], 'C', bc='Biharmonic', quad=params.Bquad)
        CTp = Basis(params.N[0], 'C', quad=params.Dquad)
    else:
        STp, SBp, CTp = ST, SB, CT
    K0p = Basis(params.N[1], 'F', dtype='D', domain=(0, params.L[1]), **kw)
    K1p = Basis(params.N[2], 'F', dtype='d', domain=(0, params.L[2]), **kw)
    FSTp = TensorProductSpace(comm, (STp, K0p, K1p), **kw0)
    FSBp = TensorProductSpace(comm, (SBp, K0p, K1p), **kw0)
    FCTp = TensorProductSpace(comm, (CTp, K0p, K1p), **kw0)
    VFSp = MixedTensorProductSpace([FSBp, FSTp, FSTp])

    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nb = params.N[0]-4   # Number of velocity modes in Shen biharmonic basis
    u_slice = slice(0, Nu)
    v_slice = slice(0, Nb)

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

    K2 = K[1]*K[1]+K[2]*K[2]
    K4 = K2**2

    # Set Nyquist frequency to zero on K that is used for odd derivatives in nonlinear terms
    Kx = FST.local_wavenumbers(scaled=True, eliminate_highest_freq=True)
    K_over_K2 = np.zeros((2,)+g.shape)
    for i in range(2):
        K_over_K2[i] = K[i+1] / np.where(K2 == 0, 1, K2)

    work = work_arrays()

    nu, dt, N = params.nu, params.dt, params.N

    alfa = K2[0] - 2.0/nu/dt
    # Collect all matrices
    mat = config.AttributeDict(
        dict(CDD=inner_product((ST, 0), (ST, 1)),
             AB=HelmholtzCoeff(N[0], 1.0, -(K2 - 2.0/nu/dt), ST.quad),
             AC=BiharmonicCoeff(N[0], nu*dt/2., (1. - nu*dt*K2), -(K2 - nu*dt/2.*K4), quad=SB.quad),
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
             BDD0=inner_product((ST0, 0), (ST0, 0)),))

    mat.ADD.axis = 0
    mat.BDD.axis = 0
    mat.SBB.axis = 0

    la = config.AttributeDict(
        dict(HelmholtzSolverG=Helmholtz(mat.ADD, mat.BDD, -np.ones((1, 1, 1)),
                                        (K2+2.0/nu/dt)),
             BiharmonicSolverU=Biharmonic(mat.SBB, mat.ABB, mat.BBB, -nu*dt/2.*np.ones((1, 1, 1)),
                                          (1.+nu*dt*K2),
                                          (-(K2 + nu*dt/2.*K4))),
             HelmholtzSolverU0=Helmholtz(mat.ADD0, mat.BDD0, np.array([-1.]), np.array([2./nu/dt])),
             TDMASolverD=TDMA(inner_product((ST, 0), (ST, 0)))))

    hdf5file = KMMWriter({"U":U[0], "V":U[1], "W":U[2]},
                         chkpoint={'current':{'U':U}, 'previous':{'U':U0}},
                         filename=params.solver+".h5",
                         mesh={"x": x0, "y": x1, "z": x2})

    return config.AttributeDict(locals())

class KMMWriter(HDF5Writer):
    def update_components(self, **context):
        """Transform to real data when storing the solution"""
        U = get_velocity(**context)    # updates U from U_hat
        if params.tstep % params.checkpoint == 0:
            # update U0 from U0_hat
            c = config.AttributeDict(context)
            U0 = get_velocity(c.U0, c.U_hat0, c.VFS)

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

def get_curl(curl, U_hat, g, work, FST, SB, ST, Kx, **context):
    """Compute curl from context"""
    curl = compute_curl(curl, U_hat, g, Kx, FST, SB, ST, work)
    return curl

def get_convection(H_hat, U_hat, g, Kx, VFSp, FSTp, FSBp, FCTp, work, mat, la, **context):
    """Compute convection from context"""
    conv_ = getConvection(params.convection)
    H_hat = conv_(H_hat, U_hat, g, Kx, VFSp, FSTp, FSBp, FCTp, work, mat, la)
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
    Uc_hat = mat.CDB.matvec(U_hat[0], Uc_hat)
    Uc_hat = la.TDMASolverD(Uc_hat)

    dudx = Uc[0] = FST.backward(Uc_hat, Uc[0])
    dvdy_h = 1j*K[1]*U_hat[1]
    dvdy = Uc[1] = FST.backward(dvdy_h, Uc[1])
    dwdz_h = 1j*K[2]*U_hat[2]
    dwdz = Uc[2] = FST.backward(dwdz_h, Uc[2])
    return dudx+dvdy+dwdz

#@profile
def Cross(c, a, b, FSTp, work):
    Uc = work[(a, 2, False)]
    Uc = cross1(Uc, a, b)
    c[0] = FSTp.forward(Uc[0], c[0])
    c[1] = FSTp.forward(Uc[1], c[1])
    c[2] = FSTp.forward(Uc[2], c[2])
    return c

@optimizer
def mult_K1j(K, a, f):
    f[0] = 1j*K[2]*a
    f[1] = -1j*K[1]*a
    return f

def compute_curl(c, u_hat, g, K, FCTp, FSTp, FSBp, work):
    F_tmp = work[(u_hat, 0, False)]
    F_tmp2 = work[(u_hat, 2, False)]
    Uc = work[(c, 2, False)]
    # Mult_CTD_3D_n is projection to T of d(u_hat)/dx (for components 1 and 2 of u_hat)
    # Corresponds to CTD.matvec(u_hat[1])/BTT.dd, CTD.matvec(u_hat[2])/BTT.dd
    #from IPython import embed; embed()
    LUsolve.Mult_CTD_3D_n(params.N[0], u_hat[1], u_hat[2], F_tmp[1], F_tmp[2], 0)
    dvdx = Uc[1] = FCTp.backward(F_tmp[1], Uc[1])
    dwdx = Uc[2] = FCTp.backward(F_tmp[2], Uc[2])
    c[0] = FSTp.backward(g, c[0])
    F_tmp2[:2] = mult_K1j(K, u_hat[0], F_tmp2[:2])

    c[1] = FSBp.backward(F_tmp2[0], c[1])
    c[1] -= dwdx
    c[2] = FSBp.backward(F_tmp2[1], c[2])
    c[2] += dvdx
    return c

def compute_derivatives(U, U_hat, FST, FCT, FSB, K, la, mat, work, **context):
    duidxj = np.zeros((3, 3)+U.shape[1:])
    F_tmp = work[(U_hat, 0)]
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
    Uc = work[(U, 1)]
    Uc2 = work[(U, 2)]
    F_tmp = work[(rhs, 0)]

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
    F_tmp = work[(rhs, 0)]
    F_tmp2 = work[(rhs, 1)]
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

        def Conv(rhs, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la):
            u_dealias = work[((3,)+VFSp.backward.output_array.shape, float, 0)]
            u_dealias = VFSp.backward(u_hat, u_dealias)
            rhs = standardConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                     FSBp, FCTp, work, mat, la)
            rhs[:] *= -1
            return rhs

    elif convection == "Divergence":

        def Conv(rhs, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la):
            u_dealias = work[((3,)+VFSp.backward.output_array.shape, float, 0)]
            u_dealias = VFSp.backward(u_hat, u_dealias)
            rhs = divergenceConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                       FSBp, FCTp, work, mat, la, False)
            rhs[:] *= -1
            return rhs

    elif convection == "Skew":

        def Conv(rhs, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la):
            u_dealias = work[((3,)+VFSp.backward.output_array.shape, float, 0)]
            u_dealias = VFSp.backward(u_hat, u_dealias)
            rhs = standardConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                     FSBp, FCTp, work, mat, la)
            rhs = divergenceConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                       FSBp, FCTp, work, mat, la, True)
            rhs *= -0.5
            return rhs

    elif convection == "Vortex":
        def Conv(rhs, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la):
            u_dealias = work[((3,)+VFSp.backward.output_array.shape, float, 0)]
            curl_dealias = work[((3,)+VFSp.backward.output_array.shape, float, 1)]
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
    diff_u = work[(g, 0)]
    diff_g = work[(g, 1)]
    u0 = work[(g, 2, False)]

    # Compute diffusion for g-equation
    diff_g = AB.matvec(g, diff_g)

    # Compute diffusion++ for u-equation
    diff_u[:] = nu*dt/2.*SBB.matvec(u, u0)
    diff_u += (1. - nu*dt*K2)*ABB.matvec(u, u0)
    diff_u -= (K2 - nu*dt/2.*K4)*BBB.matvec(u, u0)

    rhs[0] += diff_u
    rhs[1] += diff_g
    return rhs

#@profile
def ComputeRHS(rhs, u_hat, g_hat, solver,
               H_hat, H_hat1, H_hat0, VFSp, FSTp, FSBp, FCTp, work, Kx, K2,
               K4, hv, hg, mat, la, **context):
    """Compute right hand side of Navier Stokes

    Parameters
    ----------
        rhs : array
            The right hand side to be returned
        u_hat : array
            The FST of the velocity at current time
        g_hat : array
            The FST of the curl in wall normal direction
        solver : module
            The current solver module

    Remaining args are extracted from context

    """
    # Nonlinear convection term at current u_hat
    H_hat = solver.conv(H_hat, u_hat, g_hat, Kx, VFSp, FSTp, FSBp, FCTp, work, mat, la)

    # Assemble convection with Adams-Bashforth at time = n+1/2
    H_hat0 = solver.assembleAB(H_hat0, H_hat, H_hat1)

    # Assemble hv, hg and remaining rhs
    w0 = work[(hv, 0, False)]
    w1 = work[(hv, 1, False)]
    hv[:] = -K2*mat.BBD.matvec(H_hat0[0], w0)
    hv -= 1j*Kx[1]*mat.CBD.matvec(H_hat0[1], w0)
    hv -= 1j*Kx[2]*mat.CBD.matvec(H_hat0[2], w0)
    hg[:] = 1j*Kx[1]*mat.BDD.matvec(H_hat0[2], w0) - 1j*Kx[2]*mat.BDD.matvec(H_hat0[1], w1)

    rhs[0] = hv*params.dt
    rhs[1] = hg*2./params.nu

    rhs = solver.add_linear(rhs, u_hat[0], g_hat, work, mat.AB, mat.AC, mat.SBB,
                            mat.ABB, mat.BBB, params.nu, params.dt, K2, K4)

    return rhs

@optimizer
def compute_vw(u_hat, f_hat, g_hat, K_over_K2):
    u_hat[1] = -1j*(K_over_K2[0]*f_hat - K_over_K2[1]*g_hat)
    u_hat[2] = -1j*(K_over_K2[1]*f_hat + K_over_K2[0]*g_hat)
    return u_hat

#@profile
def solve_linear(u_hat, g_hat, rhs,
                 work, la, mat, K_over_K2, H_hat0, U_hat0, Sk, **context):
    """Solve final linear algebra systems"""
    f_hat = work[(u_hat[0], 0)]
    w0 = work[(u_hat[0], 1, False)]

    u_hat[0] = la.BiharmonicSolverU(u_hat[0], rhs[0])
    g_hat = la.HelmholtzSolverG(g_hat, rhs[1])

    # Compute v_hat and w_hat from u_hat and g_hat
    f_hat -= mat.CDB.matvec(u_hat[0], w0)
    f_hat = la.TDMASolverD(f_hat)
    u_hat = compute_vw(u_hat, f_hat, g_hat, K_over_K2)

    # Remains to fix wavenumber 0
    if rank == 0:
        u0_hat = work[((2, params.N[0]), complex, 0)]
        h0_hat = work[((2, params.N[0]), complex, 1)]
        w = work[((params.N[0], ), complex, 0, False)]
        w1 = work[((params.N[0], ), complex, 1, False)]

        h0_hat[0] = H_hat0[1, :, 0, 0]
        h0_hat[1] = H_hat0[2, :, 0, 0]
        u0_hat[0] = U_hat0[1, :, 0, 0]
        u0_hat[1] = U_hat0[2, :, 0, 0]

        w = mat.BDD0.matvec(2./params.nu*h0_hat[0], w)
        w -= 2./params.nu * Sk[1, :, 0, 0]
        w1 = mat.ADD0.matvec(u0_hat[0], w1)
        w += w1
        w += 2./params.nu/params.dt * mat.BDD0.matvec(u0_hat[0], w1)
        u0_hat[0] = la.HelmholtzSolverU0(u0_hat[0], w)

        w = mat.BDD0.matvec(2./params.nu*h0_hat[1], w)
        w += mat.ADD0.matvec(u0_hat[1], w1)
        w += mat.BDD0.matvec(2./params.nu/params.dt*u0_hat[1], w1)
        u0_hat[1] = la.HelmholtzSolverU0(u0_hat[1], w)

        u_hat[1, :, 0, 0] = u0_hat[0]
        u_hat[2, :, 0, 0] = u0_hat[1]
        u_hat[0, :, 0, 0] = 0           # This required for continuity

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
