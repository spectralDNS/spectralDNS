__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from .spectralinit import *
from shenfun.chebyshev.bases import ShenDirichletBasis, ShenNeumannBasis, \
    ShenBiharmonicBasis
from shenfun.chebyshev.matrices import BBBmat, SBBmat, ABBmat, BBDmat, CBDmat, CDDmat, \
    ADDmat, BDDmat, CDBmat
from shenfun.spectralbase import inner_product
from shenfun.la import TDMA
from shenfun import TensorProductSpace, Function, TestFunction, TrialFunction, \
    VectorTensorProductSpace
from shenfun.chebyshev.bases import ShenDirichletBasis, ShenBiharmonicBasis, Basis
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.chebyshev.la import Helmholtz, Biharmonic

#from ..shen.shentransform import SlabShen_R2C
from ..shen.Matrices import BiharmonicCoeff, HelmholtzCoeff
from ..shen import LUsolve

def get_context():
    """Set up context for solver"""

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(params.N[0], quad=params.Dquad)
    SB = ShenBiharmonicBasis(params.N[0], quad=params.Bquad)
    CT = Basis(params.N[0], quad=params.Dquad)
    ST0 = ShenDirichletBasis(params.N[0], quad=params.Dquad, plan=True) # For 1D problem
    K0 = C2CBasis(params.N[1])
    K1 = R2CBasis(params.N[2])

    #threads=params.threads, planner_effort=params.planner_effort["dct"]

    #CT = ST.CT  # Chebyshev transform
    FST = TensorProductSpace(comm, (ST, K0, K1))    # Dirichlet
    FSB = TensorProductSpace(comm, (SB, K0, K1))    # Biharmonic
    FCT = TensorProductSpace(comm, (CT, K0, K1))    # Regular Chebyshev
    VFS = VectorTensorProductSpace([FSB, FST, FST])

    # Padded
    STp = ShenDirichletBasis(params.N[0], quad=params.Dquad)
    SBp = ShenBiharmonicBasis(params.N[0], quad=params.Bquad)
    CTp = Basis(params.N[0], quad=params.Dquad)
    K0p = C2CBasis(params.N[1], padding_factor=1.5)
    K1p = R2CBasis(params.N[2], padding_factor=1.5)
    FSTp = TensorProductSpace(comm, (STp, K0p, K1p))
    FSBp = TensorProductSpace(comm, (SBp, K0p, K1p))
    FCTp = TensorProductSpace(comm, (CTp, K0p, K1p))
    VFSp = VectorTensorProductSpace([FSBp, FSTp, FSTp])

    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nb = params.N[0]-4   # Number of velocity modes in Shen biharmonic basis
    u_slice = slice(0, Nu)
    v_slice = slice(0, Nb)

    #FST = SlabShen_R2C(params.N, params.L, comm, threads=params.threads,
                       #communication=params.communication,
                       #planner_effort=params.planner_effort,
                       #dealias_cheb=params.dealias_cheb)

    float, complex, mpitype = datatypes("double")

    # Mesh variables
    X = FST.local_mesh(True)
    x0, x1, x2 = FST.mesh()
    K = FST.local_wavenumbers(scaled=True)

    K2 = K[1]*K[1]+K[2]*K[2]
    K_over_K2 = Function(VFS)[1:]
    for i in range(2):
        K_over_K2[i] = K[i+1] / np.where(K2==0, 1, K2)

    # Solution variables
    U  = Function(VFS, False)
    U0 = Function(VFS, False)
    U_hat  = Function(VFS)
    U_hat0 = Function(VFS)
    g = Function(FST)

    # primary variable
    u = (U_hat, g)

    H_hat  = Function(VFS)
    H_hat0 = Function(VFS)
    H_hat1 = Function(VFS)

    dU = Function(VFS)
    hv = Function(FST)
    hg = Function(FST)
    Source = Function(VFS, False)
    Sk = Function(VFS)

    work = work_arrays()

    nu, dt, N = params.nu, params.dt, params.N
    K4 = K2**2
    kx = K[0][:, 0, 0]

    alfa = K2[0] - 2.0/nu/dt
    # Collect all matrices
    mat = config.AttributeDict(dict(
        CDD = inner_product((ST, 0), (ST, 1)),
        AB = HelmholtzCoeff(kx, -1.0, -alfa, ST.quad),
        AC = BiharmonicCoeff(kx, nu*dt/2., (1. - nu*dt*K2[0]), -(K2[0] - nu*dt/2.*K4[0]), quad=SB.quad),
        # Matrices for biharmonic equation
        CBD = inner_product((SB, 0), (ST, 1)),
        ABB = inner_product((SB, 0), (SB, 2)),
        BBB = inner_product((SB, 0), (SB, 0)),
        SBB = inner_product((SB, 0), (SB, 4)),
        # Matrices for Helmholtz equation
        ADD = inner_product((ST, 0), (ST, 2)),
        BDD = inner_product((ST, 0), (ST, 0)),
        BBD = inner_product((SB, 0), (ST, 0)),
        CDB = inner_product((ST, 0), (SB, 1)),
        ADD0 = inner_product((ST0, 0), (ST0, 2)),
        BDD0 = inner_product((ST0, 0), (ST0, 0)),
        )
    )

    # Collect all linear algebra solvers
    #la = config.AttributeDict(dict(
        #HelmholtzSolverG = Helmholtz(N[0], np.sqrt(K2[0]+2.0/nu/dt), ST),
        #BiharmonicSolverU = Biharmonic(N[0], -nu*dt/2., 1.+nu*dt*K2[0],
                                    #-(K2[0] + nu*dt/2.*K4[0]), quad=SB.quad,
                                    #solver="cython"),
        #HelmholtzSolverU0 = Helmholtz(N[0], np.sqrt(2./nu/dt), ST),
        #TDMASolverD = TDMA(inner_product((ST, 0), (ST, 0)))
        #)
    #)
    mat.ADD.scale = np.ones((1,1,1))
    mat.ADD.axis = 0
    mat.BDD.scale = (K2[0]+2.0/nu/dt)[np.newaxis,:,:]
    mat.BDD.axis = 0
    mat.SBB.scale = -nu*dt/2.*np.ones((1,1,1))
    mat.ABB.scale = (1.+nu*dt*K2[0])[np.newaxis,:,:]
    mat.BBB.scale = -(K2[0] + nu*dt/2.*K4[0])[np.newaxis,:,:]
    mat.SBB.axis = 0

    la = config.AttributeDict(dict(
        HelmholtzSolverG = Helmholtz(mat.ADD, mat.BDD, np.ones((1,1,1)),
                                     (K2[0]+2.0/nu/dt)[np.newaxis,:,:]),
        BiharmonicSolverU = Biharmonic(mat.SBB, mat.ABB, mat.BBB, -nu*dt/2.*np.ones((1,1,1)),
                                       (1.+nu*dt*K2[0])[np.newaxis,:,:],
                                       (-(K2[0] + nu*dt/2.*K4[0]))[np.newaxis,:,:]),
        HelmholtzSolverU0 = Helmholtz(mat.ADD0, mat.BDD0, np.ones(1), np.array([2./nu/dt])),
        TDMASolverD = TDMA(inner_product((ST, 0), (ST, 0)))
        )
    )

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

def get_curl(curl, U_hat, g, work, FST, SB, ST, K, **context):
    """Compute curl from context"""
    curl = compute_curl(curl, U_hat, g, K, FST, SB, ST, work)
    return curl

def get_convection(H_hat, U_hat, g, K, VFSp, FSTp, FSBp, FCTp,  work, mat, la, **context):
    """Compute convection from context"""
    conv = getConvection(params.convection)
    H_hat = conv(H_hat, U_hat, g, K, VFSp, FSTp, FSBp, FCTp, work, mat, la)
    return H_hat

#def get_pressure(P_hat, Ni):
    #"""Solve for pressure if Ni is fst of convection"""
    #pass
    ##F_tmp[0] = 0
    ##LUsolve.Mult_Div_3D(N[0], K[1, 0], K[2, 0], Ni[0, u_slice], Ni[1, u_slice], Ni[2, u_slice], F_tmp[0, p_slice])
    ##HelmholtzSolverP = Helmholtz(N[0], sqrt(K2[0]), SN.quad, True)
    ##P_hat = HelmholtzSolverP(P_hat, F_tmp[0])
    ##return P_hat

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
#@profile
def compute_curl(c, u_hat, g, K, FCTp, FSTp, FSBp, work):
    F_tmp = work[(u_hat, 0, False)]
    F_tmp2 = work[(u_hat, 2, False)]
    Uc = work[(c, 2, False)]
    # Mult_CTD_3D_n is projection to T of d(u_hat)/dx (for components 1 and 2 of u_hat)
    # Corresponds to CTD.matvec(u_hat[1])/BTT.dd, CTD.matvec(u_hat[2])/BTT.dd
    LUsolve.Mult_CTD_3D_n(params.N[0], u_hat[1], u_hat[2], F_tmp[1], F_tmp[2])
    dvdx = Uc[1] = FCTp.backward(F_tmp[1], Uc[1])
    dwdx = Uc[2] = FCTp.backward(F_tmp[2], Uc[2])
    c[0] = FSTp.backward(g, c[0])
    F_tmp2[:2] = mult_K1j(K, u_hat[0], F_tmp2[:2])

    c[1] = FSBp.backward(F_tmp2[0], c[1])
    c[1] -= dwdx
    c[2] = FSBp.backward(F_tmp2[1], c[2])
    c[2] += dvdx
    return c

#def compute_derivatives(duidxj, u_hat, FST, ST, SB, la, mat, work):
    #duidxj[:] = 0
    #F_tmp = work[(u_hat, 0)]
    ## dudx = 0 from continuity equation. Use Shen Dirichlet basis
    ## Use regular Chebyshev basis for dvdx and dwdx
    #F_tmp[0] = mat.CDB.matvec(U_hat[0])
    #F_tmp[0] = la.TDMASolverD(F_tmp[0])
    #duidxj[0, 0] = FST.backward(F_tmp[0], duidxj[0, 0], ST)
    #LUsolve.Mult_CTD_3D_n(params.N[0], u_hat[1], u_hat[2], F_tmp[1], F_tmp[2])
    #duidxj[1, 0] = dvdx = FST.backward(F_tmp[1], duidxj[1, 0], ST.CT)  # proj to Cheb
    #duidxj[2, 0] = dwdx = FST.backward(F_tmp[2], duidxj[2, 0], ST.CT)  # proj to Cheb
    #duidxj[0, 1] = dudy = FST.backward(1j*K[1]*u_hat[0], duidxj[0, 1], SB) # ShenB
    #duidxj[0, 2] = dudz = FST.backward(1j*K[2]*u_hat[0], duidxj[0, 2], SB)
    #duidxj[1, 1] = dvdy = FST.backward(1j*K[1]*u_hat[1], duidxj[1, 1], ST)
    #duidxj[1, 2] = dvdz = FST.backward(1j*K[2]*u_hat[1], duidxj[1, 2], ST)
    #duidxj[2, 1] = dwdy = FST.backward(1j*K[1]*u_hat[2], duidxj[2, 1], ST)
    #duidxj[2, 2] = dwdz = FST.backward(1j*K[2]*u_hat[2], duidxj[2, 2], ST)
    #return duidxj

#def standardConvection(rhs, u_dealias, u_hat, K, FST, SB, ST, work, mat, la):
    #rhs[:] = 0
    #U = u_dealias
    #Uc = work[(U, 1)]
    #Uc2 = work[(U, 2)]
    #F_tmp = work[(rhs, 0)]

    ## dudx = 0 from continuity equation. Use Shen Dirichlet basis
    ## Use regular Chebyshev basis for dvdx and dwdx
    #F_tmp[0] = mat.CDB.matvec(u_hat[0])
    #F_tmp[0] = la.TDMASolverD(F_tmp[0])
    #dudx = Uc[0] = FST.backward(F_tmp[0], Uc[0], ST, dealias=params.dealias)

    #LUsolve.Mult_CTD_3D_n(params.N[0], u_hat[1], u_hat[2], F_tmp[1], F_tmp[2])
    #dvdx = Uc[1] = FST.backward(F_tmp[1], Uc[1], ST.CT, dealias=params.dealias)
    #dwdx = Uc[2] = FST.backward(F_tmp[2], Uc[2], ST.CT, dealias=params.dealias)

    #dudy = Uc2[0] = FST.backward(1j*K[1]*u_hat[0], Uc2[0], SB, dealias=params.dealias)
    #dudz = Uc2[1] = FST.backward(1j*K[2]*u_hat[0], Uc2[1], SB, dealias=params.dealias)
    #rhs[0] = FST.forward(U[0]*dudx + U[1]*dudy + U[2]*dudz, rhs[0], ST, dealias=params.dealias)

    #Uc2[:] = 0
    #dvdy = Uc2[0] = FST.backward(1j*K[1]*u_hat[1], Uc2[0], ST, dealias=params.dealias)

    #dvdz = Uc2[1] = FST.backward(1j*K[2]*u_hat[1], Uc2[1], ST, dealias=params.dealias)
    #rhs[1] = FST.forward(U[0]*dvdx + U[1]*dvdy + U[2]*dvdz, rhs[1], ST, dealias=params.dealias)

    #Uc2[:] = 0
    #dwdy = Uc2[0] = FST.backward(1j*K[1]*u_hat[2], Uc2[0], ST, dealias=params.dealias)
    #dwdz = Uc2[1] = FST.backward(1j*K[2]*u_hat[2], Uc2[1], ST, dealias=params.dealias)

    #rhs[2] = FST.forward(U[0]*dwdx + U[1]*dwdy + U[2]*dwdz, rhs[2], ST, dealias=params.dealias)

    #return rhs

#def divergenceConvection(rhs, u_dealias, u_hat, K, FST, SB, ST, work, mat, la, add=False):
    #"""c_i = div(u_i u_j)"""
    #if not add: rhs.fill(0)
    #F_tmp  = work[(rhs, 0)]
    #F_tmp2 = work[(rhs, 1)]
    #U = u_dealias

    #F_tmp[0] = FST.forward(U[0]*U[0], F_tmp[0], ST, dealias=params.dealias)
    #F_tmp[1] = FST.forward(U[0]*U[1], F_tmp[1], ST, dealias=params.dealias)
    #F_tmp[2] = FST.forward(U[0]*U[2], F_tmp[2], ST, dealias=params.dealias)

    #F_tmp2[0] = mat.CDD.matvec(F_tmp[0], F_tmp2[0])
    #F_tmp2[1] = mat.CDD.matvec(F_tmp[1], F_tmp2[1])
    #F_tmp2[2] = mat.CDD.matvec(F_tmp[2], F_tmp2[2])
    #F_tmp2[0] = la.TDMASolverD(F_tmp2[0])
    #F_tmp2[1] = la.TDMASolverD(F_tmp2[1])
    #F_tmp2[2] = la.TDMASolverD(F_tmp2[2])
    #rhs[0] += F_tmp2[0]
    #rhs[1] += F_tmp2[1]
    #rhs[2] += F_tmp2[2]

    #F_tmp2[0] = FST.forward(U[0]*U[1], F_tmp2[0], ST, dealias=params.dealias)
    #F_tmp2[1] = FST.forward(U[0]*U[2], F_tmp2[1], ST, dealias=params.dealias)
    #rhs[0] += 1j*K[1]*F_tmp2[0] # duvdy
    #rhs[0] += 1j*K[2]*F_tmp2[1] # duwdz

    #F_tmp[0] = FST.forward(U[1]*U[1], F_tmp[0], ST, dealias=params.dealias)
    #F_tmp[1] = FST.forward(U[1]*U[2], F_tmp[1], ST, dealias=params.dealias)
    #F_tmp[2] = FST.forward(U[2]*U[2], F_tmp[2], ST, dealias=params.dealias)
    #rhs[1] += 1j*K[1]*F_tmp[0]  # dvvdy
    #rhs[1] += 1j*K[2]*F_tmp[1]  # dvwdz
    #rhs[2] += 1j*K[1]*F_tmp[1]  # dvwdy
    #rhs[2] += 1j*K[2]*F_tmp[2]  # dwwdz

    #return rhs

def getConvection(convection):

    #if convection == "Standard":

        #def Conv(rhs, u_hat, g_hat, K, FST, SB, ST, work, mat, la):

            #u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            #u_dealias[0] = FST.backward(u_hat[0], u_dealias[0], SB, params.dealias)
            #for i in range(1, 3):
                #u_dealias[i] = FST.backward(u_hat[i], u_dealias[i], ST, params.dealias)

            #rhs = standardConvection(rhs, u_dealias, u_hat, K, FST, SB, ST,
                                        #work, mat, la)
            #rhs[:] *= -1
            #return rhs

    #elif convection == "Divergence":

        #def Conv(rhs, u_hat, g_hat, K, FST, SB, ST, work, mat, la):

            #u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            #u_dealias[0] = FST.backward(u_hat[0], u_dealias[0], SB, params.dealias)
            #for i in range(1, 3):
                #u_dealias[i] = FST.backward(u_hat[i], u_dealias[i], ST, params.dealias)

            #rhs = divergenceConvection(rhs, u_dealias, u_hat, K, FST, SB, ST,
                                        #work, mat, la, False)
            #rhs[:] *= -1
            #return rhs

    #elif convection == "Skew":

        #def Conv(rhs, u_hat, g_hat, K, FST, SB, ST, work, mat, la):

            #u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            #u_dealias[0] = FST.backward(u_hat[0], u_dealias[0], SB, params.dealias)
            #for i in range(1, 3):
                #u_dealias[i] = FST.backward(u_hat[i], u_dealias[i], ST, params.dealias)

            #rhs = standardConvection(rhs, u_dealias, u_hat, K, FST, SB, ST,
                                        #work, mat, la)
            #rhs = divergenceConvection(rhs, u_dealias, u_hat, K, FST, SB, ST,
                                        #work, mat, la, True)
            #rhs *= -0.5
            #return rhs

    #elif convection == "Vortex":
        ##@profile
    def Conv(rhs, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la):

        u_dealias = work[((3,)+VFSp.backward.output_array.shape, float, 0)]
        curl_dealias = work[((3,)+VFSp.backward.output_array.shape, float, 1)]
        #from IPython import embed; embed()
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
    diff_g = work[(g, 1, False)]
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

def ComputeRHS(rhs, u_hat, g_hat, solver,
               H_hat, H_hat1, H_hat0, VFSp, FSTp, FSBp, FCTp, work, K, K2, K4, hv, hg,
               mat, la, **context):
    """Compute right hand side of Navier Stokes

    args:
        rhs         The right hand side to be returned
        u_hat       The FST of the velocity at current time
        g_hat       The FST of the curl in wall normal direction
        solver      The current solver module

    Remaining args are extracted from context

    """
    # Nonlinear convection term at current u_hat
    H_hat = solver.conv(H_hat, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la)

    # Assemble convection with Adams-Bashforth at time = n+1/2
    H_hat0 = solver.assembleAB(H_hat0, H_hat, H_hat1)

    # Assemble hv, hg and remaining rhs
    w0 = work[(hv, 0, False)]
    w1 = work[(hv, 1, False)]
    hv[:] = -K2*mat.BBD.matvec(H_hat0[0], w0)
    hv -= 1j*K[1]*mat.CBD.matvec(H_hat0[1], w0)
    hv -= 1j*K[2]*mat.CBD.matvec(H_hat0[2], w0)
    hg[:] = 1j*K[1]*mat.BDD.matvec(H_hat0[2], w0) - 1j*K[2]*mat.BDD.matvec(H_hat0[1], w1)

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
    """"""
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

        w[:] = 2./params.nu * mat.BDD.matvec(h0_hat[0], w1)
        w -= 2./params.nu * Sk[1, :, 0, 0]
        w -= mat.ADD.matvec(u0_hat[0], w1)
        w += 2./params.nu/params.dt * mat.BDD.matvec(u0_hat[0], w1)
        u0_hat[0] = la.HelmholtzSolverU0(u0_hat[0], w)

        w[:] = 2./params.nu * mat.BDD.matvec(h0_hat[1], w1)
        w -= mat.ADD.matvec(u0_hat[1], w1)
        w += 2./params.nu/params.dt * mat.BDD.matvec(u0_hat[1], w1)
        u0_hat[1] = la.HelmholtzSolverU0(u0_hat[1], w)

        u_hat[1, :, 0, 0] = u0_hat[0]
        u_hat[2, :, 0, 0] = u0_hat[1]

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
