__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from .spectralinit import *
from ..shen.shentransform import ShenDirichletBasis, ShenNeumannBasis, \
    ShenBiharmonicBasis, SFTc, SlabShen_R2C
from ..shen.Matrices import BBBmat, SBBmat, ABBmat, BBDmat, CBDmat, CDDmat, \
    ADDmat, BDDmat, CDBmat, BiharmonicCoeff, HelmholtzCoeff
from ..shen.la import Helmholtz, TDMA, Biharmonic
from ..shen import SFTc

def get_context():
    """Set up context for solver"""

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad=params.Dquad, threads=params.threads,
                            planner_effort=params.planner_effort["dct"])
    SB = ShenBiharmonicBasis(quad=params.Bquad, threads=params.threads,
                             planner_effort=params.planner_effort["dct"])

    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nb = params.N[0]-4   # Number of velocity modes in Shen biharmonic basis
    u_slice = slice(0, Nu)
    v_slice = slice(0, Nb)

    FST = SlabShen_R2C(params.N, params.L, comm, threads=params.threads,
                       communication=params.communication,
                       planner_effort=params.planner_effort,
                       dealias_cheb=params.dealias_cheb)

    float, complex, mpitype = datatypes("double")

    # Mesh variables
    X = FST.get_local_mesh(ST)
    x0, x1, x2 = FST.get_mesh_dims(ST)
    K = FST.get_scaled_local_wavenumbermesh()
    
    # Remove oddball Nyquist
    #K[1, :, -params.N[1]/2, 0] = 0
    #K[1, :, -params.N[1]/2, -1] = 0
    #K[2, :, 0, -1] = 0
    #K[2, :, -params.N[1]/2, -1] = 0
    
    K2 = K[1]*K[1]+K[2]*K[2]
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)

    # Solution variables
    U  = zeros((3,)+FST.real_shape(), dtype=float)
    U0 = zeros((3,)+FST.real_shape(), dtype=float)
    U_hat  = zeros((3,)+FST.complex_shape(), dtype=complex)
    U_hat0 = zeros((3,)+FST.complex_shape(), dtype=complex)
    g = zeros(FST.complex_shape(), dtype=complex)
    
    # primary variable
    u = (U_hat, g)

    H_hat  = zeros((3,)+FST.complex_shape(), dtype=complex)
    H_hat0 = zeros((3,)+FST.complex_shape(), dtype=complex)
    H_hat1 = zeros((3,)+FST.complex_shape(), dtype=complex)

    dU = zeros((3,)+FST.complex_shape(), dtype=complex)
    hv = zeros(FST.complex_shape(), dtype=complex)
    hg = zeros(FST.complex_shape(), dtype=complex)
    Source = zeros((3,)+FST.real_shape(), dtype=float)
    Sk = zeros((3,)+FST.complex_shape(), dtype=complex)

    work = work_arrays()
    
    nu, dt, N = params.nu, params.dt, params.N
    K4 = K2**2
    kx = K[0, :, 0, 0]
    
    # Collect all linear algebra solvers
    la = config.AttributeDict(dict(
        HelmholtzSolverG = Helmholtz(N[0], np.sqrt(K2[0]+2.0/nu/dt), ST.quad, False),
        BiharmonicSolverU = Biharmonic(N[0], -nu*dt/2., 1.+nu*dt*K2[0],
                                    -(K2[0] + nu*dt/2.*K4[0]), quad=SB.quad,
                                    solver="cython"),
        HelmholtzSolverU0 = Helmholtz(N[0], np.sqrt(2./nu/dt), ST.quad, False),
        TDMASolverD = TDMA(ST.quad, False)
        )
    )

    alfa = K2[0] - 2.0/nu/dt
    # Collect all matrices
    mat = config.AttributeDict(dict(
        CDD = CDDmat(kx),
        AB = HelmholtzCoeff(kx, -1.0, -alfa, ST.quad),
        AC = BiharmonicCoeff(kx, nu*dt/2., (1. - nu*dt*K2[0]), -(K2[0] - nu*dt/2.*K4[0]), quad=SB.quad),
        # Matrices for biharmonic equation
        CBD = CBDmat(kx),
        ABB = ABBmat(kx),
        BBB = BBBmat(kx, SB.quad),
        SBB = SBBmat(kx),
        # Matrices for Helmholtz equation
        ADD = ADDmat(kx),
        BDD = BDDmat(kx, ST.quad),
        BBD = BBDmat(kx, SB.quad),
        CDB = CDBmat(kx)
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
            U0 = get_velocity(c.U0, c.U_hat0, c.FST, c.ST, c.SB)

assert params.precision == "double"

def end_of_tstep(context):
    """Function called at end of time step. 
    
    If returning True, the while-loop in time breaks free. Used by adaptive solvers
    to modify the time stepsize.
    """
    # Rotate solutions
    context.U_hat0[:] = context.U_hat
    context.H_hat1[:] = context.H_hat
    return False

def get_velocity(U, U_hat, FST, ST, SB, **context):
    """Compute velocity from context"""
    U[0] = FST.ifst(U_hat[0], U[0], SB)
    for i in range(1, 3):
        U[i] = FST.ifst(U_hat[i], U[i], ST)
    return U

def set_velocity(U_hat, U, FST, ST, SB, **context):
    """Set transformed velocity from context"""
    U_hat[0] = FST.fst(U[0], U_hat[0], SB)
    for i in range(1, 3):
        U_hat[i] = FST.fst(U[i], U_hat[i], ST)
    return U_hat

def get_curl(curl, U_hat, g, work, FST, SB, ST, K, **context):
    """Compute curl from context"""
    curl = compute_curl(curl, U_hat, g, K, FST, SB, ST, work)
    return curl

def get_convection(H_hat, U_hat, g, K, FST, SB, ST, work, mat, la, **context):
    """Compute convection from context"""
    conv = getConvection(params.convection)
    H_hat = conv(H_hat, U_hat, g, K, FST, SB, ST, work, mat, la)
    return H_hat

#def get_pressure(P_hat, Ni):
    #"""Solve for pressure if Ni is fst of convection"""
    #pass
    ##F_tmp[0] = 0
    ##SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], Ni[0, u_slice], Ni[1, u_slice], Ni[2, u_slice], F_tmp[0, p_slice])    
    ##HelmholtzSolverP = Helmholtz(N[0], sqrt(K2[0]), SN.quad, True)
    ##P_hat = HelmholtzSolverP(P_hat, F_tmp[0])
    ##return P_hat

#@profile
def Cross(c, a, b, S, FST, work):
    Uc = work[(a, 2, False)]
    Uc = cross1(Uc, a, b)
    c[0] = FST.fst(Uc[0], c[0], S, dealias=params.dealias)
    c[1] = FST.fst(Uc[1], c[1], S, dealias=params.dealias)
    c[2] = FST.fst(Uc[2], c[2], S, dealias=params.dealias)
    return c

@optimizer
def mult_K1j(K, a, f):
    f[0] = 1j*K[2]*a
    f[1] = -1j*K[1]*a
    return f

def compute_curl(c, u_hat, g, K, FST, SB, ST, work):
    F_tmp = work[(u_hat, 0, False)]
    F_tmp2 = work[(u_hat, 2, False)]
    Uc = work[(c, 2, False)]
    # Mult_CTD_3D_n is projection to T of d(u_hat)/dx (for components 1 and 2 of u_hat) 
    # Corresponds to CTD.matvec(u_hat[1])/BTT.dd, CTD.matvec(u_hat[2])/BTT.dd
    SFTc.Mult_CTD_3D_n(params.N[0], u_hat[1], u_hat[2], F_tmp[1], F_tmp[2])
    dvdx = Uc[1] = FST.ifct(F_tmp[1], Uc[1], ST, dealias=params.dealias)
    dwdx = Uc[2] = FST.ifct(F_tmp[2], Uc[2], ST, dealias=params.dealias)
    c[0] = FST.ifst(g, c[0], ST, dealias=params.dealias)
    F_tmp2[:2] = mult_K1j(K, u_hat[0], F_tmp2[:2])
    
    c[1] = FST.ifst(F_tmp2[0], c[1], SB, dealias=params.dealias)
    c[1] -= dwdx
    c[2] = FST.ifst(F_tmp2[1], c[2], SB, dealias=params.dealias)
    c[2] += dvdx
    return c

def compute_derivatives(duidxj, u_hat, FST, ST, SB, la, mat, work):
    duidxj[:] = 0
    F_tmp = work[(u_hat, 0)]
    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = mat.CDB.matvec(U_hat[0])
    F_tmp[0] = la.TDMASolverD(F_tmp[0])    
    duidxj[0, 0] = FST.ifst(F_tmp[0], duidxj[0, 0], ST)
    SFTc.Mult_CTD_3D_n(params.N[0], u_hat[1], u_hat[2], F_tmp[1], F_tmp[2])
    duidxj[1, 0] = dvdx = FST.ifct(F_tmp[1], duidxj[1, 0], ST)  # proj to Cheb
    duidxj[2, 0] = dwdx = FST.ifct(F_tmp[2], duidxj[2, 0], ST)  # proj to Cheb
    duidxj[0, 1] = dudy = FST.ifst(1j*K[1]*u_hat[0], duidxj[0, 1], SB) # ShenB
    duidxj[0, 2] = dudz = FST.ifst(1j*K[2]*u_hat[0], duidxj[0, 2], SB)
    duidxj[1, 1] = dvdy = FST.ifst(1j*K[1]*u_hat[1], duidxj[1, 1], ST)
    duidxj[1, 2] = dvdz = FST.ifst(1j*K[2]*u_hat[1], duidxj[1, 2], ST)    
    duidxj[2, 1] = dwdy = FST.ifst(1j*K[1]*u_hat[2], duidxj[2, 1], ST)
    duidxj[2, 2] = dwdz = FST.ifst(1j*K[2]*u_hat[2], duidxj[2, 2], ST)
    return duidxj

def standardConvection(rhs, u_dealias, u_hat, K, FST, SB, ST, work, mat, la):
    rhs[:] = 0
    U = u_dealias
    Uc = work[(U, 1)]
    Uc2 = work[(U, 2)]
    F_tmp = work[(rhs, 0)]
    
    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = mat.CDB.matvec(u_hat[0])
    F_tmp[0] = la.TDMASolverD(F_tmp[0])    
    dudx = Uc[0] = FST.ifst(F_tmp[0], Uc[0], ST, dealias=params.dealias)   
        
    SFTc.Mult_CTD_3D_n(params.N[0], u_hat[1], u_hat[2], F_tmp[1], F_tmp[2])
    dvdx = Uc[1] = FST.ifct(F_tmp[1], Uc[1], ST, dealias=params.dealias)
    dwdx = Uc[2] = FST.ifct(F_tmp[2], Uc[2], ST, dealias=params.dealias)
    
    dudy = Uc2[0] = FST.ifst(1j*K[1]*u_hat[0], Uc2[0], SB, dealias=params.dealias)    
    dudz = Uc2[1] = FST.ifst(1j*K[2]*u_hat[0], Uc2[1], SB, dealias=params.dealias)
    rhs[0] = FST.fst(U[0]*dudx + U[1]*dudy + U[2]*dudz, rhs[0], ST, dealias=params.dealias)
    
    Uc2[:] = 0
    dvdy = Uc2[0] = FST.ifst(1j*K[1]*u_hat[1], Uc2[0], ST, dealias=params.dealias)
    
    dvdz = Uc2[1] = FST.ifst(1j*K[2]*u_hat[1], Uc2[1], ST, dealias=params.dealias)
    rhs[1] = FST.fst(U[0]*dvdx + U[1]*dvdy + U[2]*dvdz, rhs[1], ST, dealias=params.dealias)
    
    Uc2[:] = 0
    dwdy = Uc2[0] = FST.ifst(1j*K[1]*u_hat[2], Uc2[0], ST, dealias=params.dealias)    
    dwdz = Uc2[1] = FST.ifst(1j*K[2]*u_hat[2], Uc2[1], ST, dealias=params.dealias)
    
    rhs[2] = FST.fst(U[0]*dwdx + U[1]*dwdy + U[2]*dwdz, rhs[2], ST, dealias=params.dealias)
    
    return rhs

def divergenceConvection(rhs, u_dealias, u_hat, K, FST, SB, ST, work, mat, la, add=False):
    """c_i = div(u_i u_j)"""
    if not add: rhs.fill(0)
    F_tmp  = work[(rhs, 0)]
    F_tmp2 = work[(rhs, 1)]
    U = u_dealias
    #U_tmp[0] = chebDerivative_3D0(U[0]*U[0], U_tmp[0])
    #U_tmp[1] = chebDerivative_3D0(U[0]*U[1], U_tmp[1])
    #U_tmp[2] = chebDerivative_3D0(U[0]*U[2], U_tmp[2])
    #rhs[0] = fss(U_tmp[0], rhs[0], ST)
    #rhs[1] = fss(U_tmp[1], rhs[1], ST)
    #rhs[2] = fss(U_tmp[2], rhs[2], ST)
    
    F_tmp[0] = FST.fst(U[0]*U[0], F_tmp[0], ST, dealias=params.dealias)
    F_tmp[1] = FST.fst(U[0]*U[1], F_tmp[1], ST, dealias=params.dealias)
    F_tmp[2] = FST.fst(U[0]*U[2], F_tmp[2], ST, dealias=params.dealias)
    
    F_tmp2[0] = mat.CDD.matvec(F_tmp[0])
    F_tmp2[1] = mat.CDD.matvec(F_tmp[1])
    F_tmp2[2] = mat.CDD.matvec(F_tmp[2])
    F_tmp2[0] = la.TDMASolverD(F_tmp2[0])
    F_tmp2[1] = la.TDMASolverD(F_tmp2[1])
    F_tmp2[2] = la.TDMASolverD(F_tmp2[2])
    rhs[0] += F_tmp2[0]
    rhs[1] += F_tmp2[1]
    rhs[2] += F_tmp2[2]
    
    F_tmp2[0] = FST.fst(U[0]*U[1], F_tmp2[0], ST, dealias=params.dealias)
    F_tmp2[1] = FST.fst(U[0]*U[2], F_tmp2[1], ST, dealias=params.dealias)    
    rhs[0] += 1j*K[1]*F_tmp2[0] # duvdy
    rhs[0] += 1j*K[2]*F_tmp2[1] # duwdz
    
    F_tmp[0] = FST.fst(U[1]*U[1], F_tmp[0], ST, dealias=params.dealias)
    F_tmp[1] = FST.fst(U[1]*U[2], F_tmp[1], ST, dealias=params.dealias)
    F_tmp[2] = FST.fst(U[2]*U[2], F_tmp[2], ST, dealias=params.dealias)
    rhs[1] += 1j*K[1]*F_tmp[0]  # dvvdy
    rhs[1] += 1j*K[2]*F_tmp[1]  # dvwdz  
    rhs[2] += 1j*K[1]*F_tmp[1]  # dvwdy
    rhs[2] += 1j*K[2]*F_tmp[2]  # dwwdz
    
    return rhs

def getConvection(convection):
    
    if convection == "Standard":
        
        def Conv(rhs, u_hat, g_hat, K, FST, SB, ST, work, mat, la):
            
            u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            u_dealias[0] = FST.ifst(u_hat[0], u_dealias[0], SB, params.dealias) 
            for i in range(1, 3):
                u_dealias[i] = FST.ifst(u_hat[i], u_dealias[i], ST, params.dealias)

            rhs = standardConvection(rhs, u_dealias, u_hat, K, FST, SB, ST,
                                        work, mat, la)
            rhs[:] *= -1
            return rhs
        
    elif convection == "Divergence":
        
        def Conv(rhs, u_hat, g_hat, K, FST, SB, ST, work, mat, la):
            
            u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            u_dealias[0] = FST.ifst(u_hat[0], u_dealias[0], SB, params.dealias) 
            for i in range(1, 3):
                u_dealias[i] = FST.ifst(u_hat[i], u_dealias[i], ST, params.dealias)

            rhs = divergenceConvection(rhs, u_dealias, u_hat, K, FST, SB, ST,
                                        work, mat, la, False)
            rhs[:] *= -1
            return rhs
        
    elif convection == "Skew":
        
        def Conv(rhs, u_hat, g_hat, K, FST, SB, ST, work, mat, la):
            
            u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            u_dealias[0] = FST.ifst(u_hat[0], u_dealias[0], SB, params.dealias) 
            for i in range(1, 3):
                u_dealias[i] = FST.ifst(u_hat[i], u_dealias[i], ST, params.dealias)

            rhs = standardConvection(rhs, u_dealias, u_hat, K, FST, SB, ST,
                                        work, mat, la)
            rhs = divergenceConvection(rhs, u_dealias, u_hat, K, FST, SB, ST,
                                        work, mat, la, True)            
            rhs *= -0.5
            return rhs

    elif convection == "Vortex":
        #@profile
        def Conv(rhs, u_hat, g_hat, K, FST, SB, ST, work, mat, la):
            
            u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            curl_dealias = work[((3,)+FST.work_shape(params.dealias), float, 1)]
            u_dealias[0] = FST.ifst(u_hat[0], u_dealias[0], SB, params.dealias) 
            for i in range(1, 3):
                u_dealias[i] = FST.ifst(u_hat[i], u_dealias[i], ST, params.dealias)
            
            curl_dealias = compute_curl(curl_dealias, u_hat, g_hat, K, FST, SB, ST, work)
            rhs = Cross(rhs, u_dealias, curl_dealias, ST, FST, work)
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
    
    # Compute diffusion for g-equation
    diff_g = AB.matvec(g, diff_g)

    # Compute diffusion++ for u-equation
    diff_u[:] = nu*dt/2.*SBB.matvec(u)
    diff_u += (1. - nu*dt*K2)*ABB.matvec(u)
    diff_u -= (K2 - nu*dt/2.*K4)*BBB.matvec(u)

    rhs[0] += diff_u
    rhs[1] += diff_g
    return rhs

def ComputeRHS(rhs, u_hat, g_hat, solver,
               H_hat, H_hat1, H_hat0, FST, ST, SB, work, K, K2, K4, hv, hg,
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
    H_hat = solver.conv(H_hat, u_hat, g_hat, K, FST, SB, ST, work, mat, la)

    # Assemble convection with Adams-Bashforth at time = n+1/2
    H_hat0 = solver.assembleAB(H_hat0, H_hat, H_hat1)
    
    # Assemble hv, hg and remaining rhs
    hv[:] = -K2*mat.BBD.matvec(H_hat0[0])
    hv -= 1j*K[1]*mat.CBD.matvec(H_hat0[1])
    hv -= 1j*K[2]*mat.CBD.matvec(H_hat0[2])        
    hg[:] = 1j*K[1]*mat.BDD.matvec(H_hat0[2]) - 1j*K[2]*mat.BDD.matvec(H_hat0[1])
    
    rhs[0] = hv*params.dt
    rhs[1] = hg*2./params.nu
    
    rhs = solver.add_linear(rhs, u_hat[0], g_hat, work, mat.AB, mat.AC, mat.SBB,
                            mat.ABB, mat.BBB, params.nu, params.dt, K2, K4)

    return rhs

@optimizer
def compute_vw(u_hat, f_hat, g_hat, K_over_K2):
    u_hat[1] = -1j*(K_over_K2[1]*f_hat - K_over_K2[2]*g_hat)
    u_hat[2] = -1j*(K_over_K2[2]*f_hat + K_over_K2[1]*g_hat)
    return u_hat

#@profile
def solve_linear(u_hat, g_hat, rhs,
                 work, la, mat, K_over_K2, H_hat0, U_hat0, Sk, **context):
    """"""
    f_hat = work[(u_hat[0], 0)]
    
    u_hat[0] = la.BiharmonicSolverU(u_hat[0], rhs[0])
    g_hat = la.HelmholtzSolverG(g_hat, rhs[1])
    
    # Compute v_hat and w_hat from u_hat and g_hat
    f_hat -= mat.CDB.matvec(u_hat[0])
    f_hat = la.TDMASolverD(f_hat)
    u_hat = compute_vw(u_hat, f_hat, g_hat, K_over_K2)

    # Remains to fix wavenumber 0
    if rank == 0:
        u0_hat = work[((2, params.N[0]), complex, 0)]
        h0_hat = work[((2, params.N[0]), complex, 1)]
        w = work[((params.N[0], ), complex, 0, False)]

        h0_hat[0] = H_hat0[1, :, 0, 0]
        h0_hat[1] = H_hat0[2, :, 0, 0]
        u0_hat[0] = U_hat0[1, :, 0, 0]
        u0_hat[1] = U_hat0[2, :, 0, 0]

        w[:] = 2./params.nu * mat.BDD.matvec(h0_hat[0])
        w -= 2./params.nu * Sk[1, :, 0, 0]        
        w -= mat.ADD.matvec(u0_hat[0])
        w += 2./params.nu/params.dt * mat.BDD.matvec(u0_hat[0])        
        u0_hat[0] = la.HelmholtzSolverU0(u0_hat[0], w)
        
        w[:] = 2./params.nu * mat.BDD.matvec(h0_hat[1])
        w -= mat.ADD.matvec(u0_hat[1])
        w += 2./params.nu/params.dt * mat.BDD.matvec(u0_hat[1])
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
