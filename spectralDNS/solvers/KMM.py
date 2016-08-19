__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from .spectralinit import *
from ..mesh.channel import SlabShen_R2C
from ..shen.shentransform import ShenDirichletBasis, ShenNeumannBasis, \
    ShenBiharmonicBasis, SFTc
from ..shen.Matrices import BBBmat, SBBmat, ABBmat, BBDmat, CBDmat, CDDmat, \
    ADDmat, BDDmat, CDBmat, BiharmonicCoeff, HelmholtzCoeff
from ..shen.la import Helmholtz, TDMA, Biharmonic
from ..shen import SFTc

def setup():
    """Set up context for solver
    
    All data structures and variables defined here will be added to the global
    namespace of the current solver.
    """

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad=params.Dquad, threads=params.threads,
                            planner_effort=params.planner_effort["dct"])
    SB = ShenBiharmonicBasis(quad=params.Bquad, threads=params.threads,
                             planner_effort=params.planner_effort["dct"])

    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nb = params.N[0]-4   # Number of velocity modes in Shen biharmonic basis
    u_slice = slice(0, Nu)
    v_slice = slice(0, Nb)

    FST = SlabShen_R2C(params.N, params.L, MPI, threads=params.threads,
                       communication=params.communication,
                       planner_effort=params.planner_effort,
                       dealias_cheb=params.dealias_cheb)

    float, complex, mpitype = datatypes("double")

    X = FST.get_local_mesh(ST)
    x0, x1, x2 = FST.get_mesh_dims(ST)

    U  = zeros((3,)+FST.real_shape(), dtype=float)
    U0 = zeros((3,)+FST.real_shape(), dtype=float)
    U_hat  = zeros((3,)+FST.complex_shape(), dtype=complex)
    U_hat0 = zeros((3,)+FST.complex_shape(), dtype=complex)
    
    # primary variables
    u = U_hat[0]
    g = zeros(FST.complex_shape(), dtype=complex)

    H_hat  = zeros((3,)+FST.complex_shape(), dtype=complex)
    H_hat0 = zeros((3,)+FST.complex_shape(), dtype=complex)
    H_hat1 = zeros((3,)+FST.complex_shape(), dtype=complex)

    dU = zeros((3,)+FST.complex_shape(), dtype=complex)
    hv = zeros(FST.complex_shape(), dtype=complex)
    hg = zeros(FST.complex_shape(), dtype=complex)
    diff0 = zeros((3,)+FST.complex_shape(), dtype=complex)
    Source = zeros((3,)+FST.real_shape(), dtype=float)
    Sk = zeros((3,)+FST.complex_shape(), dtype=complex)

    K = FST.get_scaled_local_wavenumbermesh()
    K2 = K[1]*K[1]+K[2]*K[2]
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)

    work = work_arrays()
    
    nu, dt, N = params.nu, params.dt, params.N
    K4 = K2**2
    kx = K[0, :, 0, 0]
    HelmholtzSolverG = Helmholtz(N[0], np.sqrt(K2[0]+2.0/nu/dt), ST.quad, False)
    BiharmonicSolverU = Biharmonic(N[0], -nu*dt/2., 1.+nu*dt*K2[0],
                                   -(K2[0] + nu*dt/2.*K4[0]), quad=SB.quad,
                                   solver="cython")
    HelmholtzSolverU0 = Helmholtz(N[0], np.sqrt(2./nu/dt), ST.quad, False)

    u0_hat = zeros((3, N[0]), dtype=complex)
    h0_hat = zeros((3, N[0]), dtype=complex)

    TDMASolverD = TDMA(ST.quad, False)

    alfa = K2[0] - 2.0/nu/dt
    CDD = CDDmat(kx)

    AB = HelmholtzCoeff(kx, -1.0, -alfa, ST.quad)
    AC = BiharmonicCoeff(kx, nu*dt/2., (1. - nu*dt*K2[0]), -(K2[0] - nu*dt/2.*K4[0]), quad=SB.quad)

    # Matrices for biharmonic equation
    CBD = CBDmat(kx)
    ABB = ABBmat(kx)
    BBB = BBBmat(kx, SB.quad)
    SBB = SBBmat(kx)

    # Matrices for Helmholtz equation
    ADD = ADDmat(kx)
    BDD = BDDmat(kx, ST.quad)

    BBD = BBDmat(kx, SB.quad)
    CDB = CDBmat(kx)
    
    hdf5file = KMMWriter({"U":U[0], "V":U[1], "W":U[2]},
                         chkpoint={'current':{'U':U}, 'previous':{'U':U0}},
                         filename=params.solver+".h5",
                         mesh={"x": x0, "y": x1, "z": x2})
    
    return config.ParamsBase(locals())

class KMMWriter(HDF5Writer):
    def update_components(self, **context):
        """Transform to real data when storing the solution"""
        U = get_velocity(**context)    # updates U from U_hat
        if params.tstep % params.checkpoint == 0:
            # update U0 from U0_hat
            c = config.ParamsBase(context)
            U0 = get_velocity(c.U0, c.U_hat0, c.FST, c.ST, c.SB)

assert params.precision == "double"

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

def forward_velocity(U_hat, U, FST):
    U_hat[0] = FST.fst(U[0], U_hat[0], SB)
    for i in range(1, 3):
        U_hat[i] = FST.fst(U[i], U_hat[i], ST)
    return U_hat

def solvePressure(P_hat, Ni):
    """Solve for pressure if Ni is fst of convection"""
    pass
    #F_tmp[0] = 0
    #SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], Ni[0, u_slice], Ni[1, u_slice], Ni[2, u_slice], F_tmp[0, p_slice])    
    #HelmholtzSolverP = Helmholtz(N[0], sqrt(K2[0]), SN.quad, True)
    #P_hat = HelmholtzSolverP(P_hat, F_tmp[0])
    #return P_hat

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
    
#@profile
def compute_curl(c, a_hat, g, K, FST, SB, ST, work):
    F_tmp = work[(a_hat, 0, False)]
    F_tmp2 = work[(a_hat, 2, False)]
    Uc = work[(c, 2, False)]
    # Mult_CTD_3D_n is projection to T of d(a_hat)/dx (for components 1 and 2 of a_hat) 
    # Corresponds to CTD.matvec(a_hat[1])/BTT.dd, CTD.matvec(a_hat[2])/BTT.dd
    SFTc.Mult_CTD_3D_n(params.N[0], a_hat[1], a_hat[2], F_tmp[1], F_tmp[2])
    dvdx = Uc[1] = FST.ifct(F_tmp[1], Uc[1], ST, dealias=params.dealias)
    dwdx = Uc[2] = FST.ifct(F_tmp[2], Uc[2], ST, dealias=params.dealias)
    c[0] = FST.ifst(g, c[0], ST, dealias=params.dealias)
    F_tmp2[:2] = mult_K1j(K, a_hat[0], F_tmp2[:2])
    
    c[1] = FST.ifst(F_tmp2[0], c[1], SB, dealias=params.dealias)
    c[1] -= dwdx
    c[2] = FST.ifst(F_tmp2[1], c[2], SB, dealias=params.dealias)
    c[2] += dvdx
    return c

def compute_derivatives(duidxj, u_hat, FST, ST, SB, CDB, TDMASolverD, work):
    duidxj[:] = 0
    F_tmp = work[(u_hat, 0)]
    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = CDB.matvec(U_hat[0])
    F_tmp[0] = TDMASolverD(F_tmp[0])    
    duidxj[0, 0] = FST.ifst(F_tmp[0], duidxj[0, 0], ST)
    SFTc.Mult_CTD_3D_n(N[0], u_hat[1], u_hat[2], F_tmp[1], F_tmp[2])
    duidxj[1, 0] = dvdx = FST.ifct(F_tmp[1], duidxj[1, 0], ST)  # proj to Cheb
    duidxj[2, 0] = dwdx = FST.ifct(F_tmp[2], duidxj[2, 0], ST)  # proj to Cheb
    duidxj[0, 1] = dudy = FST.ifst(1j*K[1]*u_hat[0], duidxj[0, 1], SB) # ShenB
    duidxj[0, 2] = dudz = FST.ifst(1j*K[2]*u_hat[0], duidxj[0, 2], SB)
    duidxj[1, 1] = dvdy = FST.ifst(1j*K[1]*u_hat[1], duidxj[1, 1], ST)
    duidxj[1, 2] = dvdz = FST.ifst(1j*K[2]*u_hat[1], duidxj[1, 2], ST)    
    duidxj[2, 1] = dwdy = FST.ifst(1j*K[1]*u_hat[2], duidxj[2, 1], ST)
    duidxj[2, 2] = dwdz = FST.ifst(1j*K[2]*u_hat[2], duidxj[2, 2], ST)
    return duidxj

@optimizer
def add_diffusion_u(b, u, AC, SBB, ABB, BBB, nu, dt, K2, K4):
    b[:] = nu*dt/2.*SBB.matvec(u)
    b += (1. - nu*dt*K2)*ABB.matvec(u)
    b -= (K2 - nu*dt/2.*K4)*BBB.matvec(u)    
    return b

@optimizer
def assembleAB(H_hat0, H_hat, H_hat1):
    H_hat0[:] = 1.5*H_hat - 0.5*H_hat1
    return H_hat0
    
#@profile
class ComputeRHS(RhsBase):
    
    @staticmethod
    def _getConvection(convection):
        
            #@profile
        def _standardConvection(c, u_dealias, U_hat):
            c[:] = 0
            U = u_dealias
            Uc = work[(U, 1)]
            Uc2 = work[(U, 2)]
            F_tmp = work[(c, 0)]
            
            # dudx = 0 from continuity equation. Use Shen Dirichlet basis
            # Use regular Chebyshev basis for dvdx and dwdx
            F_tmp[0] = CDB.matvec(U_hat[0])
            F_tmp[0] = TDMASolverD(F_tmp[0])    
            dudx = Uc[0] = FST.ifst(F_tmp[0], Uc[0], ST, dealias=params.dealias)   
                
            SFTc.Mult_CTD_3D_n(N[0], U_hat[1], U_hat[2], F_tmp[1], F_tmp[2])
            dvdx = Uc[1] = FST.ifct(F_tmp[1], Uc[1], ST, dealias=params.dealias)
            dwdx = Uc[2] = FST.ifct(F_tmp[2], Uc[2], ST, dealias=params.dealias)
            
            dudy = Uc2[0] = FST.ifst(1j*K[1]*U_hat[0], Uc2[0], SB, dealias=params.dealias)    
            dudz = Uc2[1] = FST.ifst(1j*K[2]*U_hat[0], Uc2[1], SB, dealias=params.dealias)
            c[0] = FST.fst(U[0]*dudx + U[1]*dudy + U[2]*dudz, c[0], ST, dealias=params.dealias)
            
            Uc2[:] = 0
            dvdy = Uc2[0] = FST.ifst(1j*K[1]*U_hat[1], Uc2[0], ST, dealias=params.dealias)
            
            dvdz = Uc2[1] = FST.ifst(1j*K[2]*U_hat[1], Uc2[1], ST, dealias=params.dealias)
            c[1] = FST.fst(U[0]*dvdx + U[1]*dvdy + U[2]*dvdz, c[1], ST, dealias=params.dealias)
            
            Uc2[:] = 0
            dwdy = Uc2[0] = FST.ifst(1j*K[1]*U_hat[2], Uc2[0], ST, dealias=params.dealias)    
            dwdz = Uc2[1] = FST.ifst(1j*K[2]*U_hat[2], Uc2[1], ST, dealias=params.dealias)
            
            c[2] = FST.fst(U[0]*dwdx + U[1]*dwdy + U[2]*dwdz, c[2], ST, dealias=params.dealias)
            
            return c

        def _divergenceConvection(c, U, U_hat, add=False):
            """c_i = div(u_i u_j)"""
            if not add: c.fill(0)
            F_tmp  = work[(c, 0)]
            F_tmp2 = work[(c, 1)]
            #U_tmp[0] = chebDerivative_3D0(U[0]*U[0], U_tmp[0])
            #U_tmp[1] = chebDerivative_3D0(U[0]*U[1], U_tmp[1])
            #U_tmp[2] = chebDerivative_3D0(U[0]*U[2], U_tmp[2])
            #c[0] = fss(U_tmp[0], c[0], ST)
            #c[1] = fss(U_tmp[1], c[1], ST)
            #c[2] = fss(U_tmp[2], c[2], ST)
            
            F_tmp[0] = FST.fst(U[0]*U[0], F_tmp[0], ST, dealias=params.dealias)
            F_tmp[1] = FST.fst(U[0]*U[1], F_tmp[1], ST, dealias=params.dealias)
            F_tmp[2] = FST.fst(U[0]*U[2], F_tmp[2], ST, dealias=params.dealias)
            
            F_tmp2[0] = CDD.matvec(F_tmp[0])
            F_tmp2[1] = CDD.matvec(F_tmp[1])
            F_tmp2[2] = CDD.matvec(F_tmp[2])
            F_tmp2[0] = TDMASolverD(F_tmp2[0])
            F_tmp2[1] = TDMASolverD(F_tmp2[1])
            F_tmp2[2] = TDMASolverD(F_tmp2[2])
            c[0] += F_tmp2[0]
            c[1] += F_tmp2[1]
            c[2] += F_tmp2[2]
            
            F_tmp2[0] = FST.fst(U[0]*U[1], F_tmp2[0], ST, dealias=params.dealias)
            F_tmp2[1] = FST.fst(U[0]*U[2], F_tmp2[1], ST, dealias=params.dealias)    
            c[0] += 1j*K[1]*F_tmp2[0] # duvdy
            c[0] += 1j*K[2]*F_tmp2[1] # duwdz
            
            F_tmp[0] = FST.fst(U[1]*U[1], F_tmp[0], ST, dealias=params.dealias)
            F_tmp[1] = FST.fst(U[1]*U[2], F_tmp[1], ST, dealias=params.dealias)
            F_tmp[2] = FST.fst(U[2]*U[2], F_tmp[2], ST, dealias=params.dealias)
            c[1] += 1j*K[1]*F_tmp[0]  # dvvdy
            c[1] += 1j*K[2]*F_tmp[1]  # dvwdz  
            c[2] += 1j*K[1]*F_tmp[1]  # dvwdy
            c[2] += 1j*K[2]*F_tmp[2]  # dwwdz
            
            return c

        if convection == "Standard":
            
            def Conv(H_hat, U_hat):
                
                u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
                u_dealias[0] = FST.ifst(U_hat[0], u_dealias[0], SB, params.dealias) 
                for i in range(1, 3):
                    u_dealias[i] = FST.ifst(U_hat[i], u_dealias[i], ST, params.dealias)

                H_hat = standardConvection(H_hat, u_dealias, U_hat)
                H_hat[:] *= -1
                return H_hat
            
        elif convection == "Divergence":
            
            def Conv(H_hat, U_hat):
                
                u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
                u_dealias[0] = FST.ifst(U_hat[0], u_dealias[0], SB, params.dealias) 
                for i in range(1, 3):
                    u_dealias[i] = FST.ifst(U_hat[i], u_dealias[i], ST, params.dealias)

                H_hat = divergenceConvection(H_hat, u_dealias, U_hat, False)
                H_hat[:] *= -1
                return H_hat
            
        elif convection == "Skew":
            
            def Conv(H_hat, U_hat):
                
                u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
                u_dealias[0] = FST.ifst(U_hat[0], u_dealias[0], SB, params.dealias) 
                for i in range(1, 3):
                    u_dealias[i] = FST.ifst(U_hat[i], u_dealias[i], ST, params.dealias)

                H_hat = standardConvection(H_hat, u_dealias, U_hat)
                H_hat = divergenceConvection(H_hat, u_dealias, U_hat, True)            
                H_hat *= -0.5
                return H_hat

        elif convection == "Vortex":
            
            def Conv(rhs, u_hat, g, K, FST, SB, ST, work):
                
                u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
                curl_dealias = work[((3,)+FST.work_shape(params.dealias), float, 1)]
                u_dealias[0] = FST.ifst(u_hat[0], u_dealias[0], SB, params.dealias) 
                for i in range(1, 3):
                    u_dealias[i] = FST.ifst(u_hat[i], u_dealias[i], ST, params.dealias)
                
                curl_dealias = compute_curl(curl_dealias, u_hat, g, K, FST, SB, ST, work)
                rhs = Cross(rhs, u_dealias, curl_dealias, ST, FST, work)
                return rhs
            
        return Conv      

    @staticmethod
    def add_linear(rhs, u, g, diff0, AB, AC, SBB, ABB, BBB, nu, dt, K2, K4):
        diff0[:] = 0
    
        # Compute diffusion for g-equation
        diff0[1] = AB.matvec(g, diff0[1])
        
        # Compute diffusion++ for u-equation
        diff0[0] = add_diffusion_u(diff0[0], u, AC, SBB, ABB, BBB, nu, dt, K2, K4)
        
        rhs[:2] += diff0[:2]
        return rhs

    def __call__(self, rhs, u_hat, g, H_hat, H_hat1, H_hat0, FST, ST, SB, work, K, K2, K4,
                 diff0, hv, hg, AB, AC, SBB, ABB, BBB, BBD, CBD, BDD, **context):
        """Compute right hand side of Navier Stokes
        
        args:
            rhs         The right hand side to be returned
            u_hat       The FFT of the velocity at current time. May differ from
                        context.U_hat since it is set by the integrator

        Remaining args may be extracted from context:
            work        Work arrays
            FFT         Transform class from mpiFFT4py
            K           Scaled wavenumber mesh
            K2          sum_i K[i]*K[i]
            K_over_K2   K / K2
            P_hat       Transformed pressure
        
        """

        # Nonlinear convection term at current u_hat
        H_hat = self.nonlinear(H_hat, u_hat, g, K, FST, SB, ST, work)
                
        # Assemble convection with Adams-Bashforth at time = n+1/2
        H_hat0 = assembleAB(H_hat0, H_hat, H_hat1)
        
        # Assemble hv, hg and remaining rhs
        hv[:] = -K2*BBD.matvec(H_hat0[0])
        hv -= 1j*K[1]*CBD.matvec(H_hat0[1])
        hv -= 1j*K[2]*CBD.matvec(H_hat0[2])        
        hg[:] = 1j*K[1]*BDD.matvec(H_hat0[2]) - 1j*K[2]*BDD.matvec(H_hat0[1])    
        
        rhs[0] = hv*params.dt
        rhs[1] = hg*2./params.nu
        
        rhs = self.add_linear(rhs, u_hat[0], g, diff0, AB, AC, SBB, ABB, BBB,
                              params.nu, params.dt, K2, K4)
        
        return rhs


#def ComputeRHS(rhs, u, g, hv, hg, H_hat, H_hat0, H_hat1, diff0, K, K2, AB, AC, SBB, ABB,
               #BBB, BBD, CBD, BDD, **context):
    
    #try:
        #H_hat = conv(H_hat, u)
    #except TypeError:
        #conv = getConvection(params.convection)
        #H_hat = conv(H_hat, u)
    
    #diff0[:] = 0
    
    ## Compute diffusion for g-equation
    #diff0[1] = AB.matvec(g, diff0[1])
    
    ## Compute diffusion++ for u-equation
    #diff0[0] = add_diffusion_u(diff0[0], u, AC, SBB, ABB, BBB, params.nu,
                               #params.dt, K2, K4)
    
    ## Assemble convection with Adams-Bashforth convection
    #assembleAB(H_hat, H_hat0, H_hat1)
    
    ## Assemble hv, hg and remaining rhs
    #hv[:] = -K2*BBD.matvec(H_hat0[0])
    #hv -= 1j*K[1]*CBD.matvec(H_hat0[1])
    #hv -= 1j*K[2]*CBD.matvec(H_hat0[2])        
    #hg[:] = 1j*K[1]*BDD.matvec(H_hat0[2]) - 1j*K[2]*BDD.matvec(H_hat0[1])    
    #rhs[0] = hv*params.dt + diff0[0]
    #rhs[1] = hg*2./params.nu + diff0[1]
    #return rhs

def set_vw_components(U_hat, u, g, CDB, TDMASolverD, K_over_K2, work, **context):
    f_hat = work[(u, 0)]
    f_hat[:] = -CDB.matvec(u)
    f_hat = TDMASolverD(f_hat)
    U_hat[1] = -1j*(K_over_K2[1]*f_hat - K_over_K2[2]*g)
    U_hat[2] = -1j*(K_over_K2[2]*f_hat + K_over_K2[1]*g)

def set_0_components(U_hat, h0_hat, u0_hat, H_hat0, U_hat0, BDD, ADD, Sk,
                     HelmholtzSolverU0, **context):
    if rank == 0:
        h0_hat[1] = H_hat0[1, :, 0, 0]
        h0_hat[2] = H_hat0[2, :, 0, 0]
        u0_hat[1] = U_hat0[1, :, 0, 0]
        u0_hat[2] = U_hat0[2, :, 0, 0]
        
        w = 2./params.nu * BDD.matvec(h0_hat[1])        
        w -= 2./params.nu * Sk[1, :, 0, 0]        
        w -= ADD.matvec(u0_hat[1])
        w += 2./params.nu/params.dt * BDD.matvec(u0_hat[1])        
        u0_hat[1] = HelmholtzSolverU0(u0_hat[1], w)
        
        w = 2./params.nu * BDD.matvec(h0_hat[2])
        w -= ADD.matvec(u0_hat[2])
        w += 2./params.nu/params.dt * BDD.matvec(u0_hat[2])
        u0_hat[2] = HelmholtzSolverU0(u0_hat[2], w)
        
        U_hat[1, :, 0, 0] = u0_hat[1]
        U_hat[2, :, 0, 0] = u0_hat[2]
    
#@profile
def solve(context):
    global conv, profiler, timer
    
    timer = Timer()
    #conv = getConvection(params.convection)
    computeRHS = ComputeRHS()
    
    if params.make_profile: profiler = cProfile.Profile()
    
    while params.t < params.T-1e-14:
        params.t += params.dt
        params.tstep += 1

        context.dU[:] = 0
        dU = computeRHS(context.dU, context.U_hat, **context)
        
        u = context.BiharmonicSolverU(context.u, dU[0])
        g = context.HelmholtzSolverG(context.g, dU[1])
        
        set_vw_components(**context)
                
        # Remains to fix wavenumber 0
        set_0_components(**context)

        update(context)

        context.hdf5file.update(params, **context)

        # Rotate solutions
        context.U_hat0[:] = context.U_hat
        context.H_hat1[:] = context.H_hat

        timer()

        if params.tstep == 1 and params.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()

    timer.final(MPI, rank)

    if params.make_profile:
        results = create_profile(profiler, comm, MPI, rank)

    regression_test(context)

    context.hdf5file.close()
