__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from .spectralinit import *
from spectralDNS.mesh.channel import setup
from ..shen.Matrices import BBBmat, SBBmat, ABBmat, BBDmat, CBDmat, CDDmat, ADDmat, BDDmat, CDBmat, BiharmonicCoeff, HelmholtzCoeff
from ..shen.la import Helmholtz, TDMA, Biharmonic
from ..shen import SFTc

vars().update(setup['KMM'](**vars()))

assert params.precision == "double"

def backward_velocity(U, U_hat, FST):
    U[0] = FST.ifst(U_hat[0], U[0], SB)
    for i in range(1, 3):
        U[i] = FST.ifst(U_hat[i], U[i], ST)
    return U

def forward_velocity(U_hat, U, FST):
    U_hat[0] = FST.fst(U[0], U_hat[0], SB)
    for i in range(1, 3):
        U_hat[i] = FST.fst(U[i], U_hat[i], ST)
    return U_hat

class KMMWriter(HDF5Writer):
    
    def update_components(self, U, U0, U_hat, U_hat0, FST, SB, ST, params, **kw):
        """Transform to real data when storing the solution"""
        if self.check_if_write(params) or params.tstep % params.checkpoint == 0:
            U = backward_velocity(U, U_hat, FST)

        if params.tstep % params.checkpoint == 0:
            U0 = backward_velocity(U0, U_hat0, FST)

hdf5file = KMMWriter({"U":U[0], "V":U[1], "W":U[2]}, 
                      chkpoint={'current':{'U':U}, 'previous':{'U':U0}},
                      filename=params.solver+".h5", mesh={"x": x0, "y": x1, "z": x2})

nu, dt, N = params.nu, params.dt, params.N
K4 = K2**2
kx = K[0, :, 0, 0]
HelmholtzSolverG = Helmholtz(N[0], sqrt(K2[0]+2.0/nu/dt), ST.quad, False)
BiharmonicSolverU = Biharmonic(N[0], -nu*dt/2., 1.+nu*dt*K2[0], -(K2[0] + nu*dt/2.*K4[0]), quad=SB.quad, solver="cython")
HelmholtzSolverU0 = Helmholtz(N[0], sqrt(2./nu/dt), ST.quad, False)

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

# 
BBD = BBDmat(kx, SB.quad)
CDB = CDBmat(kx)

def solvePressure(P_hat, Ni):
    """Solve for pressure if Ni is fst of convection"""
    pass
    #F_tmp[0] = 0
    #SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], Ni[0, u_slice], Ni[1, u_slice], Ni[2, u_slice], F_tmp[0, p_slice])    
    #HelmholtzSolverP = Helmholtz(N[0], sqrt(K2[0]), SN.quad, True)
    #P_hat = HelmholtzSolverP(P_hat, F_tmp[0])
    #return P_hat

#@profile
def Cross(a, b, c, S):
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
def Curl(a_hat, c, S):
    F_tmp = work[(a_hat, 0, False)]
    F_tmp2 = work[(a_hat, 2, False)]
    Uc = work[(c, 2, False)]
    # Mult_CTD_3D_n is projection to T of d(a_hat)/dx (for components 1 and 2 of a_hat) 
    # Corresponds to CTD.matvec(a_hat[1])/BTT.dd, CTD.matvec(a_hat[2])/BTT.dd
    SFTc.Mult_CTD_3D_n(N[0], a_hat[1], a_hat[2], F_tmp[1], F_tmp[2])
    dvdx = Uc[1] = FST.ifct(F_tmp[1], Uc[1], S, dealias=params.dealias)
    dwdx = Uc[2] = FST.ifct(F_tmp[2], Uc[2], S, dealias=params.dealias)
    c[0] = FST.ifst(g, c[0], ST, dealias=params.dealias)
    F_tmp2[:2] = mult_K1j(K, a_hat[0], F_tmp2[:2])
    
    ## Set highest wavenumber to zero
    #if FST.num_processes == 1:
        #F_tmp2[:, :, -N[1]/2] = 0
    #elif FST.rank == FST.num_processes / 2:
        #F_tmp2[:, :, 0] = 0
    #F_tmp2[:, :, :, -1] = 0
    
    c[1] = FST.ifst(F_tmp2[0], c[1], SB, dealias=params.dealias)
    c[1] -= dwdx
    c[2] = FST.ifst(F_tmp2[1], c[2], SB, dealias=params.dealias)
    c[2] += dvdx
    return c

#@profile
def standardConvection(c, U_dealiased, U_hat):
    c[:] = 0
    U = U_dealiased
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

def divergenceConvection(c, U, U_hat, add=False):
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

def getConvection(convection):
    if convection == "Standard":
        
        def Conv(H_hat, U_hat):
            
            U_dealiased = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            U_dealiased[0] = FST.ifst(U_hat[0], U_dealiased[0], SB, params.dealias) 
            for i in range(1, 3):
                U_dealiased[i] = FST.ifst(U_hat[i], U_dealiased[i], ST, params.dealias)

            H_hat = standardConvection(H_hat, U_dealiased, U_hat)
            H_hat[:] *= -1
            return H_hat
        
    elif convection == "Divergence":
        
        def Conv(H_hat, U_hat):
            
            U_dealiased = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            U_dealiased[0] = FST.ifst(U_hat[0], U_dealiased[0], SB, params.dealias) 
            for i in range(1, 3):
                U_dealiased[i] = FST.ifst(U_hat[i], U_dealiased[i], ST, params.dealias)

            H_hat = divergenceConvection(H_hat, U_dealiased, U_hat, False)
            H_hat[:] *= -1
            return H_hat
        
    elif convection == "Skew":
        
        def Conv(H_hat, U_hat):
            
            U_dealiased = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            U_dealiased[0] = FST.ifst(U_hat[0], U_dealiased[0], SB, params.dealias) 
            for i in range(1, 3):
                U_dealiased[i] = FST.ifst(U_hat[i], U_dealiased[i], ST, params.dealias)

            H_hat = standardConvection(H_hat, U_dealiased, U_hat)
            H_hat = divergenceConvection(H_hat, U_dealiased, U_hat, True)            
            H_hat *= -0.5
            return H_hat

    elif convection == "Vortex":
        
        def Conv(H_hat, U_hat):
            
            U_dealiased = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            curl_dealiased = work[((3,)+FST.work_shape(params.dealias), float, 1)]
            U_dealiased[0] = FST.ifst(U_hat[0], U_dealiased[0], SB, params.dealias) 
            for i in range(1, 3):
                U_dealiased[i] = FST.ifst(U_hat[i], U_dealiased[i], ST, params.dealias)
            
            curl_dealiased[:] = Curl(U_hat, curl_dealiased, ST)
            H_hat[:] = Cross(U_dealiased, curl_dealiased, H_hat, ST)
            
            return H_hat
        
    return Conv           

conv = getConvection(params.convection)

def compute_derivatives(duidxj, U_hat):
    duidxj[:] = 0
    F_tmp = work[(U_hat, 0)]
    
    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = CDB.matvec(U_hat[0])
    F_tmp[0] = TDMASolverD(F_tmp[0])    
    duidxj[0, 0] = FST.ifst(F_tmp[0], duidxj[0, 0], ST)
    SFTc.Mult_CTD_3D_n(N[0], U_hat[1], U_hat[2], F_tmp[1], F_tmp[2])
    duidxj[1, 0] = dvdx = FST.ifct(F_tmp[1], duidxj[1, 0], ST)  # proj to Cheb
    duidxj[2, 0] = dwdx = FST.ifct(F_tmp[2], duidxj[2, 0], ST)  # proj to Cheb
    duidxj[0, 1] = dudy = FST.ifst(1j*K[1]*U_hat[0], duidxj[0, 1], SB) # ShenB
    duidxj[0, 2] = dudz = FST.ifst(1j*K[2]*U_hat[0], duidxj[0, 2], SB)
    duidxj[1, 1] = dvdy = FST.ifst(1j*K[1]*U_hat[1], duidxj[1, 1], ST)
    duidxj[1, 2] = dvdz = FST.ifst(1j*K[2]*U_hat[1], duidxj[1, 2], ST)    
    duidxj[2, 1] = dwdy = FST.ifst(1j*K[1]*U_hat[2], duidxj[2, 1], ST)
    duidxj[2, 2] = dwdz = FST.ifst(1j*K[2]*U_hat[2], duidxj[2, 2], ST)
    return duidxj

@optimizer
def add_diffusion_u(u, d, AC, SBB, ABB, BBB, nu, dt, K2, K4):
    d[:] = nu*dt/2.*SBB.matvec(u)
    d += (1. - nu*dt*K2)*ABB.matvec(u)
    d -= (K2 - nu*dt/2.*K4)*BBB.matvec(u)    
    return d

@optimizer
def assembleAB(H_hat, H_hat0, H_hat1):
    H_hat0[:] = 1.5*H_hat - 0.5*H_hat1
    
#@profile
def ComputeRHS(dU, U_hat):
    global hv, H_hat
    
    H_hat = conv(H_hat, U_hat0)    
    diff0[:] = 0
    
    # Compute diffusion for g-equation
    diff0[1] = AB.matvec(g, diff0[1])
    
    # Compute diffusion++ for u-equation
    diff0[0] = add_diffusion_u(u, diff0[0], AC, SBB, ABB, BBB, nu, dt, K2, K4)
    
    # Assemble convection with Adams-Bashforth convection
    assembleAB(H_hat, H_hat0, H_hat1)    
    
    # Assemble hv, hg and remaining dU
    hv[:] = -K2*BBD.matvec(H_hat0[0])
    hv -= 1j*K[1]*CBD.matvec(H_hat0[1])
    hv -= 1j*K[2]*CBD.matvec(H_hat0[2])        
    hg[:] = 1j*K[1]*BDD.matvec(H_hat0[2]) - 1j*K[2]*BDD.matvec(H_hat0[1])    
    dU[0] = hv*dt + diff0[0]
    dU[1] = hg*2./nu + diff0[1]        
    return dU

#@profile
def solve():
    global dU, U_hat, g, u, conv, profiler, timer
    
    timer = Timer()
    conv = getConvection(params.convection)
    if params.make_profile: profiler = cProfile.Profile()
    
    while params.t < params.T-1e-14:
        params.t += params.dt
        params.tstep += 1

        dU[:] = 0
        dU = ComputeRHS(dU, U_hat)
        
        U_hat[0] = BiharmonicSolverU(U_hat[0], dU[0])
        g[:] = HelmholtzSolverG(g, dU[1])
        
        f_hat = work[(U_hat[0], 0)]
        f_hat[:] = -CDB.matvec(U_hat[0])
        f_hat = TDMASolverD(f_hat)
        
        U_hat[1] = -1j*(K_over_K2[1]*f_hat - K_over_K2[2]*g)
        U_hat[2] = -1j*(K_over_K2[2]*f_hat + K_over_K2[1]*g) 
        
        # Remains to fix wavenumber 0
        if rank == 0:
            h0_hat[1] = H_hat0[1, :, 0, 0]
            h0_hat[2] = H_hat0[2, :, 0, 0]
            u0_hat[1] = U_hat0[1, :, 0, 0]
            u0_hat[2] = U_hat0[2, :, 0, 0]
            
            w = 2./nu * BDD.matvec(h0_hat[1])        
            w -= 2./nu * Sk[1, :, 0, 0]        
            w -= ADD.matvec(u0_hat[1])
            w += 2./nu/dt * BDD.matvec(u0_hat[1])        
            u0_hat[1] = HelmholtzSolverU0(u0_hat[1], w)
            
            w = 2./nu * BDD.matvec(h0_hat[2])
            w -= ADD.matvec(u0_hat[2])
            w += 2./nu/dt * BDD.matvec(u0_hat[2])
            u0_hat[2] = HelmholtzSolverU0(u0_hat[2], w)
            
            U_hat[1, :, 0, 0] = u0_hat[1]
            U_hat[2, :, 0, 0] = u0_hat[2]
        
        update(**globals())
 
        hdf5file.update(**globals())

        # Rotate solutions
        U_hat0[:] = U_hat
        H_hat1[:] = H_hat
                
        timer()
        
        if params.tstep == 1 and params.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()     
            
    timer.final(MPI, rank)
    
    if params.make_profile:
        results = create_profile(**globals())
                
    regression_test(**globals())

    hdf5file.close()
