__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *
from ..shen.Matrices import BBBmat, SBBmat, ABBmat, BBDmat, CBDmat, CDDmat, ADDmat, BDDmat, CDBmat
from ..shen.Helmholtz import Helmholtz, TDMA, Biharmonic
from ..shen import SFTc

assert config.precision == "double"
hdf5file = HDF5Writer(comm, float, {"U":U[0], "V":U[1], "W":U[2], "P":P}, config.solver+".h5", 
                      mesh={"x": points, "xp": pointsp, "y": x1, "z": x2})  

HelmholtzSolverG = Helmholtz(N[0], sqrt(K2[0]+2.0/nu/dt), ST.quad, False)
BiharmonicSolverU = Biharmonic(N[0], -nu*dt/2., 1.+nu*dt*K2[0], -(K2[0] + nu*dt/2.*K2[0]**2), SB.quad)
HelmholtzSolverP = Helmholtz(N[0], sqrt(K2[0]), SN.quad, True)
HelmholtzSolverU0 = Helmholtz(N[0], sqrt(2./nu/dt), ST.quad, False)

TDMASolverD = TDMA(ST.quad, False)

alfa = K2[0] - 2.0/nu/dt
CDD = CDDmat(K[0, :, 0, 0])

# Matrics for biharmonic equation
CBD = CBDmat(K[0, :, 0, 0])
ABB = ABBmat(K[0, :, 0, 0])
BBB = BBBmat(K[0, :, 0, 0], SB.quad)
SBB = SBBmat(K[0, :, 0, 0])

# Matrices for Helmholtz equation
ADD = ADDmat(K[0, :, 0, 0])
BDD = BDDmat(K[0, :, 0, 0], ST.quad)

# 
BBD = BDDmat(K[0, :, 0, 0], SB.quad)
CDB = CDBmat(K[0, :, 0, 0])

def solvePressure(P_hat, Ni):
    """Solve for pressure if Ni is fst of convection"""
    F_tmp[0] = 0
    SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], Ni[0, u_slice], Ni[1, u_slice], Ni[2, u_slice], F_tmp[0, p_slice])    
    P_hat = HelmholtzSolverP(P_hat, F_tmp[0])
    return P_hat

def Cross(a, b, c, S):
    H[0] = a[1]*b[2]-a[2]*b[1]
    H[1] = a[2]*b[0]-a[0]*b[2]
    H[2] = a[0]*b[1]-a[1]*b[0]
    c[0] = FST.fst(H[0], c[0], S)
    c[1] = FST.fst(H[1], c[1], S)
    c[2] = FST.fst(H[2], c[2], S)    
    return c

def Curl(a_hat, c, S):
    F_tmp[:] = 0
    U_tmp2[:] = 0
    SFTc.Mult_CTD_3D(N[0], a_hat[1], a_hat[2], F_tmp[1], F_tmp[2])
    dvdx = U_tmp2[1] = FST.ifct(F_tmp[1], U_tmp2[1], S)
    dwdx = U_tmp2[2] = FST.ifct(F_tmp[2], U_tmp2[2], S)
    #c[0] = FST.ifst(1j*K[1]*a_hat[2] - 1j*K[2]*a_hat[1], c[0], S)
    c[0] = FST.ifst(g, c[0], ST)
    c[1] = FST.ifst(1j*K[2]*a_hat[0], c[1], SB)
    c[1] -= dwdx
    c[2] = FST.ifst(1j*K[1]*a_hat[0], c[2], SB)
    c[2] *= -1.0
    c[2] += dvdx
    return c

#@profile
def standardConvection(c, U, U_hat):
    c[:] = 0
    U_tmp[:] = 0
    
    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = CDB.matvec(U_hat[0])
    F_tmp[0] = TDMASolverD(F_tmp[0])    
    dudx = U_tmp[0] = FST.ifst(F_tmp[0], U_tmp[0], ST)   
        
    SFTc.Mult_CTD_3D(N[0], U_hat[1], U_hat[2], F_tmp[1], F_tmp[2])
    dvdx = U_tmp[1] = FST.ifct(F_tmp[1], U_tmp[1], ST)
    dwdx = U_tmp[2] = FST.ifct(F_tmp[2], U_tmp[2], ST)
    
    #dudx = U_tmp[0] = chebDerivative_3D0(U[0], U_tmp[0])
    #dvdx = U_tmp[1] = chebDerivative_3D0(U[1], U_tmp[1])
    #dwdx = U_tmp[2] = chebDerivative_3D0(U[2], U_tmp[2])    
    
    U_tmp2[:] = 0
    dudy_h = 1j*K[1]*U_hat[0]
    dudy = U_tmp2[0] = FST.ifst(dudy_h, U_tmp2[0], SB)    
    dudz_h = 1j*K[2]*U_hat[0]
    dudz = U_tmp2[1] = FST.ifst(dudz_h, U_tmp2[1], SB)
    H[0] = U[0]*dudx + U[1]*dudy + U[2]*dudz
    c[0] = FST.fst(H[0], c[0], ST)
    
    U_tmp2[:] = 0
    
    dvdy_h = 1j*K[1]*U_hat[1]    
    dvdy = U_tmp2[0] = FST.ifst(dvdy_h, U_tmp2[0], ST)
    ##########
    
    dvdz_h = 1j*K[2]*U_hat[1]
    dvdz = U_tmp2[1] = FST.ifst(dvdz_h, U_tmp2[1], ST)
    H[1] = U[0]*dvdx + U[1]*dvdy + U[2]*dvdz
    c[1] = FST.fst(H[1], c[1], ST)
    
    U_tmp2[:] = 0
    dwdy_h = 1j*K[1]*U_hat[2]
    dwdy = U_tmp2[0] = FST.ifst(dwdy_h, U_tmp2[0], ST)
    
    dwdz_h = 1j*K[2]*U_hat[2]
    dwdz = U_tmp2[1] = FST.ifst(dwdz_h, U_tmp2[1], ST)
    
    #########
    
    H[2] = U[0]*dwdx + U[1]*dwdy + U[2]*dwdz
    c[2] = FST.fst(H[2], c[2], ST)
    from IPython import embed; embed()
    
    return c


def getConvection(convection):
    if convection == "Standard":
        
        def Conv(H_hat, U, U_hat):
            H_hat = standardConvection(H_hat, U, U_hat)
            H_hat[:] *= -1 
            return H_hat

    elif convection == "Vortex":
        
        def Conv(H_hat, U, U_hat):
            U_tmp[:] = Curl(U_hat, U_tmp, ST)
            H_hat[:] = Cross(U, U_tmp, H_hat, ST)
            return H_hat
        
    return Conv           

conv = getConvection(config.convection)
    
#@profile
def ComputeRHS(dU):
    global hv
    
    H_hat[:] = conv(H_hat, U0, U_hat0)    
    diff0[:] = 0
    
    # Compute diffusion for g-equation
    SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GL", -1, alfa, g, diff0[1])
    
    # Compute diffusion++ for u-equation
    diff0[0] = nu*dt/2.*SBB.matvec(u)
    diff0[0] += (1. - nu*dt*K2) * ABB.matvec(u)
    diff0[0] -= (K2 - nu*dt/2.*K2**2)*BBB.matvec(u)
    
    # Compute convection
    H0[:] = 1.5*H - 0.5*H1
    H_hat0[:] = 1.5*H_hat - 0.5*H_hat1
    H_hat0[:] *= dealias    
    
    # Following modification is critical for accuracy with vortex convection, but it makes standard perform worse
    #hv[:] = -K2*BBD.matvec(H_hat0[0])
    hv[:] = FST.fss(H0[0], hv, SB)
    hv *= -K2
    hv *= dealias
    
    # Following does not seem to be critical
    hv -= 1j*K[1]*CBD.matvec(H_hat0[1])
    hv -= 1j*K[2]*CBD.matvec(H_hat0[2])    
    #dH1dx = U_tmp[1] = FST.chebDerivative_3D0(H0[1], U_tmp[1], SB)
    #dH2dx = U_tmp[2] = FST.chebDerivative_3D0(H0[2], U_tmp[2], SB)
    #F_tmp[1] = FST.fss(dH1dx, F_tmp[1], SB)
    #F_tmp[2] = FST.fss(dH2dx, F_tmp[2], SB)
    #hv -= 1j*K[1]*F_tmp[1]
    #hv -= 1j*K[2]*F_tmp[2]
    
    #hg[:] = 1j*K[1]*BDD.matvec(H_hat0[2]) - 1j*K[2]*BDD.matvec(H_hat0[1])
    F_tmp[1] = FST.fss(H0[1], F_tmp[1], ST)
    F_tmp[2] = FST.fss(H0[2], F_tmp[2], ST)
    hg[:] = 1j*K[1]*F_tmp[2] - 1j*K[2]*F_tmp[1]
    hg[:] *= dealias
    
    dU[0] = hv*dt + diff0[0]
    dU[1] = hg*2./nu + diff0[1]
        
    return dU

def regression_test(**kw):
    pass

#@profile
def solve():
    timer = Timer()
    
    while config.t < config.T-1e-14:
        config.t += dt
        config.tstep += 1

        dU[:] = 0
        dU[:] = ComputeRHS(dU)
        
        U_hat[0] = BiharmonicSolverU(U_hat[0], dU[0])
        g[:] = HelmholtzSolverG(g, dU[1])
        
        f_hat = F_tmp[0]
        f_hat = - CDB.matvec(U_hat[0])
        f_hat = TDMASolverD(f_hat)
        
        U_hat[1] = -1j*(K_over_K2[1]*f_hat - K_over_K2[2]*g)
        U_hat[2] = -1j*(K_over_K2[2]*f_hat + K_over_K2[1]*g) 
        
        # Remains to fix wavenumber 0
        
        u0_hat = zeros((3, N[0]), dtype=complex)
        h0_hat = zeros((3, N[0]), dtype=complex)
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
        
        U[0] = FST.ifst(U_hat[0], U[0], SB)
        for i in range(1, 3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)

        update(**globals())
 
        # Rotate velocities
        U_hat0[:] = U_hat
        U0[:] = U
        H1[:] = H
        H_hat1[:] = H_hat
                
        timer()
        
        if config.tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
            
    timer.final(MPI, rank)
    
    if config.make_profile:
        results = create_profile(**globals())
                
    regression_test(**globals())

    hdf5file.close()
