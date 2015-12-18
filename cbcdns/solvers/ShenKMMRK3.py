__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ShenKMM import *

a = (8./15., 5./12., 3./4.)
b = (0.0, -17./60., -5./12.)

HelmholtzSolverG = (Helmholtz(N[0], sqrt(K[1, 0]**2+K[2, 0]**2+2.0/nu/(a[0]+b[0])/dt), ST.quad, False),
                    Helmholtz(N[0], sqrt(K[1, 0]**2+K[2, 0]**2+2.0/nu/(a[1]+b[1])/dt), ST.quad, False),
                    Helmholtz(N[0], sqrt(K[1, 0]**2+K[2, 0]**2+2.0/nu/(a[2]+b[2])/dt), ST.quad, False)
                    )

BiharmonicSolverU = (Biharmonic(N[0], -nu*a[0]*dt/2., 1.+nu*a[0]*dt*K2[0], -(K2[0] + nu*a[0]*dt/2.*K2[0]**2), SB.quad),
                     Biharmonic(N[0], -nu*(a[1]+b[1])*dt/2., 1.+nu*(a[1]+b[1])*dt*K2[0], -(K2[0] + nu*(a[1]+b[1])*dt/2.*K2[0]**2), SB.quad),
                     Biharmonic(N[0], -nu*(a[2]+b[2])*dt/2., 1.+nu*(a[2]+b[2])*dt*K2[0], -(K2[0] + nu*(a[2]+b[2])*dt/2.*K2[0]**2), SB.quad)
                     )

HelmholtzSolverP = Helmholtz(N[0], sqrt(K2[0]), SN.quad, True)

HelmholtzSolverU0 = (Helmholtz(N[0], sqrt(2./nu/(a[0]+b[0])/dt), ST.quad, False),
                     Helmholtz(N[0], sqrt(2./nu/(a[1]+b[1])/dt), ST.quad, False),
                     Helmholtz(N[0], sqrt(2./nu/(a[2]+b[2])/dt), ST.quad, False))

TDMASolverD = TDMA(ST.quad, False)

alfa = K[1, 0]**2+K[2, 0]**2-2.0/nu/dt
CDD = CDDmat(K[0, :, 0, 0])

# Matrics for biharmonic equation
CBD = CBDmat(K[0, :, 0, 0])
ABB = ABBmat(K[0, :, 0, 0])
BBB = BBBmat(K[0, :, 0, 0], ST.quad)
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

def Curl(a, c, S):
    F_tmp[:] = 0
    U_tmp2[:] = 0
    SFTc.Mult_CTD_3D(N[0], a[1], a[2], F_tmp[1], F_tmp[2])
    dvdx = U_tmp2[1] = FST.ifct(F_tmp[1], U_tmp2[1], S)
    dwdx = U_tmp2[2] = FST.ifct(F_tmp[2], U_tmp2[2], S)
    #c[0] = FST.ifst(1j*K[1]*a[2] - 1j*K[2]*a[1], c[0], S)
    c[0] = FST.ifst(g, c[0], S)
    c[1] = FST.ifst(1j*K[2]*a[0], c[1], SB)
    c[1] -= dwdx
    c[2] = FST.ifst(1j*K[1]*a[0], c[2], SB)
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
    #F_tmp[0] = FST.fst(U[1], F_tmp[0], SN)
    #dvdy_h = 1j*K[1]*F_tmp[0]    
    #dvdy = U_tmp2[0] = FST.ifst(dvdy_h, U_tmp2[0], SN)
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
    
    #F_tmp[0] = FST.fst(U[2], F_tmp[0], SN)
    #dwdz_h = 1j*K[2]*F_tmp[0]    
    #dwdz = U_tmp2[1] = FST.ifst(dwdz_h, U_tmp2[1], SN)    
    #########
    
    H[2] = U[0]*dwdx + U[1]*dwdy + U[2]*dwdz
    c[2] = FST.fst(H[2], c[2], ST)
    
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
    
U_hat1 = U_hat0.copy()
U_hat2 = U_hat0.copy()
hg0 = hg.copy()
hv0 = hv.copy()
u0_hat = zeros((3, N[0]), dtype=complex)
h0_hat = zeros((3, N[0]), dtype=complex)
h0 = zeros((2, N[0]), dtype=complex)
h1 = zeros((2, N[0]), dtype=complex)
#@profile

def RKstep(U_hat, g, dU, rk):
    global conv1, hv, hg, hv0, hg0, a, b, h0, h1
    
    if rk > 0: # For rk=0 the correct values are already in U
        U[0] = FST.ifst(U_hat[0], U[0], SB)
        for i in range(1, 3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)
    
    # Compute convection
    H_hat[:] = conv(H_hat, U0, U_hat0)    
    H_hat[:] *= dealias    

    diff0[:] = 0
    
    # Compute diffusion for g-equation
    SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GL", -1, alfa, g, diff0[1])
    
    # Compute diffusion++ for u-equation
    diff0[0] = nu*(a[rk]+b[rk])*dt/2.*SBB.matvec(U_hat[0])
    diff0[0] += (1. - nu*(a[rk]+b[rk])*dt*K2) * ABB.matvec(U_hat[0])
    diff0[0] -= (K2 - nu*(a[rk]+b[rk])*dt/2.*K2**2)*BBB.matvec(U_hat[0])
    
    
    #hv[:] = -K2*BBD.matvec(H_hat[0])
    hv[:] = FST.fss(H[0], hv, SB)
    hv *= -K2
    hv -= 1j*K[1]*CBD.matvec(H_hat[1])
    hv -= 1j*K[2]*CBD.matvec(H_hat[2])    
    #hg[:] = 1j*K[1]*BDD.matvec(H_hat[2]) - 1j*K[2]*BDD.matvec(H_hat[1])
    F_tmp[1] = FST.fss(H[1], F_tmp[1], ST)
    F_tmp[2] = FST.fss(H[2], F_tmp[2], ST)
    hg[:] = 1j*K[1]*F_tmp[2] - 1j*K[2]*F_tmp[1]
    
    dU[0] = (hv*a[rk] + hv0*b[rk])*dt + diff0[0]
    dU[1] = (hg*a[rk] + hg0*b[rk])*2./nu + diff0[1]
    
    U_hat[0] = BiharmonicSolverU[rk](U_hat[0], dU[0])
    g[:] = HelmholtzSolverG[rk](g, dU[1])

    f_hat = F_tmp[0]
    f_hat = - CDB.matvec(U_hat[0])
    f_hat = TDMASolverD(f_hat)

    u0_hat[1, :] = U_hat[1, :, 0, 0]
    u0_hat[2, :] = U_hat[2, :, 0, 0]
    
    U_hat[1] = -1j*(K_over_K2[1]*f_hat - K_over_K2[2]*g)
    U_hat[2] = -1j*(K_over_K2[2]*f_hat + K_over_K2[1]*g) 
    
    # Remains to fix wavenumber 0    
    if rank == 0:
        h0_hat[1, :] = H_hat[1, :, 0, 0]
        h0_hat[2, :] = H_hat[2, :, 0, 0]
        
        h1[0] = BDD.matvec(h0_hat[1])
        h1[1] = BDD.matvec(h0_hat[2])
        h1[0] -= Sk[1, :, 0, 0]  # Subtract constant pressure gradient
        
        beta = 2./nu/(a[rk]+b[rk])/dt
        w = beta*(a[rk]*h1[0] + b[rk]*h0[0])*dt
        w -= ADD.matvec(u0_hat[1])
        w += beta*BDD.matvec(u0_hat[1])    
        u0_hat[1] = HelmholtzSolverU0[rk](u0_hat[1], w)
        
        w = beta*(a[rk]*h1[1] + b[rk]*h0[1])*dt
        w -= ADD.matvec(u0_hat[2])
        w += beta*BDD.matvec(u0_hat[2])    
        u0_hat[2] = HelmholtzSolverU0[rk](u0_hat[2], w)
            
        U_hat[1, :, 0, 0] = u0_hat[1]
        U_hat[2, :, 0, 0] = u0_hat[2]
    
    return U_hat

def regression_test(**kw):
    pass

#@profile
def solve():
    timer = Timer()
    
    while config.t < config.T-1e-10:
        config.t += dt
        config.tstep += 1

        dU[:] = 0
        hv0[:] = 0
        hg0[:] = 0
        h0[:] = 0
        for rk in range(3):            
            U_hat[:] = RKstep(U_hat, g, dU, rk)
            hv0[:] = hv
            hg0[:] = hg
            h0[:]  = h1
        
        U[0] = FST.ifst(U_hat[0], U[0], SB)
        for i in range(1, 3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)

        update(**globals())
 
        # Rotate velocities
        U_hat0[:] = U_hat
        U0[:] = U
                
        timer()
        
        if config.tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
            
    timer.final(MPI, rank)
    
    if config.make_profile:
        results = create_profile(**globals())
                
    regression_test(**globals())

    hdf5file.close()
