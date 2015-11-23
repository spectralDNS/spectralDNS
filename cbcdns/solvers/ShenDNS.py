__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *
from ..shen.Matrices import CDNmat, CDDmat, BDNmat, BDDmat, BDTmat
from ..shen.Helmholtz import Helmholtz, TDMA
from ..shen import SFTc

assert config.precision == "double"
hdf5file = HDF5Writer(comm, float, {"U":U[0], "V":U[1], "W":U[2], "P":P}, config.solver+".h5", 
                      mesh={"x": points, "xp": pointsp, "y": x1, "z": x2})  

HelmholtzSolverU = Helmholtz(N[0], sqrt(K[1, 0]**2+K[2, 0]**2+2.0/nu/dt), "GL", False)
HelmholtzSolverP = Helmholtz(N[0]-2, sqrt(K[1, 0]**2+K[2, 0]**2), SN.quad, True)
TDMASolverD = TDMA(ST.quad, False)
TDMASolverN = TDMA(SN.quad, True)

alfa = K[1, 0]**2+K[2, 0]**2-2.0/nu/dt
CDN = CDNmat(K[0, :, 0, 0])
BDN = BDNmat(K[0, :, 0, 0], ST.quad)
CDD = CDDmat(K[0, :, 0, 0])
BDD = BDDmat(K[0, :, 0, 0], ST.quad)
BDT = BDTmat(K[0, :, 0, 0], SN.quad)

#dpdx = P.copy()
#@profile
def pressuregrad(P_hat, dU):
    # Pressure gradient x-direction
    dU[0] -= CDN.matvec(P_hat)
    #dpdx[:] = FST.chebDerivative_3D0(P, dpdx, SN)
    #F_tmp[0] = FST.fct(dpdx, F_tmp[0], SN)
    #dU[0] -= BDT.matvec(F_tmp[0])
    
    # pressure gradient y-direction
    F_tmp[0] = BDN.matvec(P_hat)
    dU[1, :Nu] -= 1j*K[1, :Nu]*F_tmp[0, :Nu]
    
    # pressure gradient z-direction
    dU[2, :Nu] -= 1j*K[2, :Nu]*F_tmp[0, :Nu]    
    
    return dU

def pressurerhs(U_hat, dU):
    dU[:] = 0.
    SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], U_hat[0, u_slice], U_hat[1, u_slice], U_hat[2, u_slice], dU[p_slice])    
    dU[p_slice] *= -1./dt    
    return dU

def body_force(Sk, dU):
    dU[0, :Nu] -= Sk[0, :Nu]
    dU[1, :Nu] -= Sk[1, :Nu]
    dU[2, :Nu] -= Sk[2, :Nu]
    return dU

def Curl(a, c, S):
    F_tmp[:] = 0
    U_tmp[:] = 0
    SFTc.Mult_CTD_3D(N[0], a[1], a[2], F_tmp[1], F_tmp[2])
    dvdx = U_tmp[1] = FST.ifct(F_tmp[1], U_tmp[1], ST)
    dwdx = U_tmp[2] = FST.ifct(F_tmp[2], U_tmp[2], ST)
    c[0] = FST.ifst(1j*K[1]*a[2] - 1j*K[2]*a[1], c[0], S)
    c[1] = FST.ifst(1j*K[2]*a[0], c[1], S)
    c[1] -= dwdx
    c[2] = FST.ifst(1j*K[1]*a[0], c[2], S)
    c[2] *= -1.0
    c[2] += dvdx
    return c

def Div(a_hat):
    F_tmp[:] = 0
    U_tmp[:] = 0
    F_tmp[0] = CDD.matvec(a_hat[0])
    F_tmp[0] = TDMASolverD(F_tmp[0])    
    dudx = U_tmp[0] = FST.ifst(F_tmp[0], U_tmp[0], ST) 
    F_tmp[1] = BDD.matvec(a_hat[1])
    dvdy_h = 1j*K[1]*F_tmp[1]
    dvdy = U_tmp[1] = FST.ifst(dvdy_h, U_tmp[1], ST)
    F_tmp[2] = BDD.matvec(a_hat[2])
    dwdz_h = 1j*K[2]*F_tmp[2]
    dwdz = U_tmp[2] = FST.ifst(dwdz_h, U_tmp[2], ST)
    return dudx+dvdy+dwdz

#@profile
def standardConvection(c):
    c[:] = 0
    U_tmp[:] = 0
    
    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = CDD.matvec(U_hat0[0])
    F_tmp[0] = TDMASolverD(F_tmp[0])    
    dudx = U_tmp[0] = FST.ifst(F_tmp[0], U_tmp[0], ST)        
    
    SFTc.Mult_CTD_3D(N[0], U_hat0[1], U_hat0[2], F_tmp[1], F_tmp[2])
    dvdx = U_tmp[1] = FST.ifct(F_tmp[1], U_tmp[1], ST)
    dwdx = U_tmp[2] = FST.ifct(F_tmp[2], U_tmp[2], ST)
    
    #dudx = U_tmp[0] = chebDerivative_3D0(U0[0], U_tmp[0])
    #dvdx = U_tmp[1] = chebDerivative_3D0(U0[1], U_tmp[1])
    #dwdx = U_tmp[2] = chebDerivative_3D0(U0[2], U_tmp[2])    
    
    U_tmp2[:] = 0
    dudy_h = 1j*K[1]*U_hat0[0]
    dudy = U_tmp2[0] = FST.ifst(dudy_h, U_tmp2[0], ST)
    dudz_h = 1j*K[2]*U_hat0[0]
    dudz = U_tmp2[1] = FST.ifst(dudz_h, U_tmp2[1], ST)
    c[0] = FST.fss(U0[0]*dudx + U0[1]*dudy + U0[2]*dudz, c[0], ST)
    
    U_tmp2[:] = 0
    dvdy_h = 1j*K[1]*U_hat0[1]
    dvdy = U_tmp2[0] = FST.ifst(dvdy_h, U_tmp2[0], ST)
    dvdz_h = 1j*K[2]*U_hat0[1]
    dvdz = U_tmp2[1] = FST.ifst(dvdz_h, U_tmp2[1], ST)
    c[1] = FST.fss(U0[0]*dvdx + U0[1]*dvdy + U0[2]*dvdz, c[1], ST)
    
    U_tmp2[:] = 0
    dwdy_h = 1j*K[1]*U_hat0[2]
    dwdy = U_tmp2[0] = FST.ifst(dwdy_h, U_tmp2[0], ST)
    dwdz_h = 1j*K[2]*U_hat0[2]
    dwdz = U_tmp2[1] = FST.ifst(dwdz_h, U_tmp2[1], ST)
    c[2] = FST.fss(U0[0]*dwdx + U0[1]*dwdy + U0[2]*dwdz, c[2], ST)
    c *= -1
    return c

def divergenceConvection(c, add=False):
    """c_i = div(u_i u_j)"""
    if not add: 
        c.fill(0)
    else:
        c *= -1
    #U_tmp[0] = chebDerivative_3D0(U[0]*U[0], U_tmp[0])
    #U_tmp[1] = chebDerivative_3D0(U[0]*U[1], U_tmp[1])
    #U_tmp[2] = chebDerivative_3D0(U[0]*U[2], U_tmp[2])
    #c[0] = fss(U_tmp[0], c[0], ST)
    #c[1] = fss(U_tmp[1], c[1], ST)
    #c[2] = fss(U_tmp[2], c[2], ST)
    
    F_tmp[0] = FST.fst(U0[0]*U0[0], F_tmp[0], ST)
    F_tmp[1] = FST.fst(U0[0]*U0[1], F_tmp[1], ST)
    F_tmp[2] = FST.fst(U0[0]*U0[2], F_tmp[2], ST)
    
    c[0] += CDD.matvec(F_tmp[0])
    c[1] += CDD.matvec(F_tmp[1])
    c[2] += CDD.matvec(F_tmp[2])
    
    F_tmp2[0] = FST.fss(U0[0]*U0[1], F_tmp2[0], ST)
    F_tmp2[1] = FST.fss(U0[0]*U0[2], F_tmp2[1], ST)    
    c[0] += 1j*K[1]*F_tmp2[0] # duvdy
    c[0] += 1j*K[2]*F_tmp2[1] # duwdz
    
    F_tmp[0] = FST.fss(U0[1]*U0[1], F_tmp[0], ST)
    F_tmp[1] = FST.fss(U0[1]*U0[2], F_tmp[1], ST)
    F_tmp[2] = FST.fss(U0[2]*U0[2], F_tmp[2], ST)
    c[1] += 1j*K[1]*F_tmp[0]  # dvvdy
    c[1] += 1j*K[2]*F_tmp[1]  # dvwdz  
    c[2] += 1j*K[1]*F_tmp[1]  # dvwdy
    c[2] += 1j*K[2]*F_tmp[2]  # dwwdz
    c *= -1
    return c    

#@profile
def ComputeRHS(dU, jj):
    global conv0
    # Add convection to rhs
    if jj == 0:
        if config.convection == "Standard":
            conv0[:] = standardConvection(conv0) 
        elif config.convection == "Divergence":
            conv0[:] = divergenceConvection(conv0)
        elif config.convection == "Skew":
            conv0[:] = standardConvection(conv0) 
            conv0[:] = divergenceConvection(conv0, True)
            conv0 *= 0.5
        
        # Compute diffusion
        diff0[:] = 0
        SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GL", -1, alfa, U_hat0[0], diff0[0])
        SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GL", -1, alfa, U_hat0[1], diff0[1])
        SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GL", -1, alfa, U_hat0[2], diff0[2])    
    
    dU[:3] = 1.5*conv0 - 0.5*conv1
    dU[:3] *= dealias    
    
    # Add pressure gradient and body force
    dU = pressuregrad(P_hat, dU)
    dU = body_force(Sk, dU)
    
    # Scale by 2/nu factor
    dU[:3] *= 2./nu
    
    # Add diffusion
    dU[:3] += diff0
        
    return dU

def solvePressure(P, P_hat, U_hat):
    global F_tmp, F_tmp2
    U_tmp[:] = 0
    F_tmp2[:] = 0
    Ni = F_tmp2
    
    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = CDD.matvec(U_hat[0])
    F_tmp[0] = TDMASolverD(F_tmp[0])
    dudx = U_tmp[0] = FST.ifst(F_tmp[0], U_tmp[0], ST)      
    
    SFTc.Mult_CTD_3D(N[0], U_hat[1], U_hat[2], F_tmp[1], F_tmp[2])
    dvdx = U_tmp[1] = FST.ifct(F_tmp[1], U_tmp[1], ST)
    dwdx = U_tmp[2] = FST.ifct(F_tmp[2], U_tmp[2], ST)

    U_tmp2[:] = 0
    dudy_h = 1j*K[1]*U_hat[0]
    dudy = U_tmp2[0] = FST.ifst(dudy_h, U_tmp2[0], ST)
    dudz_h = 1j*K[2]*U_hat[0]
    dudz = U_tmp2[1] = FST.ifst(dudz_h, U_tmp2[1], ST)
    Ni[0] = FST.fst(U0[0]*dudx + U0[1]*dudy + U0[2]*dudz, Ni[0], ST)
    
    U_tmp2[:] = 0
    dvdy_h = 1j*K[1]*U_hat[1]
    dvdy = U_tmp2[0] = FST.ifst(dvdy_h, U_tmp2[0], ST)
    dvdz_h = 1j*K[2]*U_hat[1]
    dvdz = U_tmp2[1] = FST.ifst(dvdz_h, U_tmp2[1], ST)
    Ni[1] = FST.fst(U0[0]*dvdx + U0[1]*dvdy + U0[2]*dvdz, Ni[1], ST)
    
    U_tmp2[:] = 0
    dwdy_h = 1j*K[1]*U_hat[2]
    dwdy = U_tmp2[0] = FST.ifst(dwdy_h, U_tmp2[0], ST)
    dwdz_h = 1j*K[2]*U_hat[2]
    dwdz = U_tmp2[1] = FST.ifst(dwdz_h, U_tmp2[1], ST)
    Ni[2] = FST.fst(U0[0]*dwdx + U0[1]*dwdy + U0[2]*dwdz, Ni[2], ST)
    
    F_tmp[0] = 0
    SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], Ni[0, u_slice], Ni[1, u_slice], Ni[2, u_slice], F_tmp[0, p_slice])    
    P_hat = HelmholtzSolverP(P_hat, F_tmp[0])
    P = FST.ifst(P_hat, P, SN)

    
def Divu(U, U_hat, c):
    c[:] = 0
    SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], 
                       U_hat[0, u_slice], U_hat[1, u_slice], U_hat[2, u_slice], c[p_slice])
    #c[p_slice] = SFTc.TDMA_3D(a0N, b0N, bcN, c0N, c[p_slice])
    c = TDMASolverN(c)
        
    return c

def regression_test(**kw):
    pass

#@profile
def solve():
    timer = Timer()
    
    while config.t < config.T-1e-8:
        config.t += dt
        config.tstep += 1

        # Tentative momentum solve
        for jj in range(config.velocity_pressure_iters):
            dU[:] = 0
            dU[:] = ComputeRHS(dU, jj)                
            U_hat[0] = HelmholtzSolverU(U_hat[0], dU[0])
            U_hat[1] = HelmholtzSolverU(U_hat[1], dU[1])
            U_hat[2] = HelmholtzSolverU(U_hat[2], dU[2])
        
            # Pressure correction
            dU[3] = pressurerhs(U_hat, dU[3]) 
            Pcorr[:] = HelmholtzSolverP(Pcorr, dU[3])

            # Update pressure
            P_hat[p_slice] += Pcorr[p_slice]

            if jj == 0 and config.print_divergence_progress:
                print "   Divergence error"
            if config.print_divergence_progress:
                print "         Pressure correction norm %2.6e" %(linalg.norm(Pcorr))

        for i in range(3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)
                 
        # Update velocity
        dU[:] = 0
        pressuregrad(Pcorr, dU)        
        dU[0] = TDMASolverD(dU[0])
        dU[1] = TDMASolverD(dU[1])
        dU[2] = TDMASolverD(dU[2])
        U_hat[:3, u_slice] += dt*dU[:3, u_slice]  # + since pressuregrad computes negative pressure gradient

        for i in range(3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)

        update(**globals())
 
        # Rotate velocities
        U_hat1[:] = U_hat0
        U_hat0[:] = U_hat
        U0[:] = U
        
        P[:] = FST.ifst(P_hat, P, SN)        
        conv1[:] = conv0
                
        timer()
        
        if config.tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
            
    timer.final(MPI, rank)
    
    if config.make_profile:
        results = create_profile(**globals())
                
    regression_test(**globals())

    hdf5file.close()
