# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:22:12 2015

@author: Diako Darian


The code below employs a projection method to solve the 3D incompressible MHD equations. 
The spatial discretization is done by a spectral method using periodic Fourier basis functions 
in y and z directions, and non-periodic Chebyshev polynomials in x direction. 

Time discretization is done by a semi-implicit Crank-Nicolson scheme.

"""

##FIXME Not working with latest version of spectralDNS

from spectralinit import *
from ..shen import SFTc
from ..shenGeneralBCs.shentransform import ShenBasis
from ..shen.Matrices import CDNmat, CDDmat, BNDmat
from ..shen.Helmholtz import Helmholtz, TDMA

assert config.precision == "double"
hdf5file = HDF5Writer(comm, float, {"U":U[0], "V":U[1], "W":U[2], "P":P}, config.solver+".h5", 
                      mesh={"x": points, "xp": pointsp, "y": x1, "z": x2})  

BC = array([0,1,0, 0,1,0])
SR = ShenBasis(BC, quad="GL")

A_breve = zeros((N[0]-2,N[0]-2))
B_breve = zeros((N[0]-2,N[0]-2))
C_breve = zeros((N[0]-2,N[0]-2))

HelmholtzSolverU = Helmholtz(N[0], sqrt(K[1, 0]**2+K[2, 0]**2+2.0/nu/dt), "GL", False)
HelmholtzSolverB = Helmholtz(N[0], sqrt(K[1, 0]**2+K[2, 0]**2+2.0/eta/dt), "GL", True)
HelmholtzSolverP = Helmholtz(N[0], sqrt(K[1, 0]**2+K[2, 0]**2), SN.quad, True)
TDMASolverD = TDMA(ST.quad, False)
TDMASolverN = TDMA(SN.quad, True)

# The wave vector alpha in Helmholtz equation for NSE
alfa = K[1, 0]**2+K[2, 0]**2-2.0/nu/dt
# The wave vector alpha in Helmholtz equation for MHD
alpha = K[1, 0]**2+K[2, 0]**2-2.0/eta/dt

# Shen coefficients phi_j = T_j + a_j*T_{j+1} + b_j*T_{j+2}
a_j, b_j = SR.shenCoefficients(K[0,:-2,0,0],BC)
cj = SR.chebNormalizationFactor(N, SR.quad)

# Diagonal elements of the matrix B_hat = (phi_j,phi_k_breve)
a0_hat = pi/2*(cj - b_j)
b0_hat = pi/2*b_j
c0_hat = ones(N[0]-2)*(-pi/2)

# 3. Matrices from the Neumann basis functions: (phi^breve_j, phi^breve_k)
A_breve = SFTc.A_mat(K[0, :, 0, 0], a_j, b_j, a_j, b_j, A_breve)
B_breve = SFTc.B_mat(K[0, :, 0, 0], cj, a_j, b_j, a_j, b_j, B_breve) 
C_breve = SFTc.C_mat(K[0, :, 0, 0], a_j, b_j, a_j, b_j, C_breve)

Chm = CDNmat(K[0, :, 0, 0])
Bhm = BNDmat(K[0, :, 0, 0], SN.quad)
Cm = CDDmat(K[0, :, 0, 0])


def pressuregrad(P_hat, dU):
    # Pressure gradient x-direction
    dU[0] -= Chm.matvec(P_hat)
    
    # pressure gradient y-direction
    F_tmp[0] = Bhm.matvec(P_hat)
    dU[1, :Nu] -= 1j*K[1, :Nu]*F_tmp[0, :Nu]
    
    # pressure gradient z-direction
    dU[2, :Nu] -= 1j*K[2, :Nu]*F_tmp[0, :Nu]    
    
    return dU

def pressurerhs(U_hat, dU):
    dU[6] = 0.
    SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], U_hat[0, u_slice], U_hat[1, u_slice], U_hat[2, u_slice], dU[6, p_slice])    
    dU[6, p_slice] *= -1./dt    
    return dU

def body_force(Sk, dU):
    dU[0, :Nu] -= Sk[0, :Nu]
    dU[1, :Nu] -= Sk[1, :Nu]
    dU[2, :Nu] -= Sk[2, :Nu]
    return dU
    
def standardConvection(c):
    """
    (U*grad)U:
    The convective term in the momentum equation:
    x-component: (U*grad)u --> u*dudx + v*dudy + w*dudz
    y-component: (U*grad)v --> u*dvdx + v*dvdy + w*dvdz
    z-component: (U*grad)w --> u*dwdx + v*dwdy + w*dwdz
    
    From continuity equation dudx has Dirichlet bcs.
    dvdx and dwdx are expressed in Chebyshev basis.
    The rest of the terms in grad(u) are expressed in
    shen Dirichlet basis.
    """
    c[:] = 0
    U_tmp4[:] = 0
    U_tmp3[:] = 0

    F_tmp[0] = Cm.matvec(U_hat0[0])
    F_tmp[0] = TDMASolverD(F_tmp[0])    
    dudx = U_tmp4[0] = FST.ifst(F_tmp[0], U_tmp4[0], ST)        
    
    SFTc.Mult_CTD_3D(N[0], U_hat0[1], U_hat0[2], F_tmp[1], F_tmp[2])
    dvdx = U_tmp4[1] = FST.ifct(F_tmp[1], U_tmp4[1],ST)
    dwdx = U_tmp4[2] = FST.ifct(F_tmp[2], U_tmp4[2],ST)
       
    dudy_h = 1j*K[1]*U_hat0[0]
    dudy = U_tmp3[0] = FST.ifst(dudy_h, U_tmp3[0], ST)
    dudz_h = 1j*K[2]*U_hat0[0]
    dudz = U_tmp3[1] = FST.ifst(dudz_h, U_tmp3[1], ST)
    c[0] = FST.fss(U0[0]*dudx + U0[1]*dudy + U0[2]*dudz, c[0], ST)
    
    U_tmp3[:] = 0
    dvdy_h = 1j*K[1]*U_hat0[1]
    dvdy = U_tmp3[0] = FST.ifst(dvdy_h, U_tmp3[0], ST)
    dvdz_h = 1j*K[2]*U_hat0[1]
    dvdz = U_tmp3[1] = FST.ifst(dvdz_h, U_tmp3[1], ST)
    c[1] = FST.fss(U0[0]*dvdx + U0[1]*dvdy + U0[2]*dvdz, c[1], ST)
    
    U_tmp3[:] = 0
    dwdy_h = 1j*K[1]*U_hat0[2]
    dwdy = U_tmp3[0] = FST.ifst(dwdy_h, U_tmp3[0], ST)
    dwdz_h = 1j*K[2]*U_hat0[2]
    dwdz = U_tmp3[1] = FST.ifst(dwdz_h, U_tmp3[1], ST)
    c[2] = FST.fss(U0[0]*dwdx + U0[1]*dwdy + U0[2]*dwdz, c[2], ST)

    c *= -1
    return c

def magneticConvection(c):
    """
    The magentic convection in the momentum equation:
    x-component: (B*grad)bx --> bx*dbxdx + by*dbxdy + bz*dbxdz
    y-component: (B*grad)by --> bx*dbydx + by*dbydy + bz*dbydz
    z-component: (B*grad)bz --> bx*dbzdx + by*dbzdy + bz*dbzdz

    The magnetic field has Neumann bcs. Therefore, \frac{dB_i}{dx}, where i in [x,y,z], 
    must have Dirichlet bcs.
   
    The rest of the terms in grad(u) are esxpressed in
    shen Neumann basis.
    """
    c[:] = 0
    U_tmp4[:] = 0
    F_tmp[:] = 0
    F_tmp2[:] = 0

    F_tmp[0] = SFTc.C_matvecNeumann(K[0,:,0,0], C_breve, U_hat0[3], F_tmp[0])
    F_tmp[1] = SFTc.C_matvecNeumann(K[0,:,0,0], C_breve, U_hat0[4], F_tmp[1])
    F_tmp[2] = SFTc.C_matvecNeumann(K[0,:,0,0], C_breve, U_hat0[5], F_tmp[2])
    
    F_tmp2[0] = SFTc.TDMA(a0_hat, b0_hat, c0_hat, F_tmp[0])
    F_tmp2[1] = SFTc.TDMA(a0_hat, b0_hat, c0_hat, F_tmp[1])  
    F_tmp2[2] = SFTc.TDMA(a0_hat, b0_hat, c0_hat, F_tmp[2]) 
    
    dudx = U_tmp4[0] = FST.ifst(F_tmp2[0], U_tmp4[0], ST)        
    dvdx = U_tmp4[1] = FST.ifst(F_tmp2[1], U_tmp4[1], ST)
    dwdx = U_tmp4[2] = FST.ifst(F_tmp2[2], U_tmp4[2], ST)
    
    U_tmp3[:] = 0
    dudy_h = 1j*K[1]*U_hat0[3]
    dudy = U_tmp3[0] = FST.ifst(dudy_h, U_tmp3[0], SN)
    dudz_h = 1j*K[2]*U_hat0[3]
    dudz = U_tmp3[1] = FST.ifst(dudz_h, U_tmp3[1], SN)
    c[0] = FST.fss(U0[3]*dudx + U0[4]*dudy + U0[5]*dudz, c[0], ST)
    
    U_tmp3[:] = 0
    dvdy_h = 1j*K[1]*U_hat0[4]
    dvdy = U_tmp3[0] = FST.ifst(dvdy_h, U_tmp3[0], SN)
    dvdz_h = 1j*K[2]*U_hat0[4]
    dvdz = U_tmp3[1] = FST.ifst(dvdz_h, U_tmp3[1], SN)
    c[1] = FST.fss(U0[3]*dvdx + U0[4]*dvdy + U0[5]*dvdz, c[1], ST)
    
    U_tmp3[:] = 0
    dwdy_h = 1j*K[1]*U_hat0[5]
    dwdy = U_tmp3[0] = FST.ifst(dwdy_h, U_tmp3[0], SN)
    dwdz_h = 1j*K[2]*U_hat0[5]
    dwdz = U_tmp3[1] = FST.ifst(dwdz_h, U_tmp3[1], SN)
    c[2] = FST.fss(U0[3]*dwdx + U0[4]*dwdy + U0[5]*dwdz, c[2], ST)
   
    return c

def magVelConvection(c):
    """ 
    (B*grad)U:
    The first convection term in the induction equation:
    x-component: (B*grad)u --> bx*dudx + by*dudy + bz*dudz
    y-component: (B*grad)v --> bx*dvdx + by*dvdy + bz*dvdz
    z-component: (B*grad)w --> bx*dwdx + by*dwdy + bz*dwdz

    From continuity equation dudx has Dirichlet bcs.
    dvdx and dwdx are expressed in Chebyshev basis.
    The rest of the terms in grad(u) are expressed in
    shen Dirichlet basis.
    
    NB! Since (B*grad)U-term is part of the induction
    equation, the fss is taking with respect to 
    Shen Neumann basis (SN).
    """
    c[:] = 0
    U_tmp4[:] = 0
    U_tmp3[:] = 0
    
    F_tmp[0] = Cm.matvec(U_hat[0])
    F_tmp[0] = TDMASolverD(F_tmp[0])    
    #F_tmp[0, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, F_tmp[0, u_slice])    
    dudx = U_tmp4[0] = FST.ifst(F_tmp[0], U_tmp4[0], ST)        
    
    SFTc.Mult_CTD_3D(N[0], U_hat0[1], U_hat0[2], F_tmp[1], F_tmp[2])
    dvdx = U_tmp4[1] = FST.ifct(F_tmp[1], U_tmp4[1],ST)
    dwdx = U_tmp4[2] =FST. ifct(F_tmp[2], U_tmp4[2],ST)
       
    dudy_h = 1j*K[1]*U_hat[0]
    dudy = U_tmp3[0] = FST.ifst(dudy_h, U_tmp3[0], ST)
    dudz_h = 1j*K[2]*U_hat[0]
    dudz = U_tmp3[1] = FST.ifst(dudz_h, U_tmp3[1], ST)
    c[0] = FST.fss(U0[3]*dudx + U0[4]*dudy + U0[5]*dudz, c[0], SN)
    
    U_tmp3[:] = 0
    dvdy_h = 1j*K[1]*U_hat[1]
    dvdy = U_tmp3[0] = FST.ifst(dvdy_h, U_tmp3[0], ST)
    dvdz_h = 1j*K[2]*U_hat[1]
    dvdz = U_tmp3[1] = FST.ifst(dvdz_h, U_tmp3[1], ST)
    c[1] = FST.fss(U0[3]*dvdx + U0[4]*dvdy + U0[5]*dvdz, c[1], SN)
    
    U_tmp3[:] = 0
    dwdy_h = 1j*K[1]*U_hat[2]
    dwdy = U_tmp3[0] = FST.ifst(dwdy_h, U_tmp3[0], ST)
    dwdz_h = 1j*K[2]*U_hat[2]
    dwdz = U_tmp3[1] = FST.ifst(dwdz_h, U_tmp3[1], ST)
    c[2] = FST.fss(U0[3]*dwdx + U0[4]*dwdy + U0[5]*dwdz, c[2], SN)
   
    return c
  
def velMagConvection(c):
    """    
    (U*grad)B:
    The second convection term in the induction equation:
    x-component: (U*grad)bx --> u*dbxdx + v*dbxdy + w*dbxdz
    y-component: (U*grad)by --> u*dbydx + v*dbydy + w*dbydz
    z-component: (U*grad)bz --> u*dbzdx + v*dbzdy + w*dbzdz

    The magnetic field has Neumann bcs. Therefore, \frac{dB_i}{dx}, where i in [x,y,z], 
    must have Dirichlet bcs.
   
    The rest of the terms in grad(u) are esxpressed in
    shen Neumann basis.
    
    NB! Since (U*grad)B-term is part of the induction
    equation, the fss is taking with respect to 
    Shen Neumann basis (SN).
    """
    c[:]      = 0
    U_tmp4[:] = 0
    F_tmp[:]  = 0
    F_tmp2[:] = 0
    
    F_tmp[0] = SFTc.C_matvecNeumann(K[0,:,0,0], C_breve, U_hat0[3], F_tmp[0])
    F_tmp[1] = SFTc.C_matvecNeumann(K[0,:,0,0], C_breve, U_hat0[4], F_tmp[1])
    F_tmp[2] = SFTc.C_matvecNeumann(K[0,:,0,0], C_breve, U_hat0[5], F_tmp[2])
    
    F_tmp2[0] = SFTc.TDMA(a0_hat, b0_hat, c0_hat, F_tmp[0])
    F_tmp2[1] = SFTc.TDMA(a0_hat, b0_hat, c0_hat, F_tmp[1])  
    F_tmp2[2] = SFTc.TDMA(a0_hat, b0_hat, c0_hat, F_tmp[2])  
    
    dudx = U_tmp4[0] = FST.ifst(F_tmp2[0], U_tmp4[0], ST)        
    dvdx = U_tmp4[1] = FST.ifst(F_tmp2[1], U_tmp4[1], ST)
    dwdx = U_tmp4[2] = FST.ifst(F_tmp2[2], U_tmp4[2], ST)
    
    U_tmp3[:] = 0
    dudy_h = 1j*K[1]*U_hat0[3]
    dudy = U_tmp3[0] = FST.ifst(dudy_h, U_tmp3[0], SN)
    dudz_h = 1j*K[2]*U_hat0[3]
    dudz = U_tmp3[1] = FST.ifst(dudz_h, U_tmp3[1], SN)
    c[0] = FST.fss(U[0]*dudx + U[1]*dudy + U[2]*dudz, c[0], SN)
    
    U_tmp3[:] = 0
    dvdy_h = 1j*K[1]*U_hat0[4]
    dvdy = U_tmp3[0] = FST.ifst(dvdy_h, U_tmp3[0], SN)
    dvdz_h = 1j*K[2]*U_hat0[4]
    dvdz = U_tmp3[1] = FST.ifst(dvdz_h, U_tmp3[1], SN)
    c[1] = FST.fss(U[0]*dvdx + U[1]*dvdy + U[2]*dvdz, c[1], SN)
    
    U_tmp3[:] = 0
    dwdy_h = 1j*K[1]*U_hat0[5]
    dwdy = U_tmp3[0] = FST.ifst(dwdy_h, U_tmp3[0], SN)
    dwdz_h = 1j*K[2]*U_hat0[5]
    dwdz = U_tmp3[1] = FST.ifst(dwdz_h, U_tmp3[1], SN)
    c[2] = FST.fss(U[0]*dwdx + U[1]*dwdy + U[2]*dwdz, c[2], SN)
   
    return c
  
  
def ComputeRHS_U(dU, jj):
    """
    The rhs of the momentum equation:
    dU = -convection - (pressure gradient) + diffusion + (magnetic convection) + body_force
    """
    if jj == 0:
        conv0[:] = standardConvection(conv0) 
        magconv[:] = magneticConvection(magconv)
        # Compute diffusion
        diff0[:] = 0
        SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GC", -1, alfa, U_hat0[0], diff0[0])
        SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GC", -1, alfa, U_hat0[1], diff0[1])
        SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GC", -1, alfa, U_hat0[2], diff0[2])    
    
    dU[:3] = 1.5*conv0 - 0.5*conv1
    dU[:3] += magconv
    dU[:3] *= dealias    
  
    # Add pressure gradient and body force
    dU = pressuregrad(P_hat, dU)
    dU = body_force(Sk, dU)
    
    # Scale by 2/nu factor
    dU[:3] *= 2./nu
    
    dU[:3] += diff0
        
    return dU

def ComputeRHS_B(dU, jj):
    """
    The rhs of the induction equation:
    dU = -(U*grad)B + (B*grad)U + (1/Rm)*(grad**2)B 
    """
    if jj == 0:  
        magconv[:]  = velMagConvection(magconv)
        magconvU[:] = magVelConvection(magconvU)
        # Compute magnetic diffusion
        diff0[:] = 0   
        diff0[0] = SFTc.Helmholtz_AB_vectorNeumann(K[0,:,0,0], A_breve, B_breve, alpha, U_hat0[3], diff0[0])
        diff0[1] = SFTc.Helmholtz_AB_vectorNeumann(K[0,:,0,0], A_breve, B_breve, alpha, U_hat0[4], diff0[1])
        diff0[2] = SFTc.Helmholtz_AB_vectorNeumann(K[0,:,0,0], A_breve, B_breve, alpha, U_hat0[5], diff0[2])  
        
    dU[3:6] = magconvU - magconv
    dU[3:6] *= dealias    
    
    # Scale by 2/eta factor
    dU[3:6] *= 2./eta
    
    dU[3:6] += diff0
        
    return dU

def regression_test(**kw):
    pass


def solve():
    global dU, P

    timer = Timer() 
    
    while config.t < config.T-1e-8:
        config.t += dt
        config.tstep += 1
        # Tentative momentum solve
        #****************************************************
        #           (I) Mechanincal phase 
        #****************************************************
        # Iterations for magnetic field
        for ii in range(1):
            # Iterations for pressure correction
            for jj in range(config.velocity_pressure_iters):
                dU[:] = 0
                dU = ComputeRHS_U(dU, jj)    
                U_hat[0] = HelmholtzSolverU(U_hat[0],dU[0])
                U_hat[1] = HelmholtzSolverU(U_hat[1],dU[1])
                U_hat[2] = HelmholtzSolverU(U_hat[2],dU[2])
                
                # Pressure correction
                dU = pressurerhs(U_hat, dU) 
                Pcorr[:] = HelmholtzSolverP(Pcorr[:],dU[6])

                P_hat[p_slice] += Pcorr[p_slice]

                #if jj == 0:
                    #print "   Divergence error"
                #print "         Pressure correction norm %2.6e" %(linalg.norm(Pcorr))
                        
            # Update velocity
            dU[:] = 0
            pressuregrad(Pcorr, dU)
            
            dU[0] = TDMASolverD(dU[0])
            dU[1] = TDMASolverD(dU[1])
            dU[2] = TDMASolverD(dU[2])   
            U_hat[:3, u_slice] += dt*dU[:3, u_slice]

            for i in range(3):
                U[i] = FST.ifst(U_hat[i], U[i], ST)
            
            # Rotate velocities
            U_hat1[:3] = U_hat0[:3]
            U_hat0[:3] = U_hat[:3]
            U0[:3] = U[:3]
            
            P = FST.ifst(P_hat, P, SN)        
            conv1[:] = conv0 
            #******************************************
            #        (II) Magnetic phase 
            #******************************************
            dU[:] = 0
            dU = ComputeRHS_B(dU, jj) 
            U_hat[3] = HelmholtzSolverB(U_hat[3],dU[3])
            U_hat[4] = HelmholtzSolverB(U_hat[4],dU[4])
            U_hat[5] = HelmholtzSolverB(U_hat[5],dU[5])

            for i in range(3,6):
                U[i] = FST.ifst(U_hat[i], U[i], SN)

            U_hat0[3:] = U_hat[3:]
            U0[3:] = U[3:]

        update(**globals())
        
        timer()
        
        if config.tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
            
    timer.final(MPI, rank)
    
    if config.make_profile:
        results = create_profile(**globals())
                
    regression_test(**globals())

    hdf5file.close()
