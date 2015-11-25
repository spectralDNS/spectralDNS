# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:55:26 2015

@author: Diako Darian
    

The code below employs a projection method to solve the 3D incompressible Navier-Stokes equations. 
The spatial discretization is done by a spectral method using periodic Fourier basis functions 
in y and z directions, and non-periodic Chebyshev polynomials in x direction. 

Time discretization is done by a semi-implicit Crank-Nicolson scheme.

"""


from spectralinit import *
from ..shenGeneralBCs import SFTc
from ..shenGeneralBCs.ShenMatrices import B_matrix
from ..shenGeneralBCs.linalg import *
from ..shenGeneralBCs.shentransform import ShenBasis

assert config.precision == "double"
hdf5file = HDF5Writer(comm, float, {"U":U[0], "V":U[1], "W":U[2], "P":P}, config.solver+".h5", 
                      mesh={"x": points, "xp": pointsp, "y": x1, "z": x2})  


# Get points and weights for Chebyshev weighted integrals
BC1 = array([1,0,0, 1,0,0])
BC2 = array([0,1,0, 0,1,0])
BC3 = array([0,1,0, 1,0,0])
ST = ShenBasis(BC1, quad="GL")
SN = ShenBasis(BC2, quad="GC")
SR = ShenBasis(BC3, quad="GC")

points, weights = ST.points_and_weights(N[0])
pointsp, weightsp = SN.points_and_weights(N[0])

# The constant factors in the Helmholtz equations
alpha1 = K[1, 0]**2+K[2, 0]**2+2.0/nu/dt
alpha2 = K[1, 0]**2+K[2, 0]**2-2.0/nu/dt
alpha3 = K[1, 0]**2+K[2, 0]**2


Amat = zeros((N[0]-2,N[0]-2))
Bmat = zeros((N[0]-2,N[0]-2))
Cmat = zeros((N[0]-2,N[0]-2))

A_tilde = zeros((N[0]-2,N[0]-2))
B_tilde = zeros((N[0]-2,N[0]-2))
C_tilde = zeros((N[0]-2,N[0]-2))

A_breve = zeros((N[0]-2,N[0]-2))
B_breve = zeros((N[0]-2,N[0]-2))
C_breve = zeros((N[0]-2,N[0]-2))

A_hat = zeros((N[0]-2,N[0]-2))
B_hat = zeros((N[0]-2,N[0]-2))
C_hat = zeros((N[0]-2,N[0]-2))


def wavenumbers(N):
        if isinstance(N, tuple):
            if len(N) == 1:
                N = N[0]
        if isinstance(N, int): 
            return arange(N-2).astype(float)
        else:
            kk = mgrid[:N[0]-2, :N[1], :N[2]].astype(float)
            return kk[0]

def chebNormalizationFactor(N, quad):
    if quad == "GC":
	ck = ones(N[0]-2); ck[0] = 2
    elif quad == "GL":
	ck = ones(N[0]-2); ck[0] = 2; ck[-1] = 2
    return ck

def shenCoefficients(k, BC):
    """
    Shen basis functions given by
    phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},
    satisfy the imposed boundary conditions for a unique set of {a_k, b_k}.  
    """
    am = BC[0]; bm = BC[1]; cm = BC[2]
    ap = BC[3]; bp = BC[4]; cp = BC[5]

    detk = 2*am*ap + ((k + 1.)**2 + (k + 2.)**2)*(am*bp - ap*bm) - 2.*bm*bp*(k + 1.)**2*(k + 2.)**2

    Aa = am - bm*(k + 2.)**2; Ab= -ap - bp*(k + 2.)**2  
    Ac = am - bm*(k + 1.)**2; Ad= ap + bp*(k + 1.)**2

    y1 = -ap - bp*k**2 + cp; y2= -am + bm*k**2 + cm/((-1)**k) 

    ak = (1./detk)*(Aa*y1 + Ab*y2)
    bk = (1./detk)*(Ac*y1 + Ad*y2)

    return ak, bk

# Shen coefficients for the basis functions
a_k, b_k = shenCoefficients(K[0,:-2,0,0], BC1)
a_j, b_j = shenCoefficients(K[0,:-2,0,0], BC2)


# Chebyshev normalization factor
ck = chebNormalizationFactor(N, ST.quad)
cj = chebNormalizationFactor(N, SN.quad)

# The components of non-zero diagonals of the matrix B = (phi_j,phi_k)_w
a0 = zeros(N[0]-2)
b0 = zeros(N[0]-2)
Bm = B_matrix(K[0, :, 0, 0], ST.quad, a_k, b_k, a_k, b_k)
a0[2:], b0[1:], c0, d0, uud = Bm.diags()

# Matrices
# 1. Matrices from the same Dirichlet basis:
Amat = SFTc.A_mat(K[0, :, 0, 0], a_k, b_k, a_k, b_k, Amat)
Bmat = SFTc.B_mat(K[0, :, 0, 0], ck, a_k, b_k, a_k, b_k, Bmat)
Cmat = SFTc.C_mat(K[0, :, 0, 0], a_k, b_k, a_k, b_k, Cmat)
# 2. Matrices from the Neumann-Dirichlet basis functions: (phi^breve_j, phi_k)
B_tilde = SFTc.B_mat(K[0, :, 0, 0], cj, a_j, b_j, a_k, b_k, B_tilde)
C_tilde = SFTc.C_mat(K[0, :, 0, 0], a_j, b_j, a_k, b_k, C_tilde)
# 3. Matrices from the Neumann basis functions: (phi^breve_j, phi^breve_k)
A_breve = SFTc.A_mat(K[0, :, 0, 0], a_j, b_j, a_j, b_j, A_breve)
B_breve = SFTc.B_mat(K[0, :, 0, 0], cj, a_j, b_j, a_j, b_j, B_breve) 
# 4. Matrices from the Dirichlet-Neumann basis functions: (phi_j, phi^breve_k)
B_hat = SFTc.B_mat(K[0, :, 0, 0], ck, a_k, b_k, a_j, b_j, B_hat)
C_hat = SFTc.C_mat(K[0, :, 0, 0], a_k, b_k, a_j, b_j, C_hat)


def pressuregrad(P_hat, dU):
    
    F_tmp[:] = 0.0
    F_tmp[0] = SFTc.C_matvec(K[0,:,0,0], C_tilde, P_hat, F_tmp[0])  
    F_tmp[1] = SFTc.B_matvec(K[0,:,0,0], B_tilde, P_hat, F_tmp[1])
    
    # Pressure gradient x-direction
    dU[0] -= F_tmp[0]
    # pressure gradient y-direction
    dU[1] -= 1j*K[1]*F_tmp[1]
    
    # pressure gradient z-direction
    dU[2] -= 1j*K[2]*F_tmp[1]    
    
    return dU

def pressurerhs(U_hat, dU):
    dU[3] = 0.
    dU[3] = SFTc.Helmholtz_CB_matvec(K[0,:,0,0],C_hat, B_hat, K[1,0], K[2,0], U_hat[0], U_hat[1], U_hat[2], dU[3])
    dU[3] *= -1./dt    
    return dU

def body_force(Sk, dU):
    dU[0] -= Sk[0]
    dU[1] -= Sk[1]
    dU[2] -= Sk[2]
    return dU
    
def Cross(a, b, c):
    c[0] = FST.fss(a[1]*b[2]-a[2]*b[1], c[0], ST)
    c[1] = FST.fss(a[2]*b[0]-a[0]*b[2], c[1], ST)
    c[2] = FST.fss(a[0]*b[1]-a[1]*b[0], c[2], ST)
    return c

def Curl(a, c, S):
    F_tmp[:] = 0
    U_tmp[:] = 0
    F_tmp[1] = SFTc.C_matvec(K[0,:,0,0],Cmat,U_hat0[1], F_tmp[1])
    F_tmp[2] = SFTc.C_matvec(K[0,:,0,0],Cmat,U_hat0[2], F_tmp[2])
    F_tmp2[1] = SFTc.UTDMA(a_k, b_k, F_tmp[1],F_tmp2[1])  
    F_tmp2[2] = SFTc.UTDMA(a_k, b_k, F_tmp[2], F_tmp2[2])  
    dvdx = U_tmp4[1] = FST.ifct(F_tmp2[1], U_tmp4[1], ST)
    dwdx = U_tmp4[2] = FST.ifct(F_tmp2[2], U_tmp4[2], ST)  
    
    c[0] = FST.ifst(1j*K[1]*a[2] - 1j*K[2]*a[1], c[0], S)
    c[1] = FST.ifst(1j*K[2]*a[0], c[1], S)
    c[1] -= dwdx
    c[2] = FST.ifst(1j*K[1]*a[0], c[2], S)
    c[2] *= -1.0
    c[2] += dvdx
    return c

def standardConvection(c):
    c[:] = 0
    U_tmp4[:] = 0
    U_tmp3[:] = 0
    F_tmp[:] = 0
    F_tmp2[:] = 0

    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = SFTc.C_matvec(K[0,:,0,0], Cmat,U_hat0[0], F_tmp[0])
    F_tmp2[0] = SFTc.PDMA(a0, b0, c0, d0, uud, F_tmp[0], F_tmp2[0])    
    dudx = U_tmp4[0] = FST.ifst(F_tmp2[0], U_tmp4[0], ST)        
    
    F_tmp[1] = SFTc.C_matvec(K[0,:,0,0],Cmat,U_hat0[1], F_tmp[1])
    F_tmp[2] = SFTc.C_matvec(K[0,:,0,0],Cmat,U_hat0[2], F_tmp[2])
    F_tmp2[1] = SFTc.UTDMA(a_k, b_k, F_tmp[1],F_tmp2[1])  
    F_tmp2[2] = SFTc.UTDMA(a_k, b_k, F_tmp[2], F_tmp2[2])  
    
    dvdx = U_tmp4[1] = FST.ifct(F_tmp2[1], U_tmp4[1], ST)
    dwdx = U_tmp4[2] = FST.ifct(F_tmp2[2], U_tmp4[2], ST)  
    
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

def solvePressure(P_hat, Ni):
    """Solve for pressure if Ni is fst of convection"""
    F_tmp[0] = 0
    SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], Ni[0, u_slice], Ni[1, u_slice], Ni[2, u_slice], F_tmp[0, p_slice])    
    P_hat = HelmholtzSolverP(P_hat, F_tmp[0])
    return P_hat

def ComputeRHS(dU, jj):
    # Add convection to rhs
    if jj == 0:
        conv0[:] = standardConvection(conv0) 
        
        # Compute diffusion
        diff0[:] = 0
        diff0[0] = SFTc.Helmholtz_AB_matvec(K[0,:,0,0], Amat, Bmat, alpha2, U_hat0[0], diff0[0])
        diff0[1] = SFTc.Helmholtz_AB_matvec(K[0,:,0,0], Amat, Bmat, alpha2, U_hat0[1], diff0[1])
        diff0[2] = SFTc.Helmholtz_AB_matvec(K[0,:,0,0], Amat, Bmat, alpha2, U_hat0[2], diff0[2])   
    
    dU[:3] = 1.5*conv0 - 0.5*conv1
    dU[:3] *= dealias    
    
    # Add pressure gradient and body force
    dU = pressuregrad(P_hat, dU)
    dU = body_force(Sk, dU)
    
    # Scale by 2/nu factor
    dU[:3] *= 2./nu
    
    dU[:3] -= diff0
        
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
        for jj in range(config.velocity_pressure_iters):
            dU[:] = 0
            dU[:] = ComputeRHS(dU, jj)  
            U_hat[0] = SFTc.Helmholtz_AB_Solver(K[0,:,0,0], alpha1, 0, dU[0], Amat, Bmat, U_hat[0])
            U_hat[1] = SFTc.Helmholtz_AB_Solver(K[0,:,0,0], alpha1, 0, dU[1], Amat, Bmat, U_hat[1])
            U_hat[2] = SFTc.Helmholtz_AB_Solver(K[0,:,0,0], alpha1, 0, dU[2], Amat, Bmat, U_hat[2])
            
            # Pressure correction
            dU = pressurerhs(U_hat, dU) 
            Pcorr[:] = SFTc.Helmholtz_AB_Solver(K[0,:,0,0], alpha3, 1, dU[3], A_breve, B_breve, Pcorr)

            # Update pressure
            P_hat[:] += Pcorr[:]

            if jj == 0 and config.print_divergence_progress:
                print "   Divergence error"
            if config.print_divergence_progress:
                print "         Pressure correction norm %2.6e" %(linalg.norm(Pcorr))
                     
        # Update velocity
        dU[:] = 0
        pressuregrad(Pcorr, dU)
        
        dU[0] = SFTc.PDMA(a0, b0, c0, d0, uud, dU[0], dU[0])
        dU[1] = SFTc.PDMA(a0, b0, c0, d0, uud, dU[1], dU[1])
        dU[2] = SFTc.PDMA(a0, b0, c0, d0, uud, dU[2], dU[2])    
        U_hat[:3, u_slice] += dt*dU[:3, u_slice]  

        for i in range(3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)
            
        # Rotate velocities
        U_hat1[:] = U_hat0
        U_hat0[:] = U_hat
        U0[:] = U
        
        P = FST.ifst(P_hat, P, SN)        
        conv1[:] = conv0    
        
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
