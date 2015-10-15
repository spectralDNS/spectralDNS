# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:22:12 2015

@author: Diako Darian


The code below employs a projection method to solve the 3D incompressible MHD equations. 
The spatial discretization is done by a spectral method using periodic Fourier basis functions 
in y and z directions, and non-periodic Chebyshev polynomials in x direction. 

Time discretization is done by a semi-implicit Crank-Nicolson scheme.

"""
from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2, rfftn, irfftn
from mpi4py import MPI
import matplotlib.pyplot as plt
from shentransformDNR import ShenDirichletBasis, ShenNeumannBasis
from shentransform import ShenBasis
from Matrices import Chmat, Cmat, Bhmat, Bmat, BDmat, Amat
import SFTc
from OrrSommerfeld_eig import OrrSommerfeld
from scipy.fftpack import dct
import sys
import time
try:
    from cbcdns.fft.wrappyfftw import *
except ImportError:
    pass # Rely on numpy.fft routines

T = 0.1
dt = 0.001
M = 6

N = array([2**M, 2**(M-1), 2])
L = array([2, 2*pi, 4*pi/3.])

Re = 8000.
Rm = 600.
nu = 1./Re
eta = 1./Rm
B_strength = 0.000001
Ha = B_strength*L[0]*sqrt(Re*Rm)

dx = (L / N).astype(float)
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = N / num_processes

# Get points and weights for Chebyshev weighted integrals
BC = array([0,1,0, 0,1,0])
ST = ShenDirichletBasis(quad="GL")
SN = ShenNeumannBasis(quad="GL")
SR = ShenBasis(BC, quad="GL")

points, weights = ST.points_and_weights(N[0])
pointsp, weightsp = SN.points_and_weights(N[0])

x1 = arange(N[1], dtype=float)*L[1]/N[1]
x2 = arange(N[2], dtype=float)*L[2]/N[2]

# Get grid for velocity points
X = array(meshgrid(points[rank*Np[0]:(rank+1)*Np[0]], x1, x2, indexing='ij'), dtype=float)

Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
Nu = N[0]-2   # Number of velocity modes in Shen basis
Nq = N[0]-3   # Number of pressure modes in Shen basis
u_slice = slice(0, Nu)
p_slice = slice(1, Nu)

U     = empty((6, Np[0], N[1], N[2]))
U_hat = empty((6, N[0], Np[1], Nf), dtype="complex")
P     = empty((Np[0], N[1], N[2]))
P_hat = empty((N[0], Np[1], Nf), dtype="complex")
Pcorr = empty((N[0], Np[1], Nf), dtype="complex")

U0      = empty((6, Np[0], N[1], N[2]))
U_hat0  = empty((6, N[0], Np[1], Nf), dtype="complex")
U_hat1  = empty((6, N[0], Np[1], Nf), dtype="complex")
UT      = empty((6, N[0], Np[1], N[2]))

U_tmp   = empty((6, Np[0], N[1], N[2]))
U_tmp2  = empty((6, Np[0], N[1], N[2]))
U_tmp3  = empty((6, Np[0], N[1], N[2]))
U_tmp4  = empty((6, Np[0], N[1], N[2]))
F_tmp   = empty((6, N[0], Np[1], Nf), dtype="complex")
F_tmp2  = empty((6, N[0], Np[1], Nf), dtype="complex")

dU      = empty((7, N[0], Np[1], Nf), dtype="complex")
Uc      = empty((Np[0], N[1], N[2]))
Uc2     = empty((Np[0], N[1], N[2]))
Uc_hat  = empty((N[0], Np[1], Nf), dtype="complex")
Uc_hat2 = empty((N[0], Np[1], Nf), dtype="complex")
Uc_hat3 = empty((N[0], Np[1], Nf), dtype="complex")
Uc_hatT = empty((Np[0], N[1], Nf), dtype="complex")
U_mpi   = empty((num_processes, Np[0], Np[1], Nf), dtype="complex")
U_mpi2  = empty((num_processes, Np[0], Np[1], N[2]))

curl     = empty((6, Np[0], N[1], N[2]))
conv0    = empty((3, N[0], Np[1], Nf), dtype="complex")
conv1    = empty((3, N[0], Np[1], Nf), dtype="complex")
magconv  = empty((3, N[0], Np[1], Nf), dtype="complex")
magconvU = empty((3, N[0], Np[1], Nf), dtype="complex")
diff0    = empty((3, N[0], Np[1], Nf), dtype="complex")
Source   = empty((6, Np[0], N[1], N[2])) 
Sk       = empty((6, N[0], Np[1], Nf), dtype="complex") 

A_breve = zeros((N[0]-2,N[0]-2))
B_breve = zeros((N[0]-2,N[0]-2))
C_breve = zeros((N[0]-2,N[0]-2))

kx = arange(N[0]).astype(float)
ky = fftfreq(N[1], 1./N[1])[rank*Np[1]:(rank+1)*Np[1]]
kz = fftfreq(N[2], 1./N[2])[:Nf]
kz[-1] *= -1.0

# scale with physical mesh size. 
# This takes care of mapping the physical domain to a computational cube of size (2, 2pi, 2pi)
# Note that first direction cannot be different from 2 (yet)
Lp = array([2, 2*pi, 2*pi])/L
K  = array(meshgrid(kx, ky, kz, indexing='ij'), dtype=float)
K[0] *= Lp[0]; K[1] *= Lp[1]; K[2] *= Lp[2] 
K2 = sum(K*K, 0, dtype=float)
K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)

# Filter for dealiasing nonlinear convection
kmax = 2./3.*(N/2+1)
kmax[0] = N[0]-2
dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                (abs(K[2]) < kmax[2]), dtype=uint8)

def fss(u, fu, S):
    """Fast Shen scalar product of x-direction, Fourier transform of y and z"""
    Uc_hatT[:] = rfft2(u, axes=(1,2))
    n0 = U_mpi.shape[2]
    for i in range(num_processes):
        U_mpi[i] = Uc_hatT[:, i*n0:(i+1)*n0]
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [Uc_hat, MPI.DOUBLE_COMPLEX])
    fu = S.fastShenScalar(Uc_hat, fu)
    return fu

def ifst(fu, u, S):
    """Inverse Shen transform of x-direction, Fourier in y and z"""
    Uc_hat3[:] = S.ifst(fu, Uc_hat3)
    comm.Alltoall([Uc_hat3, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    n0 = U_mpi.shape[2]
    for i in range(num_processes):
        Uc_hatT[:, i*n0:(i+1)*n0] = U_mpi[i]
    u[:] = irfft2(Uc_hatT, axes=(1,2))
    return u

def fst(u, fu, S):
    """Fast Shen transform of x-direction, Fourier transform of y and z"""
    Uc_hatT[:] = rfft2(u, axes=(1,2))
    n0 = U_mpi.shape[2]
    for i in range(num_processes):
        U_mpi[i] = Uc_hatT[:, i*n0:(i+1)*n0]
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [Uc_hat, MPI.DOUBLE_COMPLEX])
    fu = S.fst(Uc_hat, fu)
    return fu

def fct(u, fu):
    """Fast Cheb transform of x-direction, Fourier transform of y and z"""
    Uc_hatT[:] = rfft2(u, axes=(1,2))
    n0 = U_mpi.shape[2]
    for i in range(num_processes):
        U_mpi[i] = Uc_hatT[:, i*n0:(i+1)*n0]
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [Uc_hat, MPI.DOUBLE_COMPLEX])
    fu = ST.fct(Uc_hat, fu)
    return fu

def ifct(fu, u):
    """Inverse Cheb transform of x-direction, Fourier in y and z"""
    Uc_hat3[:] = ST.ifct(fu, Uc_hat3)
    comm.Alltoall([Uc_hat3, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    n0 = U_mpi.shape[2]
    for i in range(num_processes):
        Uc_hatT[:, i*n0:(i+1)*n0] = U_mpi[i]
    u[:] = irfft2(Uc_hatT, axes=(1,2))
    return u

def fct0(u, fu):
    """Fast Cheb transform of x-direction. No FFT, just align data in x-direction and do fct."""
    n0 = U_mpi2.shape[2]
    for i in range(num_processes):
        U_mpi2[i] = u[:, i*n0:(i+1)*n0]
    comm.Alltoall([U_mpi2, MPI.DOUBLE], [UT[0], MPI.DOUBLE])
    fu = ST.fct(UT[0], fu)
    return fu

def ifct0(fu, u):
    """Fast Cheb transform of x-direction. No FFT, just align data in x-direction and do ifct"""
    UT[0] = ST.ifct(fu, UT[0])
    comm.Alltoall([UT[0], MPI.DOUBLE], [U_mpi2, MPI.DOUBLE])
    n0 = U_mpi2.shape[2]
    for i in range(num_processes):
        u[:, i*n0:(i+1)*n0] = U_mpi2[i]
    return u

def chebDerivative_3D0(fj, u0):
    UT[0] = fct0(fj, UT[0])
    UT[1] = SFTc.chebDerivativeCoefficients_3D(UT[0], UT[1]) 
    u0[:] = ifct0(UT[1], u0)
    return u0

def chebDerivative_3D(fj, u0):
    Uc_hat2[:] = fct(fj, Uc_hat2)
    Uc_hat[:] = SFTc.chebDerivativeCoefficients_3D(Uc_hat2, Uc_hat)    
    u0[:] = ifct(Uc_hat, u0)
    return u0

def energy(u):
    uu = sum(u, axis=(1,2))
    c = zeros(N[0])
    comm.Gather(uu, c)
    if rank == 0:
        ak = 1./(N[0]-1)*dct(c, 1, axis=0)
        w = arange(0, N[0], 1, dtype=float)
        w[2:] = 2./(1-w[2:]**2)
        w[0] = 1
        w[1::2] = 0
        return sum(ak*w)*L[1]*L[2]/N[1]/N[2]
    else:
        return 0

# Set body_force
Source[:] = 0
Source[1, :] = -2./Re

# Take Shen scalar product of body force
Sk[:] = 0
for i in range(3):
    Sk[i] = fss(Source[i], Sk[i], ST)

# The wave vector alpha in Helmholtz equation for NSE
alfa = K[1, 0]**2+K[2, 0]**2-2.0/nu/dt
alfa1 = sqrt(K[1, 0]**2+K[2, 0]**2+2.0/nu/dt)
alfa2 = sqrt(K[1, 0]**2+K[2, 0]**2)

# The wave vector alpha in Helmholtz equation for MHD
alpha = K[1, 0]**2+K[2, 0]**2-2.0/eta/dt
alpha1 = sqrt(K[1, 0]**2+K[2, 0]**2+2.0/eta/dt)

# Shen coefficients phi_j = T_j + a_j*T_{j+1} + b_j*T_{j+2}
a_j, b_j = SR.shenCoefficients(K[0,:-2,0,0],BC)
cj = SR.chebNormalizationFactor(N, SR.quad)

# 3. Matrices from the Neumann basis functions: (phi^breve_j, phi^breve_k)
A_breve = SFTc.A_mat(K[0, :, 0, 0], a_j, b_j, a_j, b_j, A_breve)
B_breve = SFTc.B_mat(K[0, :, 0, 0], cj, a_j, b_j, a_j, b_j, B_breve) 
C_breve = SFTc.C_mat(K[0, :, 0, 0], a_j, b_j, a_j, b_j, C_breve)

Chm = Chmat(K[0, :, 0, 0])
Bhm = Bhmat(K[0, :, 0, 0], SN.quad)
Cm = Cmat(K[0, :, 0, 0])

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
    F_tmp[0, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, F_tmp[0, u_slice])    
    dudx = U_tmp4[0] = ifst(F_tmp[0], U_tmp4[0], ST)        
    
    SFTc.Mult_DPhidT_3D(N[0], U_hat0[1], U_hat0[2], F_tmp[1], F_tmp[2])
    dvdx = U_tmp4[1] = ifct(F_tmp[1], U_tmp4[1])
    dwdx = U_tmp4[2] = ifct(F_tmp[2], U_tmp4[2])
       
    dudy_h = 1j*K[1]*U_hat0[0]
    dudy = U_tmp3[0] = ifst(dudy_h, U_tmp3[0], ST)
    dudz_h = 1j*K[2]*U_hat0[0]
    dudz = U_tmp3[1] = ifst(dudz_h, U_tmp3[1], ST)
    c[0] = fss(U0[0]*dudx + U0[1]*dudy + U0[2]*dudz, c[0], ST)
    
    U_tmp3[:] = 0
    dvdy_h = 1j*K[1]*U_hat0[1]
    dvdy = U_tmp3[0] = ifst(dvdy_h, U_tmp3[0], ST)
    dvdz_h = 1j*K[2]*U_hat0[1]
    dvdz = U_tmp3[1] = ifst(dvdz_h, U_tmp3[1], ST)
    c[1] = fss(U0[0]*dvdx + U0[1]*dvdy + U0[2]*dvdz, c[1], ST)
    
    U_tmp3[:] = 0
    dwdy_h = 1j*K[1]*U_hat0[2]
    dwdy = U_tmp3[0] = ifst(dwdy_h, U_tmp3[0], ST)
    dwdz_h = 1j*K[2]*U_hat0[2]
    dwdz = U_tmp3[1] = ifst(dwdz_h, U_tmp3[1], ST)
    c[2] = fss(U0[0]*dwdx + U0[1]*dwdy + U0[2]*dwdz, c[2], ST)

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
    
    dudx = U_tmp4[0] = ifst(F_tmp2[0], U_tmp4[0], ST)        
    dvdx = U_tmp4[1] = ifst(F_tmp2[1], U_tmp4[1], ST)
    dwdx = U_tmp4[2] = ifst(F_tmp2[2], U_tmp4[2], ST)
    
    U_tmp3[:] = 0
    dudy_h = 1j*K[1]*U_hat0[3]
    dudy = U_tmp3[0] = ifst(dudy_h, U_tmp3[0], SN)
    dudz_h = 1j*K[2]*U_hat0[3]
    dudz = U_tmp3[1] = ifst(dudz_h, U_tmp3[1], SN)
    c[0] = fss(U0[3]*dudx + U0[4]*dudy + U0[5]*dudz, c[0], ST)
    
    U_tmp3[:] = 0
    dvdy_h = 1j*K[1]*U_hat0[4]
    dvdy = U_tmp3[0] = ifst(dvdy_h, U_tmp3[0], SN)
    dvdz_h = 1j*K[2]*U_hat0[4]
    dvdz = U_tmp3[1] = ifst(dvdz_h, U_tmp3[1], SN)
    c[1] = fss(U0[3]*dvdx + U0[4]*dvdy + U0[5]*dvdz, c[1], ST)
    
    U_tmp3[:] = 0
    dwdy_h = 1j*K[1]*U_hat0[5]
    dwdy = U_tmp3[0] = ifst(dwdy_h, U_tmp3[0], SN)
    dwdz_h = 1j*K[2]*U_hat0[5]
    dwdz = U_tmp3[1] = ifst(dwdz_h, U_tmp3[1], SN)
    c[2] = fss(U0[3]*dwdx + U0[4]*dwdy + U0[5]*dwdz, c[2], ST)
   
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
    F_tmp[0, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, F_tmp[0, u_slice])    
    dudx = U_tmp4[0] = ifst(F_tmp[0], U_tmp4[0], ST)        
    
    SFTc.Mult_DPhidT_3D(N[0], U_hat[1], U_hat[2], F_tmp[1], F_tmp[2])
    dvdx = U_tmp4[1] = ifct(F_tmp[1], U_tmp4[1])
    dwdx = U_tmp4[2] = ifct(F_tmp[2], U_tmp4[2])
       
    dudy_h = 1j*K[1]*U_hat[0]
    dudy = U_tmp3[0] = ifst(dudy_h, U_tmp3[0], ST)
    dudz_h = 1j*K[2]*U_hat[0]
    dudz = U_tmp3[1] = ifst(dudz_h, U_tmp3[1], ST)
    c[0] = fss(U0[3]*dudx + U0[4]*dudy + U0[5]*dudz, c[0], SN)
    
    U_tmp3[:] = 0
    dvdy_h = 1j*K[1]*U_hat[1]
    dvdy = U_tmp3[0] = ifst(dvdy_h, U_tmp3[0], ST)
    dvdz_h = 1j*K[2]*U_hat[1]
    dvdz = U_tmp3[1] = ifst(dvdz_h, U_tmp3[1], ST)
    c[1] = fss(U0[3]*dvdx + U0[4]*dvdy + U0[5]*dvdz, c[1], SN)
    
    U_tmp3[:] = 0
    dwdy_h = 1j*K[1]*U_hat[2]
    dwdy = U_tmp3[0] = ifst(dwdy_h, U_tmp3[0], ST)
    dwdz_h = 1j*K[2]*U_hat[2]
    dwdz = U_tmp3[1] = ifst(dwdz_h, U_tmp3[1], ST)
    c[2] = fss(U0[3]*dwdx + U0[4]*dwdy + U0[5]*dwdz, c[2], SN)
   
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
    
    dudx = U_tmp4[0] = ifst(F_tmp2[0], U_tmp4[0], ST)        
    dvdx = U_tmp4[1] = ifst(F_tmp2[1], U_tmp4[1], ST)
    dwdx = U_tmp4[2] = ifst(F_tmp2[2], U_tmp4[2], ST)
    
    U_tmp3[:] = 0
    dudy_h = 1j*K[1]*U_hat0[3]
    dudy = U_tmp3[0] = ifst(dudy_h, U_tmp3[0], SN)
    dudz_h = 1j*K[2]*U_hat0[3]
    dudz = U_tmp3[1] = ifst(dudz_h, U_tmp3[1], SN)
    c[0] = fss(U[0]*dudx + U[1]*dudy + U[2]*dudz, c[0], SN)
    
    U_tmp3[:] = 0
    dvdy_h = 1j*K[1]*U_hat0[4]
    dvdy = U_tmp3[0] = ifst(dvdy_h, U_tmp3[0], SN)
    dvdz_h = 1j*K[2]*U_hat0[4]
    dvdz = U_tmp3[1] = ifst(dvdz_h, U_tmp3[1], SN)
    c[1] = fss(U[0]*dvdx + U[1]*dvdy + U[2]*dvdz, c[1], SN)
    
    U_tmp3[:] = 0
    dwdy_h = 1j*K[1]*U_hat0[5]
    dwdy = U_tmp3[0] = ifst(dwdy_h, U_tmp3[0], SN)
    dwdz_h = 1j*K[2]*U_hat0[5]
    dwdz = U_tmp3[1] = ifst(dwdz_h, U_tmp3[1], SN)
    c[2] = fss(U[0]*dwdx + U[1]*dwdy + U[2]*dwdz, c[2], SN)
   
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
  
# Set up for solving with TDMA
if ST.quad == "GL":
    ck = ones(N[0]-2); ck[0] = 2
    
elif ST.quad == "GC":
    ck = ones(N[0]-2); ck[0] = 2; ck[-1] = 2
a0 = ones(N[0]-4)*(-pi/2)
b0 = pi/2*(ck+1)
c0 = a0.copy()
bc = b0.copy()

# For Neumann basis:
kk = SN.wavenumbers(N[0])
ck = ones(N[0]-3)
if SN.quad == "GC": ck[-1] = 2
a0N = ones(N[0]-5)*(-pi/2)*(kk[1:-2]/(kk[1:-2]+2))**2
b0N = pi/2*(1+ck*(kk[1:]/(kk[1:]+2))**4)
c0N = a0N.copy()
bcN = b0N.copy()

a0Neu = pi/2*(cj + b_j**2)
b0Neu = -pi/2*b_j

# Diagonal elements of the matrix B_hat = (phi_j,phi_k_breve)
a0_hat = pi/2*(cj - b_j)
b0_hat = pi/2*b_j
c0_hat = ones(N[0]-2)*(-pi/2)

# Prepare LU Helmholtz solver for velocity
M = (N[0]-3)/2
u0 = zeros((2, M+1, U_hat.shape[2], U_hat.shape[3]))   # Diagonal entries of U
u1 = zeros((2, M, U_hat.shape[2], U_hat.shape[3]))     # Diagonal+1 entries of U
u2 = zeros((2, M-1, U_hat.shape[2], U_hat.shape[3]))   # Diagonal+2 entries of U
L0  = zeros((2, M, U_hat.shape[2], U_hat.shape[3]))     # The single nonzero row of L                 
SFTc.LU_Helmholtz_3D(N[0], 0, ST.quad=="GC", alfa1, u0, u1, u2, L0)

# Prepare LU Helmholtz solver Neumann for pressure
MN = (N[0]-4)/2
u0N = zeros((2, MN+1, U_hat.shape[2], U_hat.shape[3]))   # Diagonal entries of U
u1N = zeros((2, MN, U_hat.shape[2], U_hat.shape[3]))     # Diagonal+1 entries of U
u2N = zeros((2, MN-1, U_hat.shape[2], U_hat.shape[3]))   # Diagonal+2 entries of U
LN  = zeros((2, MN, U_hat.shape[2], U_hat.shape[3]))     # The single nonzero row of L
SFTc.LU_Helmholtz_3D(N[0], 1, SN.quad=="GC", alfa2, u0N, u1N, u2N, LN)

# Prepare LU Helmholtz solver Neumann for magnetic field
MN = (N[0]-4)/2
u0Nb = zeros((2, MN+1, U_hat.shape[2], U_hat.shape[3]))   # Diagonal entries of U
u1Nb = zeros((2, MN, U_hat.shape[2], U_hat.shape[3]))     # Diagonal+1 entries of U
u2Nb = zeros((2, MN-1, U_hat.shape[2], U_hat.shape[3]))   # Diagonal+2 entries of U
LNb  = zeros((2, MN, U_hat.shape[2], U_hat.shape[3]))     # The single nonzero row of L
SFTc.LU_Helmholtz_3D(N[0], 1, SN.quad=="GL", alpha1, u0Nb, u1Nb, u2Nb, LNb)


def initOS(U, U_hat, t=0.):
    for i in range(U.shape[1]):
        x = X[0, i, 0, 0]
        for j in range(U.shape[2]):
            y = X[1, i, j, 0]
            u = 0. 
            v = (cosh(Ha)-cosh(Ha*x))/(cosh(Ha)-1.0)
            Bx = (sinh(Ha*x)-Ha*x*cosh(Ha))/(Ha**2*cosh(Ha))
            By = B_strength
            U[0, i, j, :] = u
            U[1, i, j, :] = v
            U[3, i, j, :] = Bx
            U[4, i, j, :] = By
    U[2] = 0
    U[5] = 0
    for i in range(6):
        if i<3:
            U_hat[i] = fst(U[i], U_hat[i], ST)
        else:
	    U_hat[i] = fst(U[i], U_hat[i], SN)

initOS(U0, U_hat0)
conv1 = standardConvection(conv1)
initOS(U, U_hat, t=dt)
t = dt
    
P[:] = 0
P_hat = fst(P, P_hat, SN)

U0[:] = U
U_hat0[:] = U_hat

u_exact = ( cosh(Ha) - cosh(Ha*X[0,:,0,0]))/(cosh(Ha) - 1.0)

t = 0.0
tstep = 0

def steps():
    global t, tstep, e0, dU, U_hat, P_hat, Pcorr, U_hat1, U_hat0, P
    while t < T-1e-8:
        t += dt; tstep += 1
        #print "tstep ", tstep
        
        #**********************************************************************************************
        #                           (I) Mechanincal phase 
        #**********************************************************************************************
        # Iterations for magnetic field
        for ii in range(1):
	    # Iterations for pressure correction
	    for jj in range(2):
		dU[:] = 0
		dU = ComputeRHS_U(dU, jj)    
		SFTc.Solve_Helmholtz_3D_complex(N[0], 0, dU[0, u_slice], U_hat[0, u_slice], u0, u1, u2, L0)
		SFTc.Solve_Helmholtz_3D_complex(N[0], 0, dU[1, u_slice], U_hat[1, u_slice], u0, u1, u2, L0)
		SFTc.Solve_Helmholtz_3D_complex(N[0], 0, dU[2, u_slice], U_hat[2, u_slice], u0, u1, u2, L0)
		
		# Pressure correction
		dU = pressurerhs(U_hat, dU) 
		SFTc.Solve_Helmholtz_3D_complex(N[0], 1, dU[6, p_slice], Pcorr[p_slice], u0N, u1N, u2N, LN)

		P_hat[p_slice] += Pcorr[p_slice]

		#if jj == 0:
		    #print "   Divergence error"
		#print "         Pressure correction norm %2.6e" %(linalg.norm(Pcorr))
				
	    # Update velocity
	    dU[:] = 0
	    pressuregrad(Pcorr, dU)
	    
	    dU[0, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, dU[0, u_slice])
	    dU[1, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, dU[1, u_slice])
	    dU[2, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, dU[2, u_slice])    
	    U_hat[:3, u_slice] += dt*dU[:3, u_slice]

	    for i in range(3):
		U[i] = ifst(U_hat[i], U[i], ST)
	    
	    # Rotate velocities
	    U_hat1[:3] = U_hat0[:3]
	    U_hat0[:3] = U_hat[:3]
	    U0[:3] = U[:3]
	    
	    P = ifst(P_hat, P, SN)        
	    conv1[:] = conv0 
	    
	    #**********************************************************************************************
	    #                            (II) Magnetic phase 
	    #**********************************************************************************************
	    dU[:] = 0
	    dU = ComputeRHS_B(dU, jj) 
	    SFTc.Solve_Helmholtz_3D_complex(N[0], 1, dU[3, p_slice], U_hat[3,p_slice], u0Nb, u1Nb, u2Nb, LNb)
	    SFTc.Solve_Helmholtz_3D_complex(N[0], 1, dU[4, p_slice], U_hat[4,p_slice], u0Nb, u1Nb, u2Nb, LNb)
	    SFTc.Solve_Helmholtz_3D_complex(N[0], 1, dU[5, p_slice], U_hat[5,p_slice], u0Nb, u1Nb, u2Nb, LNb)
	    
	    #U_hat[3] = SFTc.Helmholtz_AB_Solver(K[0,:,0,0], alpha1**2, 1, dU[3], A_breve, B_breve, U_hat[3])
	    #U_hat[4] = SFTc.Helmholtz_AB_Solver(K[0,:,0,0], alpha1**2, 1, dU[4], A_breve, B_breve, U_hat[4])
	    #U_hat[5] = SFTc.Helmholtz_AB_Solver(K[0,:,0,0], alpha1**2, 1, dU[5], A_breve, B_breve, U_hat[5])
	    for i in range(3,6):
		U[i] = ifst(U_hat[i], U[i], SN)

	    U_hat0[3:] = U_hat[3:]
	    U0[3:] = U[3:]
        
        if tstep % 1 == 0: 
	   print "Time %2.5f Error %2.12e" %(t, linalg.norm(u_exact-U[1,:,0,0],inf))
 
                    
steps()
        
plt.plot(X[0,:,0,0], U[1,:,0,0],X[0,:,0,0],u_exact)
plt.show()