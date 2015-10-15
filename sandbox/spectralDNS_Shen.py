# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:55:26 2015

@author: Diako Darian
    

The code below employs a projection method to solve the 3D incompressible Navier-Stokes equations. 
The spatial discretization is done by a spectral method using periodic Fourier basis functions 
in y and z directions, and non-periodic Chebyshev polynomials in x direction. 

Time discretization is done by a semi-implicit Crank-Nicolson scheme.

"""

from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2, rfftn, irfftn
from mpi4py import MPI
import matplotlib.pyplot as plt
from shentransform import ShenBasis
from Shen_Matrices import B_matrix, C_matrix, A_matrix, D_matrix 
import SFTc
from OrrSommerfeld_eig import OrrSommerfeld
from scipy.fftpack import dct
import time
import sys

try:
    from cbcdns.fft.wrappyfftw import *
except ImportError:
    pass # Rely on numpy.fft routines

case = "OS"
#case = "MKK"

T = 100
dt = 0.001
M = 6

if case == "OS":
    Re = 8000.
    nu = 1./Re
    N = array([2**M, 2**(M-1), 2])
    L = array([2, 2*pi, 4*pi/3.])

elif case == "MKK":
    nu = 2e-5
    Re = 1./nu
    Re_tau = 180.
    utau = nu * Re_tau
    N = array([2**M, 2**M, 2])
    L = array([2, 4*pi, 4*pi/3.])

dx = (L / N).astype(float)
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = N / num_processes

# Get points and weights for Chebyshev weighted integrals
BC1 = array([1,0,0, 1,0,0])
BC2 = array([0,1,0, 0,1,0])
BC3 = array([0,1,0, 1,0,0])
ST = ShenBasis(BC1, quad="GL")
SN = ShenBasis(BC2, quad="GL")
SR = ShenBasis(BC3, quad="GC")

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

U     = empty((3, Np[0], N[1], N[2]))
U_hat = empty((3, N[0], Np[1], Nf), dtype="complex")
P     = empty((Np[0], N[1], N[2]))
P_hat = empty((N[0], Np[1], Nf), dtype="complex")
Pcorr = empty((N[0], Np[1], Nf), dtype="complex")

U0      = empty((3, Np[0], N[1], N[2]))
U_hat0  = empty((3, N[0], Np[1], Nf), dtype="complex")
U_hat1  = empty((3, N[0], Np[1], Nf), dtype="complex")
UT      = empty((3, N[0], Np[1], N[2]))

U_tmp   = empty((3, Np[0], N[1], N[2]))
U_tmp2  = empty((3, Np[0], N[1], N[2]))
U_tmp3  = empty((3, Np[0], N[1], N[2]))
U_tmp4  = empty((3, Np[0], N[1], N[2]))
F_tmp   = empty((3, N[0], Np[1], Nf), dtype="complex")
F_tmp2  = empty((3, N[0], Np[1], Nf), dtype="complex")

dU      = empty((4, N[0], Np[1], Nf), dtype="complex")
Uc      = empty((Np[0], N[1], N[2]))
Uc2     = empty((Np[0], N[1], N[2]))
Uc_hat  = empty((N[0], Np[1], Nf), dtype="complex")
Uc_hat2 = empty((N[0], Np[1], Nf), dtype="complex")
Uc_hat3 = empty((N[0], Np[1], Nf), dtype="complex")
Uc_hatT = empty((Np[0], N[1], Nf), dtype="complex")
U_mpi   = empty((num_processes, Np[0], Np[1], Nf), dtype="complex")
U_mpi2  = empty((num_processes, Np[0], Np[1], N[2]))

curl    = empty((3, Np[0], N[1], N[2]))
conv0   = empty((3, N[0], Np[1], Nf), dtype="complex")
conv1   = empty((3, N[0], Np[1], Nf), dtype="complex")
diff0   = empty((3, N[0], Np[1], Nf), dtype="complex")
Source  = empty((3, Np[0], N[1], N[2])) 
Sk      = empty((3, N[0], Np[1], Nf), dtype="complex") 

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
#kmax[0] = (N[0]-2)/4.
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
        akk = 1./(N[0]-1)*dct(c, 1, axis=0)
        w = arange(0, N[0], 1, dtype=float)
        w[2:] = 2./(1-w[2:]**2)
        w[0] = 1
        w[1::2] = 0
        return sum(akk*w)*L[1]*L[2]/N[1]/N[2]
    else:
        return 0
    
# Set body_force
Source[:] = 0
if case == "OS":
    Source[1, :] = -2./Re
    
elif case == "MKK":
    Source[1, :] = -utau**2


# Take Shen scalar product of body force
Sk[:] = 0
for i in range(3):
    Sk[i] = fss(Source[i], Sk[i], ST)
    
# The constant factors in the Helmholtz equations
alpha1 = K[1, 0]**2+K[2, 0]**2+2.0/nu/dt
alpha2 = K[1, 0]**2+K[2, 0]**2-2.0/nu/dt
alpha3 = K[1, 0]**2+K[2, 0]**2

# Shen coefficients for the basis functions
a_k, b_k = ST.shenCoefficients(K[0,:-2,0,0],BC1)
a_j, b_j = SN.shenCoefficients(K[0,:-2,0,0],BC2)


# Chebyshev normalization factor
ck = ST.chebNormalizationFactor(N, ST.quad)
cj = SN.chebNormalizationFactor(N, SN.quad)

# The components of non-zero diagonals of the matrix B = (phi_j,phi_k)_w
a0 = zeros(N[0]-2)
b0 = zeros(N[0]-2)
Bm = B_matrix(K[0, :, 0, 0], ST.quad, a_k, b_k, a_k, b_k)
a0[2:], b0[1:], c0, d0, e0 = Bm.diags()

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
    dU[3] = SFTc.Helmholtz_CB_vector(K[0,:,0,0],C_hat, B_hat, K[1,0], K[2,0], U_hat[0], U_hat[1], U_hat[2], dU[3])
    dU[3] *= -1./dt    
    return dU

def body_force(Sk, dU):
    dU[0] -= Sk[0]
    dU[1] -= Sk[1]
    dU[2] -= Sk[2]
    return dU
    
def Cross(a, b, c):
    c[0] = fss(a[1]*b[2]-a[2]*b[1], c[0], ST)
    c[1] = fss(a[2]*b[0]-a[0]*b[2], c[1], ST)
    c[2] = fss(a[0]*b[1]-a[1]*b[0], c[2], ST)
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
    F_tmp2[0] = SFTc.PDMA(a0, b0, c0, d0, e0, F_tmp[0], F_tmp2[0])    
    dudx = U_tmp4[0] = ifst(F_tmp2[0], U_tmp4[0], ST)        
    
    F_tmp[1] = SFTc.C_matvec(K[0,:,0,0],Cmat,U_hat0[1], F_tmp[1])
    F_tmp[2] = SFTc.C_matvec(K[0,:,0,0],Cmat,U_hat0[2], F_tmp[2])
    F_tmp2[1] = SFTc.UTDMA(a_k, b_k, F_tmp[1],F_tmp2[1])  
    F_tmp2[2] = SFTc.UTDMA(a_k, b_k, F_tmp[2], F_tmp2[2])  
    
    dvdx = U_tmp4[1] = ifct(F_tmp2[1], U_tmp4[1])
    dwdx = U_tmp4[2] = ifct(F_tmp2[2], U_tmp4[2])  
    
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


def ComputeRHS(dU, jj):
    # Add convection to rhs
    if jj == 0:
        conv0[:] = standardConvection(conv0) 
        
        # Compute diffusion
        diff0[:] = 0
        diff0[0] = SFTc.Helmholtz_AB_vector(K[0,:,0,0], Amat, Bmat, alpha2, U_hat0[0], diff0[0])
        diff0[1] = SFTc.Helmholtz_AB_vector(K[0,:,0,0], Amat, Bmat, alpha2, U_hat0[1], diff0[1])
        diff0[2] = SFTc.Helmholtz_AB_vector(K[0,:,0,0], Amat, Bmat, alpha2, U_hat0[2], diff0[2])   
    
    dU[:3] = 1.5*conv0 - 0.5*conv1
    dU[:3] *= dealias    
    
    # Add pressure gradient and body force
    dU = pressuregrad(P_hat, dU)
    dU = body_force(Sk, dU)
    
    # Scale by 2/nu factor
    dU[:3] *= 2./nu
    
    dU[:3] -= diff0
        
    return dU

def initOS(OS, U, U_hat, t=0.):
    eps = 1e-4
    F_tmp[:] = 0.0
    for i in range(U.shape[1]):
        x = X[0, i, 0, 0]
        OS.interp(x)
        for j in range(U.shape[2]):
            y = X[1, i, j, 0]
            v = (1-x**2) + eps*dot(OS.f, real(OS.dphidy*exp(1j*(y-OS.eigval*t))))
            u = -eps*dot(OS.f, real(1j*OS.phi*exp(1j*(y-OS.eigval*t))))  
            U[0, i, j, :] = u
            U[1, i, j, :] = v
    U[2] = 0
    for i in range(3):
        U_hat[i] = fst(U[i], U_hat[i], ST)

def OSconv(OS, conv, t=0.):
    """For verification"""
    eps = 1e-4
    for i in range(U_tmp.shape[1]):
        x = X[0, i, 0, 0]
        OS.interp(x)
        for j in range(U_tmp.shape[2]):
            y = X[1, i, j, 0]
            v = (1-x**2) + eps*dot(OS.f, real(OS.dphidy*exp(1j*(y-OS.eigval*t))))
            u = -eps*dot(OS.f, real(1j*OS.phi*exp(1j*(y-OS.eigval*t))))  
            dudx = -eps*dot(OS.f, real(1j*OS.dphidy*exp(1j*(y-OS.eigval*t))))
            dudy = eps*dot(OS.f, real(OS.phi*exp(1j*(y-OS.eigval*t))))
            d2phidy = dot(OS.D2, OS.phi)
            dvdx = -2*x + eps*dot(OS.f, real(d2phidy*exp(1j*(y-OS.eigval*t))))
            dvdy = eps*dot(OS.f, real(1j*OS.dphidy*exp(1j*(y-OS.eigval*t))))
            
            conv[0, i, j, :] = u*dudx+v*dudy
            conv[1, i, j, :] = u*dvdx+v*dvdy
    conv[2] = 0
    return conv


if case == "OS":    
    OS = OrrSommerfeld(Re=Re, N=N[0])
    initOS(OS, U0, U_hat0)
    conv1 = standardConvection(conv1)
    initOS(OS, U, U_hat, t=dt)
    t = dt
    en0 = 0.5*energy(U[0]**2+(U[1]-(1-X[0]**2))**2)
    P[:] = 0
    P_hat = fst(P, P_hat, SN)

elif case == "MKK":
    # Initialize with pertubation ala perturbU (https://github.com/wyldckat/perturbU) for openfoam
    Y = where(X[0]<0, 1+X[0], 1-X[0])
    Um = 20*utau
    U[:] = 0
    U[1] = Um*(Y-0.5*Y**2)
    Xplus = Y*Re_tau
    Yplus = X[1]*Re_tau
    Zplus = X[2]*Re_tau
    duplus = Um*0.25/utau 
    alfaplus = 2*pi/500.
    betaplus = 2*pi/200.
    sigma = 0.00055
    epsilon = Um/200.
    dev = 1+0.3*random.randn(Y.shape[0], Y.shape[1], Y.shape[2])
    dd = utau*duplus/2.0*Xplus/40.*exp(-sigma*Xplus**2+0.5)*cos(betaplus*Zplus)*dev
    U[1] += dd
    U[2] += epsilon*sin(alfaplus*Yplus)*Xplus*exp(-sigma*Xplus**2)*dev
    if rank == 0:
        U[:, 0] = 0
    if rank == num_processes-1:
        U[:, -1] = 0

    for i in range(3):
        U_hat[i] = fst(U[i], U_hat[i], ST)

    P[:] = 0
    P_hat = fst(P, P_hat, SN)


U0[:] = U
U_hat0[:] = U_hat

if case == "OS":
    plt.figure()
    im1 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[0,:,:,0], 100)
    plt.colorbar(im1)
    plt.draw()

    plt.figure()
    im2 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[1,:,:,0] - (1-X[0,:,:,0]**2), 100)
    plt.colorbar(im2)
    plt.draw()

    plt.figure()
    im3 = plt.contourf(X[1,:,:,0], X[0,:,:,0], P[:,:,0], 100)
    plt.colorbar(im3)
    plt.draw()
    
    plt.figure()
    im4 = plt.quiver(X[1, :,:,0], X[0,:,:,0], U[1,:,:,0]-(1-X[0,:,:,0]**2), U[0,:,:,0])
    plt.draw()
    
    plt.pause(1e-6)

elif case == "MKK":
    plt.figure()
    im1 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[0,:,:,0], 100)
    plt.colorbar(im1)
    plt.draw()

    plt.figure()
    im2 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[1,:,:,0], 100)
    plt.colorbar(im2)
    plt.draw()

    plt.figure()
    im3 = plt.contourf(X[1,:,:,0], X[0,:,:,0], P[:,:,0], 100)
    plt.colorbar(im3)
    plt.draw()

    plt.pause(1e-6)
    
en0 = 0.5*energy(U[0]**2+(U[1]-(1-X[0]**2))**2)


t = 0.0
tstep = 0

def steps():
    global t, tstep, en0, dU, U_hat, P_hat, Pcorr, U_hat1, U_hat0, P
    while t < T-1e-8:
        t += dt; tstep += 1
        print "tstep ", tstep
        # Tentative momentum solve
        
        for jj in range(2):
            dU[:] = 0
            dU = ComputeRHS(dU, jj)  
            U_hat[0] = SFTc.Helmholtz_AB_Solver(K[0,:,0,0], alpha1, 0, dU[0], Amat, Bmat, U_hat[0])
            U_hat[1] = SFTc.Helmholtz_AB_Solver(K[0,:,0,0], alpha1, 0, dU[1], Amat, Bmat, U_hat[1])
            U_hat[2] = SFTc.Helmholtz_AB_Solver(K[0,:,0,0], alpha1, 0, dU[2], Amat, Bmat, U_hat[2])
            
            # Pressure correction
            dU = pressurerhs(U_hat, dU) 
            Pcorr[:] = SFTc.Helmholtz_AB_Solver(K[0,:,0,0], alpha3, 1, dU[3], A_breve, B_breve, Pcorr)

            # Update pressure
            P_hat[:] += Pcorr[:]
    
        # Update velocity
        dU[:] = 0
        pressuregrad(Pcorr, dU)
        
        dU[0] = SFTc.PDMA(a0, b0, c0, d0, e0, dU[0], dU[0])
        dU[1] = SFTc.PDMA(a0, b0, c0, d0, e0, dU[1], dU[1])
        dU[2] = SFTc.PDMA(a0, b0, c0, d0, e0, dU[2], dU[2])    
        U_hat[:3, u_slice] += dt*dU[:3, u_slice]  

        for i in range(3):
            U[i] = ifst(U_hat[i], U[i], ST)
            
        # Rotate velocities
        U_hat1[:] = U_hat0
        U_hat0[:] = U_hat
        U0[:] = U
        
        P = ifst(P_hat, P, SN)        
        conv1[:] = conv0    
        if tstep % 1 == 0:   
            if case == "OS":
                pert = (U[1] - (1-X[0]**2))**2 + U[0]**2
                initOS(OS, U_tmp4, U_hat1, t=t)
                e1 = 0.5*energy(pert)
                e2 = 0.5*energy((U_tmp4[1] - (1-X[0]**2))**2 + U_tmp4[0]**2)
                exact = exp(2*imag(OS.eigval)*t)
                if rank == 0:
                    print "Time %2.5f Norms %2.12e %2.12e %2.12e %2.12e" %(t, e1/en0, e2/en0, exp(2*imag(OS.eigval)*t), e1/en0-exact)

            elif case == "MKK":
                en0 = energy(U0[0]**2)
                e1 = energy(U0[2]**2)
                if rank == 0:
                    print "Time %2.5f Energy %2.12e %2.12e " %(t, en0, e1)
                
        #if tstep == 100:
            #Source[2, :] = 0
            #Sk[2] = fss(Source[2], Sk[2], ST)

        if tstep % 100 == 0:
            if case == "OS":
                im1.ax.clear()
                im1.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[1, :, :, 0]-(1-X[0,:,:,0]**2), 100)         
                im1.autoscale()
                im2.ax.clear()
                im2.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[0, :, :, 0], 100) 
                im2.autoscale()
                im3.ax.clear()
                im3.ax.contourf(X[1, :,:,0], X[0, :,:,0], P[:, :, 0], 100) 
                im3.autoscale()
                im4.set_UVC(U[1,:,:,0]-(1-X[0,:,:,0]**2), U[0,:,:,0])
                
            elif case == "MKK":
                im1.ax.clear()
                im1.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[1, :, :, 0], 100)         
                im1.autoscale()
                im2.ax.clear()
                im2.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[0, :, :, 0], 100) 
                im2.autoscale()
                im3.ax.clear()
                im3.ax.contourf(X[1, :,:,0], X[0, :,:,0], P[:, :, 0], 100) 
                im3.autoscale()
                
            plt.pause(1e-6)                
steps()

# ---- Tests of the general Shen transform -------------

if case == "OS":
    pert = (U[1] - (1-X[0]**2))**2 + U[0]**2
    initOS(OS, U_tmp4, U_hat1, t=t)
    e1 = 0.5*energy(pert)
    e2 = 0.5*energy((U_tmp4[1] - (1-X[0]**2))**2 + U_tmp4[0]**2)

    if rank == 0:
        print "Time %2.5f Norms %2.10e %2.10e %2.10e" %(t, e1/en0, e2/en0, exp(2*imag(OS.eigval)*t))
