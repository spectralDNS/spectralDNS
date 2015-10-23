# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:20:11 2015

@author: Diako Darian
"""
from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2, rfftn, irfftn
from mpi4py import MPI
import matplotlib.pyplot as plt
from shentransformDNR import ShenNeumannBasis, ShenDirichletBasis
from elsasserShenTransform import ElsasserMinusBasis, ElsasserPlusBasis
import SFTc
from scipy.fftpack import dct
import sys
import time


import scipy.sparse as sps
import scipy.io

try:
    from cbcdns.fft.wrappyfftw import *
except ImportError:
    pass # Rely on numpy.fft routines

T = 0.1
dt = 0.001
M = 4

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
SP = ElsasserPlusBasis(quad="GL")
SM = ElsasserMinusBasis(quad="GL")
SN = ShenNeumannBasis(quad="GL")
ST = ShenDirichletBasis(quad="GL")

points, weights = SP.points_and_weights(N[0])
pointsm, weightsm = SM.points_and_weights(N[0])
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

Amm = zeros((N[0]-2,N[0]-2))
Bmm = zeros((N[0]-2,N[0]-2))
Cmm = zeros((N[0]-2,N[0]-2))

Bmat = zeros((N[0]-2,N[0]-2))


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
    Sk[i] = fss(Source[i], Sk[i], SM)


# Shen coefficients phi_j = T_j + a_j*T_{j+1} + b_j*T_{j+2}
bm_k = SM.shenCoefficients(K[0,:-2,0,0])
bp_k = SP.shenCoefficients(K[0,:-2,0,0])
b_k  = -(K[0,:-2,0,0]/(K[0,:-2,0,0]+2))**2
ck = ones(N[0]-2)
ck[0] = 2
a_j = zeros(N[0]-2)
# 3. Matrices from the Neumann basis functions: (phi^breve_j, phi^breve_k)
Amm = SFTc.B_mat(K[0, :, 0, 0],ck, a_j, b_k, a_j, b_k,Amm)
#Amm = SFTc.Abb_mat(b_k, Amm) 
#Bmm = SFTc.Bbb_mat(ck, b_k, Bmm) 
Cmm = SFTc.Cbb_mat(b_k, Cmm)

#print "Cond:", linalg.cond(Cmm)
#plt.spy(Amm, markersize=2)
#plt.show()
alpha = K[1, 0]**2+K[2, 0]**2

U[0] = (4./3.)*X[0]-(4./9.)*X[0]**3
U_tmp[1] = (4./3.)*(1.0 - X[0]**2)

U_hat[0] = fst(U_tmp[0], U_hat[0], SM)
U_hat[1] = fst(U_tmp[1], U_hat[1], ST)

#F_tmp[0] = SFTc.Amm_matvec(Amm, U_hat[0], F_tmp[0])  

F_tmp[1] = 0.0
F_tmp[0] = 0.0
F_tmp[1] = SFTc.B_matvec(K[0,:,0,0], Amm, U_hat[1], F_tmp[1])
F_tmp[0] = SFTc.Cbb_matvec(Cmm, U_hat[0], F_tmp[0])

allclose(F_tmp[0],F_tmp[1])
sys.exit()
#U_hat[0, (N[0]-3)]  = F_tmp[1, (N[0]-5)]/Amm[(N[0]-5),(N[0]-3)]

#for k in xrange(N[0]-4,1,-1):
    #s = 0.
    #for j in xrange(k+2, N[0]-2, 2):
	#s += Amm[k-2,j]*U_hat[0, j]	
    #U_hat[0, k] = (F_tmp[1, (k-2)] - s)/Amm[k-2,k]
    #print U_hat[0,k,1,1]

U_tmp[0] = 0    
U_tmp[0] = ifst(U_hat[0], U_tmp[0], SM)
print U_tmp[0,:,0,0]

#print linalg.norm(F_tmp[0,:,0,0] - F_tmp[1,:,0,0], inf)

#assert allclose(F_tmp[0,0,0,:], F_tmp[1,0,0,:])

U_tmp[0,:] /= 1.0e288
plt.plot(X[0,:,2,0], U[0,:,2,0],X[0,:,2,0], U_tmp[0,:,2,0])
plt.show()
#sys.exit()
