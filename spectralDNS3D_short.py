__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-01-02"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2, rfftn, irfftn
from mpi4py import MPI
from mpi.wrappyfftw import *

nu = 0.000625
T = 0.1
dt = 0.01
M = 4
N = 2**M
L = 2 * pi
dx = L / N
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = N / num_processes     
X = mgrid[rank*Np:(rank+1)*Np, :N, :N].astype(float)*L/N
Nf = N/2+1
U     = empty((3, Np, N, N))                  
U_hat = empty((3, N, Np, Nf), dtype="complex")
P     = empty((Np, N, N))
P_hat = empty((N, Np, Nf), dtype="complex")
U_hat0  = empty((3, N, Np, Nf), dtype="complex")
U_hat1  = empty((3, N, Np, Nf), dtype="complex")
dU      = empty((3, N, Np, Nf), dtype="complex")
Uc_hat  = empty((N, Np, Nf), dtype="complex")
Uc_hatT = empty((Np, N, Nf), dtype="complex")
U_mpi   = empty((num_processes, Np, Np, Nf), dtype="complex")
curl    = empty((3, Np, N, N))

kx = fftfreq(N, 1./N)
kz = kx[:Nf].copy(); kz[-1] *= -1
KX = array(meshgrid(kx, kx[rank*Np:(rank+1)*Np], kz, indexing='ij'), dtype=int)
KK = sum(KX*KX, 0, dtype=int)
KX_over_Ksq = KX.astype(float) / where(KK==0, 1, KK).astype(float)
kmax = 2./3.*(N/2+1)
dealias = array((abs(KX[0]) < kmax)*(abs(KX[1]) < kmax)*(abs(KX[2]) < kmax), dtype=bool)
a = [1./6., 1./3., 1./3., 1./6.]
b = [0.5, 0.5, 1.]

def ifftn_mpi(fu, u):
    Uc_hat[:] = ifft(fu, axis=0)
    comm.Alltoall([Uc_hat, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    for i in range(num_processes): 
        Uc_hatT[:, i*Np:(i+1)*Np] = U_mpi[i]
    u[:] = irfft2(Uc_hatT, axes=(1,2))
    
def fftn_mpi(u, fu):
    Uc_hatT[:] = rfft2(u, axes=(1,2))
    for i in range(num_processes): 
        U_mpi[i] = Uc_hatT[:, i*Np:(i+1)*Np]
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [fu, MPI.DOUBLE_COMPLEX])  
    fu[:] = fft(fu, axis=0)

def Cross(a, b, c):
    fftn_mpi(a[1]*b[2]-a[2]*b[1], c[0])
    fftn_mpi(a[2]*b[0]-a[0]*b[2], c[1])
    fftn_mpi(a[0]*b[1]-a[1]*b[0], c[2])

def Curl(a, c):
    ifftn_mpi(1j*(KX[0]*a[1]-KX[1]*a[0]), c[2])
    ifftn_mpi(1j*(KX[2]*a[0]-KX[0]*a[2]), c[1])
    ifftn_mpi(1j*(KX[1]*a[2]-KX[2]*a[1]), c[0])
    
def ComputeRHS(dU, rk):
    if rk > 0:
        for i in range(3): ifftn_mpi(U_hat[i], U[i])
    Curl(U_hat, curl)
    Cross(U, curl, dU)
    dU[:] *= dealias
    P_hat[:] = sum(dU*KX_over_Ksq, 0)
    dU[:] -= P_hat*KX    
    dU[:] -= nu*KK*U_hat

U[0] = sin(X[0])*cos(X[1])*cos(X[2])
U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
U[2] = 0 
for i in range(3): fftn_mpi(U[i], U_hat[i])
t = 0.0
tstep = 0
while t < T-1e-8:
    t += dt; tstep += 1
    U_hat1[:] = U_hat0[:] = U_hat
    for rk in range(4):
        ComputeRHS(dU, rk)        
        if rk < 3: U_hat[:] = U_hat0 + b[rk]*dt*dU
        U_hat1[:] += a[rk]*dt*dU  
    U_hat[:] = U_hat1[:]
    for i in range(3): ifftn_mpi(U_hat[i], U[i])
    
kk = comm.reduce(0.5*sum(U*U)*dx*dx*dx/L**3)
if rank == 0:
    print kk
