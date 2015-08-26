__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-01-02"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2, rfftn, irfftn
from mpi4py import MPI
import matplotlib.pyplot as plt
from shentransform import ShenDirichletBasis, ShenNeumannBasis
from Matrices import Chmat, Cmat, Bhmat, Bmat
import SFTc
from OrrSommerfeld_eig import OrrSommerfeld
from scipy.fftpack import dct

try:
    from cbcdns.fft.wrappyfftw import *
except ImportError:
    pass # Rely on numpy.fft routines

Re = 8000.
nu = 1./Re
#nu = 2e-5
#Re = 1./nu
#Re_tau = 180.
#utau = nu * Re_tau
T = 1
dt = 0.001
M = 5
N = array([2**M, 2**(M), 2])
#N = array([2**M, 4, 2])
L = array([2, 2*pi, 2*pi/3.])
dx = (L / N).astype(float)

comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = N / num_processes
# Get points and weights for Chebyshev weighted integrals
ST = ShenDirichletBasis(quad="GC")
SN = ShenNeumannBasis(quad="GL")
points, weights = ST.points_and_weights(N[0])
pointsp, weightsp = SN.points_and_weights(N[0])

x1 = arange(N[1], dtype=float)*L[1]/N[1]
x2 = arange(N[2], dtype=float)*L[2]/N[2]

# Get grid for velocity points
X = array(meshgrid(points[rank*Np[0]:(rank+1)*Np[0]], x1, x2, indexing='ij'), dtype=float)
#Xp = array(meshgrid(pointsp[rank*Np[0]:(rank+1)*Np[0]], x1, x2, indexing='ij'), dtype=float)

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
curl    = empty((3, Np[0], N[1], N[2]))
conv0   = empty((3, N[0], Np[1], Nf), dtype="complex")
conv1   = empty((3, N[0], Np[1], Nf), dtype="complex")
diff0   = empty((3, N[0], Np[1], Nf), dtype="complex")
Source  = empty((3, Np[0], N[1], N[2])) 
Sk      = empty((3, N[0], Np[1], Nf), dtype="complex") 

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
    
    #ww = repeat(w, ak.shape[-2]*ak.shape[-1]).reshape(ak.shape)
    #c = comm.reduce(ww*ak)
    #if rank == 0:
        #return c*L[1]*L[2]/N[1]/N[2]


# Set body_force
Source[:] = 0
#Source[1, :] = -utau**2
Source[1, :] = -2./Re
#Source[2, :] = 0.00001*random.randn(*Source.shape[1:])

# Take Shen scalar product of body force
Sk[:] = 0
Sk[1] = fss(Source[1], Sk[1], ST)
#Sk[2] = fss(Source[2], Sk[2], ST)

alfa = K[1, 0]**2+K[2, 0]**2-2.0/nu/dt
alfa1 = sqrt(K[1, 0]**2+K[2, 0]**2+2.0/nu/dt)
alfa2 = sqrt(K[1, 0]**2+K[2, 0]**2)

Chm = Chmat(K[0, :, 0, 0])
Bhm = Bhmat(K[0, :, 0, 0], SN.quad)
Cm = Cmat(K[0, :, 0, 0])

#@profile
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
    dU[3] = 0.
    SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], U_hat[0, u_slice], U_hat[1, u_slice], U_hat[2, u_slice], dU[3, p_slice])    
    dU[3, p_slice] *= -1./dt    
    return dU

def body_force(Sk, dU):
    dU[0, :Nu] -= Sk[0, :Nu]
    dU[1, :Nu] -= Sk[1, :Nu]
    dU[2, :Nu] -= Sk[2, :Nu]
    return dU
    
def Cross(a, b, c):
    c[0] = fss(a[1]*b[2]-a[2]*b[1], c[0], ST)
    c[1] = fss(a[2]*b[0]-a[0]*b[2], c[1], ST)
    c[2] = fss(a[0]*b[1]-a[1]*b[0], c[2], ST)
    return c

def Curl(u0, uh, c):
    
    #c[2] = chebDerivative_3D(u0[1], c[2])
    SFTc.Mult_DPhidT_3D(N[0], uh[1], uh[2], F_tmp[1], F_tmp[2])
    c[2] = ifct(F_tmp[1], c[2])        
    Uc[:] = ifst(1j*K[1, :Nu]*uh[0, :Nu], Uc, ST)    
    c[2] -= Uc
    
    #F_tmp[0] = -Cm.matvec(uh[2])
    #F_tmp[0, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, F_tmp[0, u_slice])    
    #c[1] = ifst(F_tmp[0], c[1], ST)    
    c[1] = ifct(F_tmp[2], c[1])    
    Uc[:] = ifst(1j*K[2]*uh[0, :Nu], Uc, ST) 
    c[1] += Uc    
    
    c[0] = ifst(1j*(K[1]*uh[2]-K[2]*uh[1]), c[0], ST)
    return c

#@profile
def standardConvection(c):
    c[:] = 0
    U_tmp4[:] = 0
    U_tmp3[:] = 0
    
    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = Cm.matvec(U_hat0[0])
    F_tmp[0, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, F_tmp[0, u_slice])    
    dudx = U_tmp4[0] = ifst(F_tmp[0], U_tmp4[0], ST)        
    
    #SFTc.Mult_DPhidT_3D(N[0], U_hat0[1], U_hat0[2], F_tmp[1], F_tmp[2])
    #dvdx = U_tmp4[1] = ifct(F_tmp[1], U_tmp4[1])
    #dwdx = U_tmp4[2] = ifct(F_tmp[2], U_tmp4[2])
    
    #dudx = U_tmp4[0] = chebDerivative_3D(U0[0], U_tmp4[0])
    dvdx = U_tmp4[1] = chebDerivative_3D(U0[1], U_tmp4[1])
    dwdx = U_tmp4[2] = chebDerivative_3D(U0[2], U_tmp4[2])    
    
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

def divergenceConvection(c, add=False):
    """c_i = div(u_i u_j)"""
    if not add: c.fill(0)
    #duudx = U_tmp[0] = chebDerivative_3D(U[0]*U[0], U_tmp[0])
    #duvdx = U_tmp[1] = chebDerivative_3D(U[0]*U[1], U_tmp[1])
    #duwdx = U_tmp[2] = chebDerivative_3D(U[0]*U[2], U_tmp[2])
    
    F_tmp[0] = fst(U0[0]*U0[0], F_tmp[0], ST)
    F_tmp[1] = fst(U0[0]*U0[1], F_tmp[1], ST)
    F_tmp[2] = fst(U0[0]*U0[2], F_tmp[2], ST)
    
    c[0] += Cm.matvec(F_tmp[0])
    c[1] += Cm.matvec(F_tmp[1])
    c[2] += Cm.matvec(F_tmp[2])
    
    F_tmp2[0] = fss(U0[0]*U0[1], F_tmp2[0], ST)
    F_tmp2[1] = fss(U0[0]*U0[2], F_tmp2[1], ST)    
    c[0] += 1j*K[1]*F_tmp2[0] # duvdy
    c[0] += 1j*K[2]*F_tmp2[1] # duwdz
    
    F_tmp[0] = fss(U0[1]*U0[1], F_tmp[0], ST)
    F_tmp[1] = fss(U0[1]*U0[2], F_tmp[1], ST)
    F_tmp[2] = fss(U0[2]*U0[2], F_tmp[2], ST)
    c[1] += 1j*K[1]*F_tmp[0]  # dvvdy
    c[1] += 1j*K[2]*F_tmp[1]  # dvwdz  
    c[2] += 1j*K[1]*F_tmp[1]  # dvwdy
    c[2] += 1j*K[2]*F_tmp[2]  # dwwdz
    c *= -1
    return c

#@profile
def ComputeRHS(dU, jj):
    # Add convection to rhs
    if jj == 0:
        #conv0[:] = divergenceConvection(conv0) 
        conv0[:] = standardConvection(conv0) 
        
        # Compute diffusion
        diff0[:] = 0
        SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GC", -1, alfa, U_hat0[0], diff0[0])
        SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GC", -1, alfa, U_hat0[1], diff0[1])
        SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GC", -1, alfa, U_hat0[2], diff0[2])    
    
    dU[:3] = 1.5*conv0 - 0.5*conv1
    dU[:3] *= dealias    
    
    # Add pressure gradient and body force
    dU = pressuregrad(P_hat, dU)
    dU = body_force(Sk, dU)
    
    # Scale by 2/nu factor
    dU[:3] *= 2./nu
    
    dU[:3] += diff0
        
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

def initOS(OS, U, U_hat, t=0.):
    eps = 1e-4
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

    
OS = OrrSommerfeld(Re=Re, N=N[0])
initOS(OS, U0, U_hat0)
conv1 = standardConvection(conv1)
initOS(OS, U, U_hat, t=dt)
U0[:] = U
U_hat0[:] = U_hat
t = dt
#conv0 = standardConvection(conv0)

e0 = 0.5*energy(U[0]**2+(U[1]-(1-X[0]**2))**2)

## Compute pressure
##dconv = 0.5*(conv0+conv1)
#dconv = conv1
#SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], 
                 #dconv[0, u_slice], dconv[1, u_slice], dconv[2, u_slice], F_tmp[0, p_slice])

#SFTc.Solve_Helmholtz_3D_complex(N[0], 1, F_tmp[0, p_slice], P_hat[p_slice], u0N, u1N, u2N, LN)
P[:] = 0
P = ifst(P_hat, P, SN)

#U[:] = 0
#U[1] = 1-X[0]**2


# Initialize solution
##U[0] = sin(pi*X[0])*cos(X[1])
##U[1] = sin(pi*X[0])*cos(2*X[1])
##U[2] = sin(pi*X[0])*cos(3*X[2])
##U[0] = cos(pi*X[0])+1
##U[1] = cos(pi*X[0])+1
##U[2] = cos(pi*X[0])+1

#dU[:] = 0+0*1j
#dU = pressurerhs(U_hat, dU)
#dU[3, p_slice] *= (dt)

#SFTc.LU_Helmholtz_3D(N[0], 1, SN.quad=="GC", zeros((K[0].shape[1], K[0].shape[2])), u0N, u1N, u2N, LN)
#SFTc.Solve_Helmholtz_3D_complex(N[0], 1, dU[3, p_slice], Pcorr[p_slice], u0N, u1N, u2N, LN)

##P_hat[p_slice] = SFTc.TDMA_3D_complex(a0N, b0N, bcN, c0N, dU[3, p_slice])
#Uc = ifst(Pcorr[p_slice], Uc, SN)

##P[:] = pi*cos(pi*X[0])*cos(X[1]) - 2.*sin(pi*X[0])*sin(2*X[1]) - 3.*sin(pi*X[0])*sin(3.*X[2])

##P[:] = -pi*sin(pi*X[0])

#U[:] = 0
##U[0] = (1-X[0]**2)*sin(pi*X[0])
#for i in range(3):
    #U_hat[i] = fst(U[i], U_hat[i], ST)

#Pcorr[:] = 0
#Pcorr[:Nu-1] += Cu[0]*U_hat[1, 1:Nu]
#Pcorr[1:Nu]  += Cu[1]*U_hat[1, :Nu-1]
#Pcorr[u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, Pcorr[u_slice])
#Uc = ifst(Pcorr[u_slice], Uc, ST)

#Pcorr[:] = 0
#SFTc.Mult_Div_3D(N[0], zeros((K[0].shape[1], K[0].shape[2])), 
                       #zeros((K[0].shape[1], K[0].shape[2])), 
                       #U_hat[0, u_slice], U_hat[1, u_slice], U_hat[2, u_slice], Pcorr[p_slice])
#Pcorr[p_slice] = SFTc.TDMA_3D_complex(a0N, b0N, bcN, c0N, Pcorr[p_slice])
#Uc = ifst(Pcorr[p_slice], Uc, SN)


#Y = where(X[0]<0, 1+X[0], 1-X[0])

#Um = 20*utau

#U[:] = 0
#U[1] = Um*(Y-0.5*Y**2)
#Xplus = Y*Re_tau
#Yplus = X[1]*Re_tau
#Zplus = X[2]*Re_tau

#duplus = Um*0.25/utau 
#alfaplus = 2*pi/500.
#betaplus = 2*pi/200.
#sigma = 0.00055
#epsilon = Um/2000.
#dev = 1+0.01*random.randn(Y.shape[0], Y.shape[1], Y.shape[2])

#dd = utau*duplus/2.0*Xplus/400.*exp(-sigma*Xplus**2+0.5)*cos(betaplus*Zplus)*dev
#U[1] += dd
#U[2] += epsilon*sin(alfaplus*Yplus)*Xplus*exp(-sigma*Xplus**2)*dev

#for i in range(3):
    #U_hat[i] = fst(U[i], U_hat[i], ST)


#P[:] = 0
#P_hat = fst(P, P_hat, SN)

#U0[:] = U
#U_hat0[:] = U_hat

#conv1 = divergenceConvection(conv1)

#curl = OSconv(OS, curl)

#assert allclose(conv0, conv1)

#curl[:] = Curl(U0, U_hat0, curl)
#conv1[:] = Cross(U0, curl, conv1)


## Check convection vs analytical solution
#conv1 *= dealias
#conv1 *= (-1.0)
#conv1[0, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, conv1[0, u_slice])
#conv1[1, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, conv1[1, u_slice])
#U_tmp2[0] = ifst(conv1[0, u_slice], U_tmp2[0], ST)
#U_tmp2[1] = ifst(conv1[1, u_slice], U_tmp2[1], ST)

##plt.figure();plt.imshow(U_tmp2[1, :,:,0]);plt.colorbar()
##plt.figure();plt.imshow(curl[1, :,:,0]);plt.colorbar()
#assert allclose(curl, U_tmp2, atol=1e-7)



## Check diffusion
#dU[:] = 0
#beta = K[1, 0]**2+K[2, 0]**2
#SFTc.Mult_Helmholtz_3D_complex(N[0], ST.quad=="GC", 1, beta, U_hat0[0], dU[0])
#dU[0, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, dU[0, u_slice])
#U_tmp[0] = ifst(dU[0, u_slice], U_tmp[0], ST)
#OSLaplace(OS, U, U_tmp2)

#assert allclose(U_tmp[0], U_tmp2[0], atol=1e-7)


plt.figure()
im1 = plt.imshow(U[0, :, :, 0])
plt.colorbar(im1)
plt.draw()

plt.figure()
im2 = plt.imshow(U[1, :, :, 0]- (1-X[0,:,:,0]**2))
plt.colorbar(im2)
plt.draw()

plt.figure()
im3 = plt.imshow(P[:, :, 0])
plt.colorbar(im3)
plt.draw()

plt.pause(1e-6)

def Divu(U, U_hat, c):
    c[:] = 0
    SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], 
                       U_hat[0, u_slice], U_hat[1, u_slice], U_hat[2, u_slice], c[p_slice])
    c[p_slice] = SFTc.TDMA_3D_complex(a0N, b0N, bcN, c0N, c[p_slice])
        
    return c

e0 = 0.5*energy(U[0]**2+(U[1]-(1-X[0]**2))**2)

t = 0.0
tstep = 0
while t < T-1e-8:
    t += dt; tstep += 1
    #print "tstep ", tstep
    # Tentative momentum solve
    
    for jj in range(2):
        dU[:] = 0
        dU = ComputeRHS(dU, jj)    
        SFTc.Solve_Helmholtz_3D_complex(N[0], 0, dU[0, u_slice], U_hat[0, u_slice], u0, u1, u2, L0)
        SFTc.Solve_Helmholtz_3D_complex(N[0], 0, dU[1, u_slice], U_hat[1, u_slice], u0, u1, u2, L0)
        SFTc.Solve_Helmholtz_3D_complex(N[0], 0, dU[2, u_slice], U_hat[2, u_slice], u0, u1, u2, L0)
        
        # Pressure correction
        dU = pressurerhs(U_hat, dU) 
        SFTc.Solve_Helmholtz_3D_complex(N[0], 1, dU[3, p_slice], Pcorr[p_slice], u0N, u1N, u2N, LN)

        # Update pressure
        #dU[3] *= (-dt)  # Get div(u) in dU[3]
        #dU[3, p_slice] = SFTc.TDMA_3D_complex(a0N, b0N, bcN, c0N, dU[3, p_slice])
        #P_hat[p_slice] += (Pcorr[p_slice] - nu*dU[3, p_slice])
        P_hat[p_slice] += Pcorr[p_slice]

        #for i in range(3):
            #U[i] = ifst(U_hat[i, u_slice], U[i], ST)
        #Uc_hat = Divu(U, U_hat, Uc_hat)
        #Uc = ifst(Uc_hat[p_slice], Uc, SN)
        #if jj == 0:
            #print "   Divergence error"
        #print "         Pressure correction norm %2.6e" %(linalg.norm(Pcorr))
                        
    # Update velocity
    dU[:] = 0
    pressuregrad(Pcorr, dU)
    
    dU[0, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, dU[0, u_slice])
    dU[1, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, dU[1, u_slice])
    dU[2, u_slice] = SFTc.TDMA_3D_complex(a0, b0, bc, c0, dU[2, u_slice])    
    U_hat[:3, u_slice] += dt*dU[:3, u_slice]  # + since pressuregrad computes negative pressure gradient

    for i in range(3):
        U[i] = ifst(U_hat[i], U[i], ST)
        
    # Rotate velocities
    U_hat1[:] = U_hat0
    U_hat0[:] = U_hat
    U0[:] = U
    
    P = ifst(P_hat, P, SN)        
    conv1[:] = conv0
    if tstep % 10 == 0:    
        pert = (U[1] - (1-X[0]**2))**2 + U[0]**2
        initOS(OS, U_tmp4, U_hat1, t=t)
        e1 = 0.5*energy(pert)
        e2 = 0.5*energy((U_tmp4[1] - (1-X[0]**2))**2 + U_tmp4[0]**2)
        exact = exp(2*imag(OS.eigval)*t)
        if rank == 0:
            print "Time %2.5f Norms %2.12e %2.12e %2.12e %2.12e" %(t, e1/e0, e2/e0, exp(2*imag(OS.eigval)*t), e1/e0-exact)
    #if tstep == 100:
        #Source[2, :] = 0
        #Sk[2] = fss(Source[2], Sk[2], ST)
            
    if tstep % 100 == 0:
        im1.set_data(U[0, :, :, 0])
        im1.autoscale()
        im2.set_data(U[1, :, :, 0]-(1-X[0,:,:,0]**2))
        im2.autoscale()
        im3.set_data(P[:, :, 0])
        im3.autoscale()
        plt.pause(1e-6)
        

pert = (U[1] - (1-X[0]**2))**2 + U[0]**2
initOS(OS, U_tmp4, U_hat1, t=t)
e1 = 0.5*energy(pert)
e2 = 0.5*energy((U_tmp4[1] - (1-X[0]**2))**2 + U_tmp4[0]**2)

if rank == 0:
    print "Time %2.5f Norms %2.10e %2.10e %2.10e" %(t, e1/e0, e2/e0, exp(2*imag(OS.eigval)*t))
    
