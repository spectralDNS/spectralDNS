__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-12-30"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..fft.wrappyfftw import *
from cbcdns import config
from ..optimization import optimizer
from numpy import array, sum, meshgrid, mgrid, where, abs, pi, uint8

__all__ = ['setup', 'ifft2_mpi', 'fft2_mpi']

def create_wavenumber_arrays(N, Nf, Np, rank, float):
    # Set wavenumbers in grid
    kx = fftfreq(N[0], 1./N[0])
    ky = fftfreq(N[1], 1./N[1])[:Nf]
    ky[-1] *= -1
    Lp = 2*pi/config.L
    K = array(meshgrid(kx, ky[rank*Np[1]/2:(rank*Np[1]/2+Npf)], indexing='ij'), dtype=float)
    K[0] *= Lp[0]; K[1] *= Lp[1]
    K2 = sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)

    # Filter for dealiasing nonlinear convection
    kmax = 2./3.*(N/2)
    dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1]), dtype=uint8)
    return K, K2, K_over_K2, dealias

def setupNS(comm, float, complex, uint8, mpitype, N, L, array, meshgrid, mgrid,
          sum, where, num_processes, rank, conj, pi, **kwargs):
    
    # Each cpu gets ownership of Np indices
    Np = N / num_processes     

    # Create the physical mesh
    X = mgrid[rank*Np[0]:(rank+1)*Np[0], :N[1]].astype(float)
    X[0] *= L[0]/N[0]; X[1] *= L[1]/N[1]

    # Solution array and Fourier coefficients
    # Because of real transforms and symmetries, N/2+1 coefficients are sufficient
    Nf = N[1]/2+1
    Npf = Np[1]/2+1 if rank+1 == num_processes else Np[1]/2

    U     = empty((2, Np[0], N[1]), dtype=float)
    U_hat = empty((2, N[0], Npf), dtype=complex)
    P     = empty((Np[0], N[1]), dtype=float)
    P_hat = empty((N[0], Npf), dtype=complex)
    curl  = empty((Np[0], N[1]), dtype=float)
    F_tmp = empty((2, N[0], Npf), dtype=complex)
    dU     = empty((2, N[0], Npf), dtype=complex)
    
    init_fft(N, Nf, Np, Npf, complex, num_processes, comm, rank, mpitype, conj)
    
    K, K2, K_over_K2, dealias = create_wavenumber_arrays(N, Nf, Np, rank, float)

    del kwargs
    return locals()

def setupBoussinesq(comm, float, complex, uint8, mpitype, N, L, array, meshgrid, mgrid,
                    sum, where, num_processes, rank, conj, pi, resize, **kwargs):
    
    # Each cpu gets ownership of Np indices
    Np = N / num_processes     

    # Create the mesh
    X = mgrid[rank*Np[0]:(rank+1)*Np[0], :N[1]].astype(float)
    X[0] *= L[0]/N[0]; X[1] *= L[1]/N[1]

    # Solution array and Fourier coefficients
    # Because of real transforms and symmetries, N/2+1 coefficients are sufficient
    Nf = N[1]/2+1
    Npf = Np[1]/2+1 if rank+1 == num_processes else Np[1]/2

    Ur     = empty((3, Np[0], N[1]), dtype=float)
    Ur_hat = empty((3, N[0], Npf), dtype=complex)
    P      = empty((Np[0], N[1]), dtype=float)
    P_hat  = empty((N[0], Npf), dtype=complex)
    curl   = empty((Np[0], N[1]), dtype=float)
    dU     = empty((3, N[0], Npf), dtype=complex)
     
    # Create views into large data structures
    rho     = Ur[2]
    rho_hat = Ur_hat[2]
    U       = Ur[:2] 
    U_hat   = Ur_hat[:2]

    U_tmp = empty((2, Np[0], N[1]), dtype=float)
    F_tmp = empty((2, N[0], Npf), dtype=complex)

    init_fft(N, Nf, Np, Npf, complex, num_processes, comm, rank, mpitype, conj)

    K, K2, K_over_K2, dealias = create_wavenumber_arrays(N, Nf, Np, rank, float)
    
    del kwargs
    return locals()

setup = {"NS2D": setupNS,
         "Bq2D": setupBoussinesq}[config.solver]

def init_fft(N, Nf, Np, Npf, complex, num_processes, comm, rank, mpitype, conj):
    # Initialize MPI work arrays globally
    U_recv = empty((N[0], Np[1]/2), dtype=complex)
    fft_y = empty(N[0], dtype=complex)
    fft_x = empty(N[0], dtype=complex)
    plane_recv = empty(Np[0], dtype=complex)
    Uc_hat = empty((N[0], Npf), dtype=complex)
    Uc_hatT = empty((Np[0], Nf), dtype=complex)
    U_send = empty((num_processes, Np[0], Np[1]/2), dtype=complex)
    U_sendr = U_send.reshape((N[0], Np[1]/2))
    globals().update(locals())

@optimizer
def transpose_x(U_send, Uc_hatT, num_processes, Np):
    # Align data in x-direction
    for i in range(num_processes): 
        U_send[i] = Uc_hatT[:, i*Np[1]/2:(i+1)*Np[1]/2]
    return U_send

@optimizer
def transpose_y(Uc_hatT, U_recv, num_processes, Np):
    for i in range(num_processes): 
        Uc_hatT[:, i*Np[1]/2:(i+1)*Np[1]/2] = U_recv[i*Np[0]:(i+1)*Np[0]]
    return Uc_hatT

@optimizer
def swap_Nq(fft_y, fu, fft_x, N):
    f = fu[:, 0]        
    fft_x[0] = f[0].real
    fft_x[1:N[0]/2] = 0.5*(f[1:N[0]/2]+conj(f[:N[0]/2:-1]))
    fft_x[N[0]/2] = f[N[0]/2].real        
    fu[:N[0]/2+1, 0] = fft_x[:N[0]/2+1]        
    fu[N[0]/2+1:, 0] = conj(fft_x[(N[0]/2-1):0:-1])
    
    fft_y[0] = f[0].imag
    fft_y[1:N[0]/2] = -0.5*1j*(f[1:N[0]/2]-conj(f[:N[0]/2:-1]))
    fft_y[N[0]/2] = f[N[0]/2].imag
    fft_y[N[0]/2+1:] = conj(fft_y[(N[0]/2-1):0:-1])
    return fft_y

def fft2_mpi(u, fu):
    global U_send, fft_y
    if num_processes == 1:
        fu[:] = rfft2(u, axes=(0,1))
        return fu    
    
    Uc_hatT[:] = rfft(u, axis=1)
    Uc_hatT[:, 0] += 1j*Uc_hatT[:, -1]
    
    U_send = transpose_x(U_send, Uc_hatT, num_processes, Np)
            
    # Communicate all values
    comm.Alltoall([U_send, mpitype], [U_recv, mpitype])
    
    fu[:, :Np[1]/2] = fft(U_recv, axis=0)
    
    # Handle Nyquist frequency
    if rank == 0:        
        fft_y = swap_Nq(fft_y, fu, fft_x, N)
        comm.Send([fft_y, mpitype], dest=num_processes-1, tag=77)
        
    elif rank == num_processes-1:
        comm.Recv([fft_y, mpitype], source=0, tag=77)
        fu[:, -1] = fft_y 
        
    return fu

def ifft2_mpi(fu, u):
    global Uc_hatT
    if num_processes == 1:
        u[:] = irfft2(fu, axes=(0,1))
        return u

    Uc_hat[:] = ifft(fu, axis=0)    
    U_sendr[:] = Uc_hat[:, :Np[1]/2]

    comm.Alltoall([U_send, mpitype], [U_recv, mpitype])

    Uc_hatT = transpose_y(Uc_hatT, U_recv, num_processes, Np)
    
    if rank == num_processes-1:
        fft_y[:] = Uc_hat[:, -1]

    comm.Scatter(fft_y, plane_recv, root=num_processes-1)
    Uc_hatT[:, -1] = plane_recv
    
    u[:] = irfft(Uc_hatT, 1)
    return u
