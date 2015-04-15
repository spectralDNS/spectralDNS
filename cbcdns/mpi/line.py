__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-12-30"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..fft.wrappyfftw import *
from cbcdns import config
from ..optimization import optimizer

__all__ = ['setup', 'ifft2_mpi', 'fft2_mpi']

def setupNS(comm, float, complex, uint8, mpitype, N, L, array, meshgrid, mgrid,
          sum, where, num_processes, rank, conj, **kwargs):
    
    # Each cpu gets ownership of Np indices
    Np = N / num_processes     

    # Create the mesh
    X = mgrid[rank*Np:(rank+1)*Np, :N].astype(float)*L/N

    # Solution array and Fourier coefficients
    # Because of real transforms and symmetries, N/2+1 coefficients are sufficient
    Nf = N/2+1
    Npf = Np/2+1 if rank+1 == num_processes else Np/2

    U     = empty((2, Np, N), dtype=float)
    U_hat = empty((2, N, Npf), dtype=complex)
    P     = empty((Np, N), dtype=float)
    P_hat = empty((N, Npf), dtype=complex)
    curl  = empty((Np, N), dtype=float)
    F_tmp = empty((2, N, Npf), dtype=complex)

    init_fft(N, Nf, Np, Npf, complex, num_processes, comm, rank, mpitype, conj)

    # RK4 arrays
    U_hat0 = empty((2, N, Npf), dtype=complex)
    U_hat1 = empty((2, N, Npf), dtype=complex)
    dU     = empty((2, N, Npf), dtype=complex)

    # Set wavenumbers in grid
    kx = fftfreq(N, 1./N)
    ky = kx[:Nf].copy(); ky[-1] *= -1
    K = array(meshgrid(kx, ky[rank*Np/2:(rank*Np/2+Npf)], indexing='ij'), dtype=int)
    K2 = sum(K*K, 0, dtype=int)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)

    # Filter for dealiasing nonlinear convection
    kmax = 2./3.*(N/2+1)
    dealias = array((abs(K[0]) < kmax)*(abs(K[1]) < kmax), dtype=uint8)
    del kwargs
    return locals()

def setupBoussinesq(comm, float, complex, uint8, mpitype, N, L, array, meshgrid, mgrid,
                    sum, where, num_processes, rank, conj, **kwargs):
    
    # Each cpu gets ownership of Np indices
    Np = N / num_processes     

    # Create the mesh
    X = mgrid[rank*Np:(rank+1)*Np, :N].astype(float)*L/N

    # Solution array and Fourier coefficients
    # Because of real transforms and symmetries, N/2+1 coefficients are sufficient
    Nf = N/2+1
    Npf = Np/2+1 if rank+1 == num_processes else Np/2

    Ur     = empty((3, Np, N), dtype=float)
    Ur_hat = empty((3, N, Npf), dtype=complex)
    P     = empty((Np, N), dtype=float)
    P_hat = empty((N, Npf), dtype=complex)
    curl   = empty((Np, N), dtype=float)
    dU     = empty((3, N, Npf), dtype=complex)
 
    # Create views into large data structures
    rho     = Ur[2]
    rho_hat = Ur_hat[2]
    U     = Ur[:2] 
    U_hat = Ur_hat[:2]

    F_tmp = empty((2, Np, N), dtype=float)
    F_tmp_hat = empty((2, N, Npf), dtype=complex)

    init_fft(N, Nf, Np, Npf, complex, num_processes, comm, rank, mpitype, conj)
    
    # RK4 arrays
    Ur_hat0 = empty((3, N, Npf), dtype=complex)
    Ur_hat1 = empty((3, N, Npf), dtype=complex)
    
    # Set wavenumbers in grid
    kx = fftfreq(N, 1./N)
    ky = kx[:Nf].copy(); ky[-1] *= -1
    K = array(meshgrid(kx, ky[rank*Np/2:(rank*Np/2+Npf)], indexing='ij'), dtype=int)
    K2 = sum(K*K, 0, dtype=int)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)

    # Filter for dealiasing nonlinear convection
    kmax = 2./3.*(N/2+1)
    dealias = array((abs(K[0]) < kmax)*(abs(K[1]) < kmax), dtype=uint8)
    del kwargs
    return locals()

setup = {"NS2D": setupNS,
         "Bq2D": setupBoussinesq}[config.solver]

def init_fft(N, Nf, Np, Npf, complex, num_processes, comm, rank, mpitype, conj):
    # Initialize MPI work arrays globally
    U_recv = empty((N, Np/2), dtype=complex)
    fft_y = empty(N, dtype=complex)
    fft_x = empty(N, dtype=complex)
    plane_recv = empty(Np, dtype=complex)
    Uc_hat = empty((N, Npf), dtype=complex)
    Uc_hatT = empty((Np, Nf), dtype=complex)
    U_send = empty((num_processes, Np, Np/2), dtype=complex)
    U_sendr = U_send.reshape((N, Np/2))
    globals().update(locals())

@optimizer
def transpose_x(U_send, Uc_hatT, num_processes, Np):
    # Align data in x-direction
    for i in range(num_processes): 
        U_send[i] = Uc_hatT[:, i*Np/2:(i+1)*Np/2]
    return U_send

@optimizer
def transpose_y(Uc_hatT, U_recv, num_processes, Np):
    for i in range(num_processes): 
        Uc_hatT[:, i*Np/2:(i+1)*Np/2] = U_recv[i*Np:(i+1)*Np]
    return Uc_hatT

@optimizer
def swap_Nq(fft_y, fu, fft_x, N):
    f = fu[:, 0]        
    fft_x[0] = f[0].real
    fft_x[1:N/2] = 0.5*(f[1:N/2]+conj(f[:N/2:-1]))
    fft_x[N/2] = f[N/2].real        
    fu[:N/2+1, 0] = fft_x[:N/2+1]        
    fu[N/2+1:, 0] = conj(fft_x[(N/2-1):0:-1])
    
    fft_y[0] = f[0].imag
    fft_y[1:N/2] = -0.5*1j*(f[1:N/2]-conj(f[:N/2:-1]))
    fft_y[N/2] = f[N/2].imag
    fft_y[N/2+1:] = conj(fft_y[(N/2-1):0:-1])
    return fft_y
@profile
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
    
    fu[:, :Np/2] = fft(U_recv, axis=0)
        
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
    U_sendr[:] = Uc_hat[:, :Np/2]

    comm.Alltoall([U_send, mpitype], [U_recv, mpitype])

    Uc_hatT = transpose_y(Uc_hatT, U_recv, num_processes, Np)
    
    if rank == num_processes-1:
        fft_y[:] = Uc_hat[:, -1]

    comm.Scatter(fft_y, plane_recv, root=num_processes-1)
    Uc_hatT[:, -1] = plane_recv
    
    u[:] = irfft(Uc_hatT, 1)
    return u
