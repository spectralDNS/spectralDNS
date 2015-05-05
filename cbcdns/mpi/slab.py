__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-12-30"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..fft.wrappyfftw import *
from cbcdns import config
from ..optimization import optimizer

__all__ = ['setup', 'ifftn_mpi', 'fftn_mpi']

@optimizer
def transpose_Uc(Uc_hatT, U_mpi, num_processes, Np, Nf):
    for i in xrange(num_processes): 
        Uc_hatT[:, i*Np:(i+1)*Np] = U_mpi[i]
    return Uc_hatT

@optimizer
def transpose_Umpi(U_mpi, Uc_hatT, num_processes, Np, Nf):
    for i in xrange(num_processes): 
        U_mpi[i] = Uc_hatT[:, i*Np:(i+1)*Np]
    return U_mpi

def setupDNS(comm, float, complex, uint8, mpitype, N, L, array, meshgrid, mgrid,
             sum, where, num_processes, rank, pi, **kwargs):
    
    if not num_processes in [2**i for i in range(config.M[0]+1)]:
        raise IOError("Number of cpus must be in ", [2**i for i in range(config.M+1)])

    # Each cpu gets ownership of Np slices
    Np = N / num_processes     

    # Create the physical mesh
    X = mgrid[rank*Np[0]:(rank+1)*Np[0], :N[1], :N[2]].astype(float)
    X[0] *= L[0]/N[0]; X[1] *= L[1]/N[1]; X[2] *= L[2]/N[2]

    """
    Solution U is real and as such its transform, U_hat = fft(U)(k), 
    is such that fft(U)(k) = conj(fft(U)(N-k)) and thus it is sufficient 
    to store N/2+1 Fourier coefficients in the first transformed direction (y).
    For means of efficient MPI communication, the physical box (N^3) is
    shared by processors along the first direction, whereas the Fourier 
    coefficients are shared along the third direction. The y-direction
    is N/2+1 in Fourier space.
    """

    Nf = N[2]/2+1
    U     = empty((3, Np[0], N[1], N[2]), dtype=float)  
    U_hat = empty((3, N[0], Np[1], Nf), dtype=complex)
    P     = empty((Np[0], N[1], N[2]), dtype=float)
    P_hat = empty((N[0], Np[1], Nf), dtype=complex)

    # Temporal storage arrays (Not required by all temporal integrators)
    U_hat0 = empty((3, N[0], Np[1], Nf), dtype=complex)
    U_hat1 = empty((3, N[0], Np[1], Nf), dtype=complex)
    dU     = empty((3, N[0], Np[1], Nf), dtype=complex)

    # work arrays (Not required by all convection methods)
    U_tmp  = empty((3, Np[0], N[0], N[0]), dtype=float)
    F_tmp  = empty((3, N[0], Np[1], Nf), dtype=complex)
    curl   = empty((3, Np[0], N[1], N[2]), dtype=float)   
    Source = None
    
    init_fft(N, Nf, Np, complex, num_processes, comm, rank, mpitype)
    
    # Set wavenumbers in grid
    kx = fftfreq(N[0], 1./N[0])
    ky = fftfreq(N[1], 1./N[1])[rank*Np[1]:(rank+1)*Np[1]]
    kz = fftfreq(N[2], 1./N[2])[:Nf]
    kz[-1] *= -1
    Lp = 2*pi/L
    K  = array(meshgrid(kx, ky, kz, indexing='ij'), dtype=int)
    K[0] *= Lp[0]; K[1] *= Lp[1]; K[2] *= Lp[2] # scale with physical mesh size. This takes care of mapping the physical domain to a computational cube of size (2pi)**3
    K2 = sum(K*K, 0, dtype=int)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)

    # Filter for dealiasing nonlinear convection
    kmax = 2./3.*(N/2+1)
    dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                    (abs(K[2]) < kmax[2]), dtype=uint8)
    del kwargs
    return locals() # Lazy (need only return what is needed)

def setupMHD(comm, float, complex, uint8, mpitype, N, L, array, meshgrid, mgrid,
             sum, where, num_processes, rank, **kwargs):
    
    if not num_processes in [2**i for i in range(config.M+1)]:
        raise IOError("Number of cpus must be in ", [2**i for i in range(config.M+1)])

    # Each cpu gets ownership of Np slices
    Np = N / num_processes     

    # Create the physical mesh
    X = mgrid[rank*Np:(rank+1)*Np, :N, :N].astype(float)*L/N

    """
    Solution U is real and as such its transform, U_hat = fft(U)(k), 
    is such that fft(U)(k) = conj(fft(U)(N-k)) and thus it is sufficient 
    to store N/2+1 Fourier coefficients in the first transformed direction (y).
    For means of efficient MPI communication, the physical box (N^3) is
    shared by processors along the first direction, whereas the Fourier 
    coefficients are shared along the third direction. The y-direction
    is N/2+1 in Fourier space.
    """

    Nf = N/2+1
    UB     = empty((6, Np, N, N), dtype=float)  
    UB_hat = empty((6, N, Np, Nf), dtype=complex)
    P      = empty((Np, N, N), dtype=float)
    P_hat  = empty((N, Np, Nf), dtype=complex)
    
    # Create views into large data structures
    U     = UB[:3] 
    U_hat = UB_hat[:3]
    B     = UB[3:]
    B_hat = UB_hat[3:]

    # Temporal storage arrays (Not required by all temporal integrators)
    UB_hat0 = empty((6, N, Np, Nf), dtype=complex)
    UB_hat1 = empty((6, N, Np, Nf), dtype=complex)
    dU      = empty((6, N, Np, Nf), dtype=complex)

    # work arrays (Not required by all convection methods)
    U_tmp  = empty((3, Np, N, N), dtype=float)
    F_tmp  = empty((3, 3, N, Np, Nf), dtype=complex)
    curl   = empty((3, Np, N, N), dtype=float)   
    Source = None
    
    init_fft(N, Nf, Np, complex, num_processes, comm, rank, mpitype)
    
    # Set wavenumbers in grid
    kx = fftfreq(N, 1./N).astype(int)
    kz = kx[:Nf].copy(); kz[-1] *= -1
    K  = array(meshgrid(kx, kx[rank*Np:(rank+1)*Np], kz, indexing='ij'), dtype=int)
    K2 = sum(K*K, 0, dtype=int)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)

    # Filter for dealiasing nonlinear convection
    kmax = 2./3.*(N/2+1)
    dealias = array((abs(K[0]) < kmax)*(abs(K[1]) < kmax)*
                    (abs(K[2]) < kmax), dtype=uint8)
    del kwargs
    return locals() # Lazy (need only return what is needed)

setup = {"MHD": setupMHD,
         "NS":  setupDNS,
         "VV":  setupDNS}[config.solver]        

def init_fft(N, Nf, Np, complex, num_processes, comm, rank, mpitype):
    # Initialize MPI work arrays globally
    Uc_hat  = empty((N[0], Np[1], Nf), dtype=complex)
    Uc_hatT = empty((Np[0], N[1], Nf), dtype=complex)
    Uc_send = Uc_hat.reshape((num_processes, Np[0], Np[1], Nf))
    U_mpi   = empty((num_processes, Np[0], Np[0], Nf), dtype=complex)
    globals().update(locals())
    
#@profile    
def ifftn_mpi(fu, u):
    """ifft in three directions using mpi.
    Need to do ifft in reversed order of fft
    """
    if num_processes == 1:
        u = irfftn(fu, axes=(0,1,2))
        return u
    
    # Do first owned direction
    Uc_hat[:] = ifft(fu, axis=0)
        
    if config.communication == 'alltoall':
        # Communicate all values
        comm.Alltoall([Uc_hat, mpitype], [U_mpi, mpitype])
        Uc_hatT[:] = transpose_Uc(Uc_hatT, U_mpi, num_processes, Np[1], Nf)
    
    else:
        for i in xrange(num_processes):
            if not i == rank:
                comm.Sendrecv_replace([Uc_send[i], mpitype], i, 0, i, 0)   
            Uc_hatT[:, i*Np[1]:(i+1)*Np[1]] = Uc_send[i]
        
    # Do last two directions
    u = irfft2(Uc_hatT, axes=(1,2))
    return u

#@profile
def fftn_mpi(u, fu):
    """fft in three directions using mpi
    """
    if num_processes == 1:
        fu = rfftn(u, axes=(0,1,2))
        return fu
    
    if config.communication == 'alltoall':
        # Do 2 ffts in y-z directions on owned data
        Uc_hatT[:] = rfft2(u, axes=(1,2))
        
        # Transform data to align with x-direction  
        U_mpi[:] = transpose_Umpi(U_mpi, Uc_hatT, num_processes, Np[1], Nf)
            
        # Communicate all values
        comm.Alltoall([U_mpi, mpitype], [fu, mpitype])  
    
    else:
        # Communicating intermediate result 
        ft = fu.transpose(1,0,2)
        ft[:] = rfft2(u, axes=(1,2))
        fu_send = fu.reshape((num_processes, Np[1], Np[1], Nf))
        for i in xrange(num_processes):
            if not i == rank:
                comm.Sendrecv_replace([fu_send[i], mpitype], i, 0, i, 0)   
        fu_send[:] = fu_send.transpose(0,2,1,3)
                      
    # Do fft for last direction 
    fu = fft(fu, axis=0)
    return fu
     