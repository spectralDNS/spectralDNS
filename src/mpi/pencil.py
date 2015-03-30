__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-12-30"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from wrappyfftw import *
from ..optimization import transform_Uc_xz, transform_Uc_yx, transform_Uc_xy, transform_Uc_zx

__all__ = ['setup', 'ifftn_mpi', 'fftn_mpi']


def setup(comm, float, complex, uint8, mpitype, linspace, N, L, array, meshgrid,
          sum, where, num_processes, rank, P1, arange, MPI, convection, 
          hdf5file, mgrid, **kwargs):

    # Each cpu gets ownership of a pencil of size N1*N2*N in real space
    # and (N1/2+1)*N2*N in Fourier space. However, the Nyquist mode is
    # neglected and as such the actual number is N1/2*N2*N in Fourier space.
    assert num_processes > 1 and P1 < num_processes
    
    P2 = num_processes / P1
    N1 = N/P1
    N2 = N/P2
    
    if not (num_processes % 2 == 0 or num_processes == 1):
        raise IOError("Number of cpus must be even")

    if not ((P1 == 1 or P1 % 2 == 0) and (P2 == 1 or P2 % 2 == 0)):
        raise IOError("Number of cpus in each direction must be even")

    # Create two communicator groups for each rank
    # The goups correspond to chunks in the xy-plane and the xz-plane
    #procxz = arange(num_processes)[rank%P1::P1]
    #procxy = arange(num_processes)[(rank/P1)*P1:(rank/P1+1)*P1]
    #group1 = comm.Get_group()
    #groups xy = MPI.Group.Incl(group1, procxy)
    #commxz = comm.Create(groupxy)
    #group2 = comm.Get_group()
    #groupxz = MPI.Group.Incl(group2, procxz)
    #commxy = comm.Create(groupxz)
    commxz = comm.Split(rank/P1)
    commxy = comm.Split(rank%P1)
    
    xzrank = commxz.Get_rank() # Local rank in xz-plane
    xyrank = commxy.Get_rank() # Local rank in xy-plane
    
    # Create the physical mesh
    #x = linspace(0, L, N+1).astype(float)[:-1]
    x1 = slice(xzrank * N1, (xzrank+1) * N1, 1)
    x2 = slice(xyrank * N2, (xyrank+1) * N2, 1)
    #X = array(meshgrid(x[x1], x[x2], x, indexing='ij'), dtype=float)
    X = mgrid[x1, x2, :N].astype(float)*L/N
    hdf5file.x1 = x1
    hdf5file.x2 = x2

    """
    Solution U is real and as such its transform, U_hat = fft(U)(k), 
    is such that fft(U)(k) = conj(fft(U)(N-k)) and thus it is sufficient 
    to store N/2+1 Fourier coefficients in the first transformed direction 
    (y). However, the Nyquist mode (k=N/2+1) is neglected in the 3D fft.
    The Nyquist mode in included in temporary arrays simply because rfft/irfft 
    expect N/2+1 modes.
    """

    Nf = N/2+1 # Total Fourier coefficients in z-direction
    U     = empty((3, N1, N2, N), dtype=float)
    U_hat = empty((3, N2, N, N1/2), dtype=complex)
    P     = empty((N1, N2, N), dtype=float)
    P_hat = empty((N2, N, N1/2), dtype=complex)

    # Temporal storage arrays (Not required by all temporal integrators)
    U_hat0  = empty((3, N2, N, N1/2), dtype=complex)
    U_hat1  = empty((3, N2, N, N1/2), dtype=complex)
    dU      = empty((3, N2, N, N1/2), dtype=complex)

    init_fft(N1, N2, Nf, N, complex, P1, P2, mpitype, commxz, commxy)    
    
    # work arrays (Not required by all convection methods)
    U_tmp = empty((3, N1, N2, N), dtype=float)
    F_tmp   = empty((3, N2, N, N1/2), dtype=complex)
    curl = empty((3, N1, N2, N), dtype=float)

    # Set wavenumbers in grid
    kx = fftfreq(N, 1./N).astype(int)
    k2 = slice(xyrank*N2, (xyrank+1)*N2, 1)
    k1 = slice(xzrank*N1/2, (xzrank+1)*N1/2, 1)
    K = array(meshgrid(kx[k2], kx, kx[k1], indexing='ij'), dtype=int)
    K2 = sum(K*K, 0, dtype=int)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)

    # Filter for dealiasing nonlinear convection
    kmax = 2./3.*(N/2+1)
    dealias = array((abs(K[0]) < kmax)*(abs(K[1]) < kmax)*
                    (abs(K[2]) < kmax), dtype=uint8)
    del kwargs
    return locals()

def init_fft(N1, N2, Nf, N, complex, P1, P2, mpitype, commxz, commxy):
    # Initialize MPI work arrays globally
    Uc_hat_z  = empty((N1, N2, Nf), dtype=complex)
    Uc_hat_x  = empty((N, N2, N1/2), dtype=complex)
    Uc_hat_xr = empty((N, N2, N1/2), dtype=complex)
    Uc_hat_y  = zeros((N2, N, N1/2), dtype=complex)
    globals().update(locals())

def ifftn_mpi(fu, u):
    """ifft in three directions using mpi.
    Need to do ifft in reversed order of fft
    """
    # Do first owned direction
    Uc_hat_y[:] = ifft(fu, axis=1)

    # Transform to x all but k=N/2 (the neglected Nyquist mode)
    Uc_hat_x[:] = 0
    if transform_Uc_xy:
        transform_Uc_xy(Uc_hat_x, Uc_hat_y, P2, N2)
    else:
        for i in range(P2): 
            Uc_hat_x[i*N2:(i+1)*N2] = Uc_hat_y[:, i*N2:(i+1)*N2]
           
    # Communicate in xz-plane and do fft in x-direction
    commxy.Alltoall([Uc_hat_x, mpitype], [Uc_hat_xr, mpitype])
    Uc_hat_x[:] = ifft(Uc_hat_xr, axis=0)
        
    # Communicate and transform in xy-plane
    commxz.Alltoall([Uc_hat_x, mpitype], [Uc_hat_xr, mpitype])
    if transform_Uc_zx:
        transform_Uc_zx(Uc_hat_z, Uc_hat_xr, P1, N1)
    else:
        for i in range(P1):
            Uc_hat_z[:, :, i*N1/2:(i+1)*N1/2] = Uc_hat_xr[i*N1:(i+1)*N1]
            
    # Do fft for y-direction
    Uc_hat_z[:, :, -1] = 0
    u = irfft(Uc_hat_z, axis=2)
    return u
        
#@profile        
def fftn_mpi(u, fu):
    """fft in three directions using mpi
    """    
    # Do fft in z direction on owned data
    Uc_hat_z[:] = rfft(u, axis=2)
    
    # Transform to x direction neglecting k=N/2 (Nyquist)
    if transform_Uc_xz:
        transform_Uc_xz(Uc_hat_x, Uc_hat_z, P1, N1)
    else:
        for i in range(P1):
            Uc_hat_x[i*N1:(i+1)*N1] = Uc_hat_z[:, :, i*N1/2:(i+1)*N1/2]
    
    # Communicate and do fft in x-direction
    commxz.Alltoall([Uc_hat_x, mpitype], [Uc_hat_xr, mpitype])
    Uc_hat_x[:] = fft(Uc_hat_xr, axis=0)        
    
    # Communicate and transform to final z-direction
    commxy.Alltoall([Uc_hat_x, mpitype], [Uc_hat_xr, mpitype])  
    if transform_Uc_yx:
        transform_Uc_yx(Uc_hat_y, Uc_hat_xr, P2, N2)
    else:
        for i in range(P2): 
            Uc_hat_y[:, i*N2:(i+1)*N2] = Uc_hat_xr[i*N2:(i+1)*N2]
                                   
    # Do fft for last direction 
    fu = fft(Uc_hat_y, axis=1)
    return fu
