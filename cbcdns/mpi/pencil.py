__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-12-30"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..fft.wrappyfftw import *
from cbcdns import config
from ..optimization import optimizer
from numpy import array, sum, meshgrid, mgrid, where, abs, pi, uint8, rollaxis

#__all__ = ['setup', 'ifftn_mpi', 'fftn_mpi']
__all__ = ['setup']

@optimizer
def transform_Uc_xz(Uc_hat_x, Uc_hat_z, P1):
    #n0 = Uc_hat_z.shape[0]
    #n1 = Uc_hat_x.shape[2]
    #for i in range(P1):
        #Uc_hat_x[i*n0:(i+1)*n0] = Uc_hat_z[:, :, i*n1:(i+1)*n1]
        
    sz = Uc_hat_z.shape
    sx = Uc_hat_x.shape
    Uc_hat_x[:] = rollaxis(Uc_hat_z[:,:,:-1].reshape((sz[0], sz[1], P1, sx[2])), 2).reshape(sx)
    return Uc_hat_x
            
@optimizer
def transform_Uc_zx(Uc_hat_z, Uc_hat_xr, P1):
    #n0 = Uc_hat_z.shape[0]
    #n1 = Uc_hat_xr.shape[2]
    #for i in range(P1):
        #Uc_hat_z[:, :, i*n1:(i+1)*n1] = Uc_hat_xr[i*n0:(i+1)*n0]
        
    sz = Uc_hat_z.shape
    sx = Uc_hat_xr.shape
    Uc_hat_z[:, :, :-1] = rollaxis(Uc_hat_xr.reshape((P1, sz[0], sz[1], sx[2])), 0, 3).reshape((sz[0], sz[1], sz[2]-1))        
    return Uc_hat_z

@optimizer
def transform_Uc_xy(Uc_hat_x, Uc_hat_y, P2):
    #n0 = Uc_hat_y.shape[0]
    #n1 = Uc_hat_x.shape[1]
    #for i in range(P2): 
        #Uc_hat_x[i*n0:(i+1)*n0] = Uc_hat_y[:, i*n1:(i+1)*n1]
        
    sy = Uc_hat_y.shape
    sx = Uc_hat_x.shape
    Uc_hat_x[:] = rollaxis(Uc_hat_y.reshape((sy[0], P2, sx[1], sx[2])), 1).reshape(sx)        
    return Uc_hat_x

@optimizer
def transform_Uc_yx(Uc_hat_y, Uc_hat_xr, P2):
    #n0 = Uc_hat_y.shape[0]
    #n1 = Uc_hat_xr.shape[1]
    #for i in range(P2): 
        #Uc_hat_y[:, i*n1:(i+1)*n1] = Uc_hat_xr[i*n0:(i+1)*n0]
        
    sy = Uc_hat_y.shape
    sx = Uc_hat_xr.shape
    Uc_hat_y[:] = rollaxis(Uc_hat_xr.reshape((P2, sy[0], sx[1], sx[2])), 1).reshape(sy)  
        
    return Uc_hat_y

def create_wavenumber_arrays(N, N1, N2, xyrank, xzrank, float):
    # Set wavenumbers in grid
    kx = fftfreq(N[0], 1./N[0]).astype(int)
    ky = fftfreq(N[1], 1./N[1]).astype(int)
    kz = fftfreq(N[2], 1./N[2]).astype(int)
    Lp = 2*pi/config.L
    k2 = slice(xyrank*N2[0], (xyrank+1)*N2[0], 1)
    k1 = slice(xzrank*N1[2]/2, (xzrank+1)*N1[2]/2, 1)
    K  = array(meshgrid(kx[k2], ky, kz[k1], indexing='ij'), dtype=float)

    # Filter for dealiasing nonlinear convection
    kmax = 2./3.*(N/2+1)
    dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                    (abs(K[2]) < kmax[2]), dtype=uint8)

    # Scale with physical mesh size. This takes care of mapping the physical domain to a computational cube of size (2pi)**3
    K[0] *= Lp[0]; K[1] *= Lp[1]; K[2] *= Lp[2] 
    K2 = sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)    
    return K, K2, K_over_K2, dealias

def setupDNS(comm, float, complex, uint8, mpitype, N, L,
             num_processes, rank, mgrid, **kwargs):

    # Each cpu gets ownership of a pencil of size N1*N2*N in real space
    # and (N1/2+1)*N2*N in Fourier space. However, the Nyquist mode is
    # neglected and as such the actual number is N1/2*N2*N in Fourier space.    
    assert num_processes > 1 and config.P1 < num_processes
    P1 = config.P1
    
    P2 = num_processes / P1
    N1 = N/P1
    N2 = N/P2
    
    if not (num_processes % 2 == 0 or num_processes == 1):
        raise IOError("Number of cpus must be even")

    if not ((P1 == 1 or P1 % 2 == 0) and (P2 == 1 or P2 % 2 == 0)):
        raise IOError("Number of cpus in each direction must be even")

    # Create two communicator groups for each rank
    # The goups correspond to chunks in the xy-plane and the xz-plane
    commxz = comm.Split(rank/P1)
    commxy = comm.Split(rank%P1)
    
    xzrank = commxz.Get_rank() # Local rank in xz-plane
    xyrank = commxy.Get_rank() # Local rank in xy-plane
    
    # Create the physical mesh
    x1 = slice(xzrank * N1[0], (xzrank+1) * N1[0], 1)
    x2 = slice(xyrank * N2[1], (xyrank+1) * N2[1], 1)
    X = mgrid[x1, x2, :N[2]].astype(float)
    X[0] *= L[0]/N[0]; X[1] *= L[1]/N[1]; X[2] *= L[2]/N[2]

    """
    Solution U is real and as such its transform, U_hat = fft(U)(k), 
    is such that fft(U)(k) = conj(fft(U)(N-k)) and thus it is sufficient 
    to store N/2+1 Fourier coefficients in the first transformed direction 
    (y). However, the Nyquist mode (k=N/2+1) is neglected in the 3D fft.
    The Nyquist mode in included in temporary arrays simply because rfft/irfft 
    expect N/2+1 modes.
    """
    FFT = FastFourierTransform(N, P1, comm, mpitype)

    Nf = N[2]/2+1 # Total Fourier coefficients in z-direction
    U     = empty((3,) + FFT.real_shape(), dtype=float)
    U_hat = empty((3,) + FFT.complex_shape(), dtype=complex)
    P     = empty(FFT.real_shape(), dtype=float)
    P_hat = empty(FFT.complex_shape(), dtype=complex)

    # RHS array
    dU = empty((3,) + FFT.complex_shape(), dtype=complex)
    
    # work arrays (Not required by all convection methods)
    U_tmp  = empty((3,) + FFT.real_shape(), dtype=float)
    F_tmp  = empty((3,) + FFT.complex_shape(), dtype=complex)
    curl   = empty((3,) + FFT.real_shape(), dtype=float)
    Source = None
    
    K, K2, K_over_K2, dealias = create_wavenumber_arrays(N, N1, N2, xyrank, xzrank, float)

    del kwargs
    return locals()

def setupMHD(comm, float, complex, uint8, mpitype, N, L,
             num_processes, rank, mgrid, **kwargs):

    # Each cpu gets ownership of a pencil of size N1*N2*N in real space
    # and (N1/2+1)*N2*N in Fourier space. However, the Nyquist mode is
    # neglected and as such the actual number is N1/2*N2*N in Fourier space.
    assert num_processes > 1 and config.P1 < num_processes
    P1 = config.P1
    
    P2 = num_processes / P1
    N1 = N/P1
    N2 = N/P2
    
    if not (num_processes % 2 == 0 or num_processes == 1):
        raise IOError("Number of cpus must be even")

    if not ((P1 == 1 or P1 % 2 == 0) and (P2 == 1 or P2 % 2 == 0)):
        raise IOError("Number of cpus in each direction must be even")

    # Create two communicator groups for each rank
    # The goups correspond to chunks in the xy-plane and the xz-plane
    commxz = comm.Split(rank/P1)
    commxy = comm.Split(rank%P1)
    
    xzrank = commxz.Get_rank() # Local rank in xz-plane
    xyrank = commxy.Get_rank() # Local rank in xy-plane
    
    # Create the physical mesh
    x1 = slice(xzrank * N1[0], (xzrank+1) * N1[0], 1)
    x2 = slice(xyrank * N2[1], (xyrank+1) * N2[1], 1)
    X = mgrid[x1, x2, :N[2]].astype(float)
    X[0] *= L[0]/N[0]; X[1] *= L[1]/N[1]; X[2] *= L[2]/N[2]
    
    FFT = FastFourierTransform(N, P1, comm, mpitype)

    Nf = N[2]/2+1 # Total Fourier coefficients in z-direction
    UB     = empty((6,) + FFT.real_shape(), dtype=float)
    UB_hat = empty((6,) + FFT.complex_shape(), dtype=complex)
    P      = empty(FFT.real_shape(), dtype=float)
    P_hat  = empty(FFT.complex_shape(), dtype=complex)

    # Create views into large data structures
    U     = UB[:3] 
    U_hat = UB_hat[:3]
    B     = UB[3:]
    B_hat = UB_hat[3:]

    # RHS array
    dU = empty((6,) + FFT.complex_shape(), dtype=complex)

    # work arrays (Not required by all convection methods)
    U_tmp  = empty((3,) + FFT.real_shape(), dtype=float)
    F_tmp  = empty((3, 3,) + FFT.complex_shape(), dtype=complex)
    curl   = empty((3,) + FFT.real_shape(), dtype=float)
    Source = None
    
    K, K2, K_over_K2, dealias = create_wavenumber_arrays(N, N1, N2, xyrank, xzrank, float)
    
    del kwargs
    return locals()

setup = {"MHD": setupMHD,
         "NS":  setupDNS,
         "VV":  setupDNS}[config.solver]

class FastFourierTransform(object):
    
    def __init__(self, N, P1, comm, mpitype):
        self.N = N
        self.Nf = Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
        self.comm = comm
        self.mpitype = mpitype
        self.num_processes = comm.Get_size()
        self.rank = comm.Get_rank()        
        self.P1 = P1
        self.P2 = P2 = self.num_processes / P1
        self.N1 = N1 = N / P1
        self.N2 = N2 = N / P2
        self.commxz = comm.Split(self.rank/P1)
        self.commxy = comm.Split(self.rank%P1)

        # Initialize MPI work arrays globally
        self.Uc_hat_z  = empty((N1[0], N2[1], Nf), dtype=complex)
        self.Uc_hat_x  = empty((N[0], N2[1], N1[2]/2), dtype=complex)
        self.Uc_hat_xr = empty((N[0], N2[1], N1[2]/2), dtype=complex)
        self.Uc_hat_y  = zeros((N2[0], N[1], N1[2]/2), dtype=complex)

    def real_shape(self):
        """The local shape of the real data"""
        return (self.N1[0], self.N2[1], self.N[2])

    def complex_shape(self):
        """The local shape of the complex data"""
        return (self.N2[0], self.N[1], self.N1[2]/2)
    
    def complex_shape_T(self):
        """The local transposed shape of the complex data"""
        return (self.Np[0], self.N[1], self.Nf)
        
    def complex_shape_I(self):
        """A local intermediate shape of the complex data"""
        return (self.Np[0], self.num_processes, self.Np[1], self.Nf)

    #def complex_shape_padded_T(self):
        #"""The local shape of the transposed complex data padded in x and z directions"""
        #return (3*self.Np[0]/2, 3*self.N[1]/2, 3*self.N[2]/4+1)

    #def real_shape_padded(self):
        #"""The local shape of the real data"""
        #return (3*self.Np[0]/2, 3*self.N[1]/2, 3*self.N[2]/2)
    
    #def complex_shape_padded(self):
        #return (3*self.N[0]/2, 3*self.Np[1]/2, 3*self.N[2]/4+1)
    
    def ifftn(self, fu, u):
        """ifft in three directions using mpi.
        Need to do ifft in reversed order of fft
        """
        # Do first owned direction
        self.Uc_hat_y[:] = ifft(fu, axis=1)

        # Transform to x all but k=N/2 (the neglected Nyquist mode)
        self.Uc_hat_x[:] = 0
        self.Uc_hat_x[:] = transform_Uc_xy(self.Uc_hat_x, self.Uc_hat_y, self.P2)
            
        # Communicate in xz-plane and do fft in x-direction
        self.commxy.Alltoall([self.Uc_hat_x, self.mpitype], [self.Uc_hat_xr, self.mpitype])
        self.Uc_hat_x[:] = ifft(self.Uc_hat_xr, axis=0)
            
        # Communicate and transform in xy-plane
        self.commxz.Alltoall([self.Uc_hat_x, self.mpitype], [self.Uc_hat_xr, self.mpitype])
        self.Uc_hat_z[:] = transform_Uc_zx(self.Uc_hat_z, self.Uc_hat_xr, self.P1)
                
        # Do fft for y-direction
        self.Uc_hat_z[:, :, -1] = 0
        u[:] = irfft(self.Uc_hat_z, axis=2)
        return u

    #@profile
    def fftn(self, u, fu):
        """fft in three directions using mpi
        """
        # Do fft in z direction on owned data
        self.Uc_hat_z[:] = rfft(u, axis=2)
        
        # Transform to x direction neglecting k=N/2 (Nyquist)
        self.Uc_hat_x[:] = transform_Uc_xz(self.Uc_hat_x, self.Uc_hat_z, self.P1)
        
        # Communicate and do fft in x-direction
        self.commxz.Alltoall([self.Uc_hat_x, self.mpitype], [self.Uc_hat_xr, self.mpitype])
        self.Uc_hat_x[:] = fft(self.Uc_hat_xr, axis=0)        
        
        # Communicate and transform to final z-direction
        self.commxy.Alltoall([self.Uc_hat_x, self.mpitype], [self.Uc_hat_xr, self.mpitype])  
        self.Uc_hat_y[:] = transform_Uc_yx(self.Uc_hat_y, self.Uc_hat_xr, self.P2)
                                    
        # Do fft for last direction 
        fu[:] = fft(self.Uc_hat_y, axis=1)
        return fu
