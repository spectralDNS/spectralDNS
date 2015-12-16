__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-12-30"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from cbcdns import config
from ..fft.wrappyfftw import *
from ..shen.shentransform import ShenDirichletBasis, ShenNeumannBasis, ShenBiharmonicBasis, SFTc
from ..shenGeneralBCs.shentransform import ShenBasis
from ..optimization import optimizer
from numpy import array, sum, meshgrid, mgrid, where, abs, pi, uint8, rollaxis, arange

__all__ = ['setup', 'ifftn_mpi', 'fftn_mpi', 'FastShenFourierTransform']

@optimizer
def transpose_Uc(Uc_hatT, U_mpi, num_processes):
    Uc_hatT[:] = rollaxis(U_mpi, 1).reshape(Uc_hatT.shape)
    return Uc_hatT

@optimizer
def transpose_Umpi(U_mpi, Uc_hatT, num_processes):
    U_mpi[:] = rollaxis(Uc_hatT.reshape(Np[0], num_processes, Np[1], Nf), 1)
    return U_mpi

def create_wavenumber_arrays(N, Np, Nf, rank, float):
    kx = fftfreq(N[0], 1./N[0])
    ky = fftfreq(N[1], 1./N[1])[rank*Np[1]:(rank+1)*Np[1]]
    kz = fftfreq(N[2], 1./N[2])[:Nf]
    kz[-1] *= -1
    Lp = 2*pi/config.L
    K  = array(meshgrid(kx, ky, kz, indexing='ij'), dtype=float)
    K[0] *= Lp[0]; K[1] *= Lp[1]; K[2] *= Lp[2] # scale with physical mesh size. This takes care of mapping the physical domain to a computational cube of size (2pi)**3
    K2 = sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)

    # Filter for dealiasing nonlinear convection
    kmax = 2./3.*(N/2+1)
    dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                    (abs(K[2]) < kmax[2]), dtype=uint8)
    
    return K, K2, K_over_K2, dealias

def setupDNS(comm, float, complex, mpitype, N, L, mgrid,
             num_processes, rank, **kwargs):
    
    if not num_processes in [2**i for i in range(config.M[0]+1)]:
        raise IOError("Number of cpus must be in ", [2**i for i in range(config.M[0]+1)])

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

    # RHS array
    dU     = empty((3, N[0], Np[1], Nf), dtype=complex)

    # work arrays (Not required by all convection methods)
    U_tmp  = empty((3, Np[0], N[1], N[2]), dtype=float)
    F_tmp  = empty((3, N[0], Np[1], Nf), dtype=complex)
    curl   = empty((3, Np[0], N[1], N[2]), dtype=float)   
    Source = None
    
    init_fft(N, Nf, Np, complex, num_processes, comm, rank, mpitype)
        
    K, K2, K_over_K2, dealias = create_wavenumber_arrays(N, Np, Nf, rank, float)
    
    del kwargs
    return locals() # Lazy (need only return what is needed)

def setupMHD(comm, float, complex, mpitype, N, L, mgrid,
             num_processes, rank, **kwargs):
    
    if not num_processes in [2**i for i in range(config.M[0]+1)]:
        raise IOError("Number of cpus must be in ", [2**i for i in range(config.M[0]+1)])

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
    UB     = empty((6, Np[0], N[1], N[2]), dtype=float)  
    UB_hat = empty((6, N[0], Np[1], Nf), dtype=complex)
    P      = empty((Np[0], N[1], N[2]), dtype=float)
    P_hat  = empty((N[0], Np[1], Nf), dtype=complex)
    
    # Create views into large data structures
    U     = UB[:3] 
    U_hat = UB_hat[:3]
    B     = UB[3:]
    B_hat = UB_hat[3:]

    # RHS array
    dU = empty((6, N[0], Np[1], Nf), dtype=complex)

    # work arrays (Not required by all convection methods)
    U_tmp  = empty((3, Np[0], N[1], N[2]), dtype=float)
    F_tmp  = empty((3, 3, N[0], Np[1], Nf), dtype=complex)
    curl   = empty((3, Np[0], N[1], N[2]), dtype=float)   
    Source = None
    
    init_fft(N, Nf, Np, complex, num_processes, comm, rank, mpitype)
    
    K, K2, K_over_K2, dealias = create_wavenumber_arrays(N, Np, Nf, rank, float)
    
    del kwargs
    return locals() # Lazy (need only return what is needed)

def setupShen(comm, float, complex, mpitype, N, L, mgrid,
              num_processes, rank, MPI, **kwargs):
    if not num_processes in [2**i for i in range(config.M[0]+1)]:
        raise IOError("Number of cpus must be in ", [2**i for i in range(config.M[0]+1)])
    
    Np = N / num_processes

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad="GL")
    SN = ShenNeumannBasis(quad="GC")
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
    
    FST = FastShenFourierTransform(N, MPI)

    U     = empty((3,)+FST.real_shape(), dtype=float)
    U_hat = empty((3,)+FST.complex_shape(), dtype=complex)
    P     = empty(FST.real_shape(), dtype=float)
    P_hat = empty(FST.complex_shape(), dtype=complex)
    Pcorr = empty(FST.complex_shape(), dtype=complex)

    U0      = empty((3,)+FST.real_shape(), dtype=float)
    U_hat0  = empty((3,)+FST.complex_shape(), dtype=complex)
    U_hat1  = empty((3,)+FST.complex_shape(), dtype=complex)

    U_tmp   = empty((3,)+FST.real_shape(), dtype=float)
    U_tmp2  = empty((3,)+FST.real_shape(), dtype=float)
    U_tmp3  = empty((3,)+FST.real_shape(), dtype=float)
    F_tmp   = empty((3,)+FST.complex_shape(), dtype=complex)
    F_tmp2  = empty((3,)+FST.complex_shape(), dtype=complex)

    dU      = empty((3,)+FST.complex_shape(), dtype=complex)

    conv0   = empty((3,)+FST.complex_shape(), dtype=complex)
    conv1   = empty((3,)+FST.complex_shape(), dtype=complex)
    diff0   = empty((3,)+FST.complex_shape(), dtype=complex)
    Source  = empty((3,)+FST.real_shape(), dtype=float) 
    Sk      = empty((3,)+FST.complex_shape(), dtype=complex) 

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
    kmax[0] = N[0]
    dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                    (abs(K[2]) < kmax[2]), dtype=uint8)
    
    del kwargs 
    return locals()

def setupShenKMM(comm, float, complex, mpitype, N, L, mgrid,
                 num_processes, rank, MPI, **kwargs):
    if not num_processes in [2**i for i in range(config.M[0]+1)]:
        raise IOError("Number of cpus must be in ", [2**i for i in range(config.M[0]+1)])
    
    Np = N / num_processes

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad="GL")
    SB = ShenBiharmonicBasis(quad="GL")
    SN = ShenNeumannBasis(quad="GL")   # For pressure calculation
    points, weights = ST.points_and_weights(N[0])
    pointsp, weightsp = SB.points_and_weights(N[0])

    x1 = arange(N[1], dtype=float)*L[1]/N[1]
    x2 = arange(N[2], dtype=float)*L[2]/N[2]

    # Get grid for velocity points
    X = array(meshgrid(points[rank*Np[0]:(rank+1)*Np[0]], x1, x2, indexing='ij'), dtype=float)

    Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
    Nu = N[0]-2   # Number of velocity modes in Shen basis
    Nb = N[0]-4   # Number of velocity modes in Shen biharmonic basis
    u_slice = slice(0, Nu)
    v_slice = slice(0, Nb)
    
    FST = FastShenFourierTransform(N, MPI)

    U     = empty((3,)+FST.real_shape(), dtype=float)
    U_hat = empty((3,)+FST.complex_shape(), dtype=complex)
    P     = empty(FST.real_shape(), dtype=float)
    P_hat = empty(FST.complex_shape(), dtype=complex)

    U0      = empty((3,)+FST.real_shape(), dtype=float)
    U_hat0  = empty((3,)+FST.complex_shape(), dtype=complex)
    
    # We're solving for:
    u = U_hat0[0]
    g = empty(FST.complex_shape(), dtype=complex)

    U_tmp   = empty((3,)+FST.real_shape(), dtype=float)
    U_tmp2  = empty((3,)+FST.real_shape(), dtype=float)
    F_tmp   = empty((3,)+FST.complex_shape(), dtype=complex)
    F_tmp2  = empty((3,)+FST.complex_shape(), dtype=complex)

    dU      = empty((3,)+FST.complex_shape(), dtype=complex)
    conv0   = empty((3,)+FST.complex_shape(), dtype=complex)
    conv1   = empty((3,)+FST.complex_shape(), dtype=complex)
    hv      = empty(FST.complex_shape(), dtype=complex)
    hg      = empty(FST.complex_shape(), dtype=complex)
    diff0   = empty((3,)+FST.complex_shape(), dtype=complex)
    Source  = empty((3,)+FST.real_shape(), dtype=float) 
    Sk      = empty((3,)+FST.complex_shape(), dtype=complex) 

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
    K2 = K[1]*K[1]+K[2]*K[2]
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)

    # Filter for dealiasing nonlinear convection
    kmax = 2./3.*(N/2+1)
    kmax[0] = N[0]
    dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                    (abs(K[2]) < kmax[2]), dtype=uint8)
    
    del kwargs 
    return locals()

def setupShenMHD(comm, float, complex, mpitype, N, L, mgrid,
              num_processes, rank, MPI, **kwargs):
    if not num_processes in [2**i for i in range(config.M[0]+1)]:
        raise IOError("Number of cpus must be in ", [2**i for i in range(config.M[0]+1)])
    
    Np = N / num_processes

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad="GL")
    SN = ShenNeumannBasis(quad="GC")
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


    FST = FastShenFourierTransform(N, MPI)

    U     = empty((6,)+FST.real_shape(), dtype=float)
    U_hat = empty((6,)+FST.complex_shape(), dtype=complex)
    P     = empty(FST.real_shape(), dtype=float)
    P_hat = empty(FST.complex_shape(), dtype=complex)
    Pcorr = empty(FST.complex_shape(), dtype=complex)

    U0      = empty((6,)+FST.real_shape(), dtype=float)
    U_hat0  = empty((6,)+FST.complex_shape(), dtype=complex)
    U_hat1  = empty((6,)+FST.complex_shape(), dtype=complex)
    UT      = empty((6,)+FST.complex_shape(), dtype=float)
    
    
    U_tmp   = empty((6,)+FST.real_shape(), dtype=float)
    U_tmp2  = empty((6,)+FST.real_shape(), dtype=float)
    U_tmp3  = empty((6,)+FST.real_shape(), dtype=float)
    U_tmp4  = empty((6,)+FST.real_shape(), dtype=float)
    F_tmp   = empty((6,)+FST.complex_shape(), dtype=complex)
    F_tmp2  = empty((6,)+FST.complex_shape(), dtype=complex)

    dU      = empty((7,)+FST.complex_shape(), dtype=complex)

    conv0    = empty((3,)+FST.complex_shape(), dtype=complex)
    conv1    = empty((3,)+FST.complex_shape(), dtype=complex)
    magconv  = empty((3,)+FST.complex_shape(), dtype=complex)
    magconvU = empty((3,)+FST.complex_shape(), dtype=complex)
    diff0    = empty((3,)+FST.complex_shape(), dtype=complex)
    Source   = empty((3,)+FST.real_shape(), dtype=float) 
    Sk       = empty((3,)+FST.complex_shape(), dtype=complex) 
    
    #Uc      = empty((Np[0], N[1], N[2]))
    #Uc2     = empty((Np[0], N[1], N[2]))
    #Uc_hat  = empty((N[0], Np[1], Nf), dtype="complex")
    #Uc_hat2 = empty((N[0], Np[1], Nf), dtype="complex")
    #Uc_hat3 = empty((N[0], Np[1], Nf), dtype="complex")
    #Uc_hatT = empty((Np[0], N[1], Nf), dtype="complex")
    #U_mpi   = empty((num_processes, Np[0], Np[1], Nf), dtype="complex")
    #U_mpi2  = empty((num_processes, Np[0], Np[1], N[2])) 

    kx = arange(N[0]).astype(float)
    ky = fftfreq(N[1], 1./N[1])[rank*Np[1]:(rank+1)*Np[1]]
    kz = fftfreq(N[2], 1./N[2])[:Nf]
    kz[-1] *= -1.0

    #mpidouble = MPI.DOUBLE
    #init_fst(N, Nf, Np, complex, num_processes, comm, rank, mpitype, mpidouble)
    
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
    kmax[0] = N[0]
    dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                    (abs(K[2]) < kmax[2]), dtype=uint8)
    
    del kwargs 
    return locals()

def setupShenGeneralBCs(comm, float, complex, mpitype, N, L, mgrid,
              num_processes, rank, MPI, **kwargs):
    if not num_processes in [2**i for i in range(config.M[0]+1)]:
        raise IOError("Number of cpus must be in ", [2**i for i in range(config.M[0]+1)])
    
    Np = N / num_processes

    # Get points and weights for Chebyshev weighted integrals
    BC1 = array([1,0,0, 1,0,0])
    BC2 = array([0,1,0, 0,1,0])
    ST = ShenBasis(BC1, quad="GL")
    SN = ShenBasis(BC2, quad="GC")
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

    FST = FastShenFourierTransform(N, MPI)

    U     = empty((3,)+FST.real_shape(), dtype=float)
    U_hat = empty((3,)+FST.complex_shape(), dtype=complex)
    P     = empty(FST.real_shape(), dtype=float)
    P_hat = empty(FST.complex_shape(), dtype=complex)
    Pcorr = empty(FST.complex_shape(), dtype=complex)

    U0      = empty((3,)+FST.real_shape(), dtype=float)
    U_hat0  = empty((3,)+FST.complex_shape(), dtype=complex)
    U_hat1  = empty((3,)+FST.complex_shape(), dtype=complex)

    U_tmp   = empty((3,)+FST.real_shape(), dtype=float)
    U_tmp2  = empty((3,)+FST.real_shape(), dtype=float)
    U_tmp3  = empty((3,)+FST.real_shape(), dtype=float)
    U_tmp4  = empty((3,)+FST.real_shape(), dtype=float)        
    F_tmp   = empty((3,)+FST.complex_shape(), dtype=complex)
    F_tmp2  = empty((3,)+FST.complex_shape(), dtype=complex)

    dU      = empty((4,)+FST.complex_shape(), dtype=complex)

    conv0   = empty((3,)+FST.complex_shape(), dtype=complex)
    conv1   = empty((3,)+FST.complex_shape(), dtype=complex)
    diff0   = empty((3,)+FST.complex_shape(), dtype=complex)
    Source  = empty((3,)+FST.real_shape(), dtype=float) 
    Sk      = empty((3,)+FST.complex_shape(), dtype=complex) 
    
    
    #Uc      = empty((Np[0], N[1], N[2]))
    #Uc2     = empty((Np[0], N[1], N[2]))
    #Uc_hat  = empty((N[0], Np[1], Nf), dtype="complex")
    #Uc_hat2 = empty((N[0], Np[1], Nf), dtype="complex")
    #Uc_hat3 = empty((N[0], Np[1], Nf), dtype="complex")
    #Uc_hatT = empty((Np[0], N[1], Nf), dtype="complex")
    #U_mpi   = empty((num_processes, Np[0], Np[1], Nf), dtype="complex")
    #U_mpi2  = empty((num_processes, Np[0], Np[1], N[2]))


    kx = arange(N[0]).astype(float)
    ky = fftfreq(N[1], 1./N[1])[rank*Np[1]:(rank+1)*Np[1]]
    kz = fftfreq(N[2], 1./N[2])[:Nf]
    kz[-1] *= -1.0

    #mpidouble = MPI.DOUBLE
    #init_fst(N, Nf, Np, complex, num_processes, comm, rank, mpitype, mpidouble)
    
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
    kmax[0] = N[0]
    dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                    (abs(K[2]) < kmax[2]), dtype=uint8)
    
    del kwargs 
    return locals()
        
setup = {"MHD": setupMHD,
         "NS":  setupDNS,
         "VV":  setupDNS,
         "IPCS": setupShen,
         "IPCSR": setupShen,
         "KMM": setupShenKMM,
         "KMMRK3": setupShenKMM,
         "ChannelRK4": setupShen,
         "IPCS_MHD": setupShenMHD,
         "IPCS_GeneralBCs": setupShenGeneralBCs}[config.solver]        

def init_fft(N, Nf, Np, complex, num_processes, comm, rank, mpitype):
    # Initialize MPI work arrays globally
    Uc_hat  = empty((N[0], Np[1], Nf), dtype=complex)
    Uc_hatT = empty((Np[0], N[1], Nf), dtype=complex)
    Uc_send = Uc_hat.reshape((num_processes, Np[0], Np[1], Nf))
    U_mpi   = empty((num_processes, Np[0], Np[1], Nf), dtype=complex)
    globals().update(locals())
    
#@profile    
def ifftn_mpi(fu, u):
    """ifft in three directions using mpi.
    Need to do ifft in reversed order of fft
    """
    if num_processes == 1:
        u[:] = irfftn(fu, axes=(0,1,2))
        return u
    
    # Do first owned direction
    Uc_hat[:] = ifft(fu, axis=0)
        
    if config.communication == 'alltoall':
        # Communicate all values
        comm.Alltoall([Uc_hat, mpitype], [U_mpi, mpitype])
        Uc_hatT[:] = transpose_Uc(Uc_hatT, U_mpi, num_processes)
    
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
        fu[:] = rfftn(u, axes=(0,1,2))
        return fu
    
    if config.communication == 'alltoall':
        # Do 2 ffts in y-z directions on owned data
        Uc_hatT[:] = rfft2(u, axes=(1,2))
        
        # Transform data to align with x-direction  
        U_mpi[:] = transpose_Umpi(U_mpi, Uc_hatT, num_processes)
            
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
    fu[:] = fft(fu, axis=0)
    return fu
     
     
class FastShenFourierTransform(object):
    def __init__(self, N, MPI):
        self.N = N         # The global size of the problem
        self.Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
        self.comm = MPI.COMM_WORLD
        self.num_processes = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.Np = N / self.num_processes     
        self.mpitype = MPI.F_DOUBLE_COMPLEX
        
        # Initialize intermediate MPI work arrays
        self.U_mpi   = empty((self.num_processes, self.Np[0], self.Np[1], self.Nf), dtype=complex)
        self.U_mpi2  = empty((self.num_processes, self.Np[0], self.Np[1], self.N[2]))
        self.UT      = empty((3, self.N[0], self.Np[1], self.N[2]))
        self.Uc_hat  = empty(self.complex_shape(), dtype=complex)
        self.Uc_hatT = empty(self.complex_shape_T(), dtype=complex)
        
    def real_shape(self):
        """The local shape of the real data"""
        return (self.Np[0], self.N[1], self.N[2])

    def complex_shape(self):
        """The local shape of the complex data"""
        return (self.N[0], self.Np[1], self.Nf)
    
    def complex_shape_T(self):
        """The local transposed shape of the complex data"""
        return (self.Np[0], self.N[1], self.Nf)
        
    def complex_shape_I(self):
        """A local intermediate shape of the complex data"""
        return (self.Np[0], self.num_processes, self.Np[1], self.Nf)
    
    def fss(self, u, fu, S):
        """Fast Shen scalar product of x-direction, Fourier transform of y and z"""
        self.Uc_hatT[:] = rfft2(u, axes=(1,2))
        self.U_mpi[:] = rollaxis(self.Uc_hatT.reshape(self.complex_shape_I()), 1)
        self.comm.Alltoall([self.U_mpi, self.mpitype], [self.Uc_hat, self.mpitype])
        fu = S.fastShenScalar(self.Uc_hat, fu)
        return fu

    def ifst(self, fu, u, S):
        """Inverse Shen transform of x-direction, Fourier in y and z"""
        self.Uc_hat[:] = S.ifst(fu, self.Uc_hat)
        self.comm.Alltoall([self.Uc_hat, self.mpitype], [self.U_mpi, self.mpitype])
        self.Uc_hatT[:] = rollaxis(self.U_mpi, 1).reshape(self.complex_shape_T())
        u[:] = irfft2(self.Uc_hatT, axes=(1,2))
        return u

    def fst(self, u, fu, S):
        """Fast Shen transform of x-direction, Fourier transform of y and z"""
        self.Uc_hatT[:] = rfft2(u, axes=(1,2))
        self.U_mpi[:] = rollaxis(self.Uc_hatT.reshape(self.complex_shape_I()), 1)
        self.comm.Alltoall([self.U_mpi, self.mpitype], [self.Uc_hat, self.mpitype])
        fu = S.fst(self.Uc_hat, fu)
        return fu

    def fct(self, u, fu, S):
        """Fast Cheb transform of x-direction, Fourier transform of y and z"""
        self.Uc_hatT[:] = rfft2(u, axes=(1,2))
        self.U_mpi[:] = rollaxis(self.Uc_hatT.reshape(self.complex_shape_I()), 1)
        self.comm.Alltoall([self.U_mpi, self.mpitype], [self.Uc_hat, self.mpitype])
        fu = S.fct(self.Uc_hat, fu)
        return fu

    def ifct(self, fu, u, S):
        """Inverse Cheb transform of x-direction, Fourier in y and z"""
        self.Uc_hat[:] = S.ifct(fu, self.Uc_hat)
        self.comm.Alltoall([self.Uc_hat, self.mpitype], [self.U_mpi, self.mpitype])
        self.Uc_hatT[:] = rollaxis(self.U_mpi, 1).reshape(self.complex_shape_T())
        u[:] = irfft2(self.Uc_hatT, axes=(1,2))
        return u

    def fct0(self, u, fu, S):
        """Fast Cheb transform of x-direction. No FFT, just align data in x-direction and do fct."""
        self.U_mpi2[:] = rollaxis(u.reshape(self.Np[0], self.num_processes, self.Np[1], self.N[2]), 1)
        self.comm.Alltoall([self.U_mpi2, self.mpitype], [self.UT[0], self.mpitype])
        fu = S.fct(self.UT[0], fu)
        return fu

    def ifct0(self, fu, u, S):
        """Fast Cheb transform of x-direction. No FFT, just align data in x-direction and do ifct"""
        self.UT[0] = S.ifct(fu, self.UT[0])
        self.comm.Alltoall([self.UT[0], self.mpitype], [self.U_mpi2, self.mpitype])
        u[:] = rollaxis(self.U_mpi2, 1).reshape(u.shape)
        return u
    
    def chebDerivative_3D0(self, fj, u0, S):
        self.UT[0] = self.fct0(fj, self.UT[0], S)
        self.UT[1] = SFTc.chebDerivativeCoefficients_3D(self.UT[0], self.UT[1]) 
        u0[:] = self.ifct0(self.UT[1], u0, S)
        return u0


#class FFT(object):
    
    #def __init__(self, N, comm, mpitype):
        #self.N = N
        #self.Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
        #self.num_processes = comm.Get_size()
        #self.rank = comm.Get_rank()
        #self.Np = N / num_processes     
        
        ## Initialize MPI work arrays globally
        #self.Uc_hat  = empty((N[0], self.Np[1], self.Nf), dtype=complex)
        #self.Uc_hatT = empty((self.Np[0], N[1], self.Nf), dtype=complex)
        #self.Uc_send = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
        #self.U_mpi   = empty((self.num_processes, self.Np[0], self.Np[1], self.Nf), dtype=complex)
    
    #def ifftn(self, fu, u):
        #"""ifft in three directions using mpi.
        #Need to do ifft in reversed order of fft
        #"""
        #if self.num_processes == 1:
            #u[:] = irfftn(fu, axes=(0,1,2))
            #return u
        
        ## Do first owned direction
        #self.Uc_hat[:] = ifft(fu, axis=0)
            
        #if config.communication == 'alltoall':
            ## Communicate all values
            #self.comm.Alltoall([self.Uc_hat, self.mpitype], [self.U_mpi, self.mpitype])
            #self.Uc_hatT[:] = rollaxis(self.U_mpi, 1).reshape(self.Uc_hatT.shape)
        
        #else:
            #for i in xrange(self.num_processes):
                #if not i == self.rank:
                    #self.comm.Sendrecv_replace([self.Uc_send[i], self.mpitype], i, 0, i, 0)   
                #self.Uc_hatT[:, i*self.Np[1]:(i+1)*self.Np[1]] = self.Uc_send[i]
            
        ## Do last two directions
        #u = irfft2(self.Uc_hatT, axes=(1,2))
        #return u

    ##@profile
    #def fftn(self, u, fu):
        #"""fft in three directions using mpi
        #"""
        #if self.num_processes == 1:
            #fu[:] = rfftn(u, axes=(0,1,2))
            #return fu
        
        #if config.communication == 'alltoall':
            ## Do 2 ffts in y-z directions on owned data
            #self.Uc_hatT[:] = rfft2(u, axes=(1,2))
            
            ## Transform data to align with x-direction  
            #self.U_mpi[:] = rollaxis(self.Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
                
            ## Communicate all values
            #self.comm.Alltoall([self.U_mpi, self.mpitype], [fu, self.mpitype])  
        
        #else:
            ## Communicating intermediate result 
            #ft = fu.transpose(1,0,2)
            #ft[:] = rfft2(u, axes=(1,2))
            #fu_send = fu.reshape((self.num_processes, self.Np[1], self.Np[1], self.Nf))
            #for i in xrange(self.num_processes):
                #if not i == self.rank:
                    #self.comm.Sendrecv_replace([fu_send[i], self.mpitype], i, 0, i, 0)   
            #fu_send[:] = fu_send.transpose(0,2,1,3)
                        
        ## Do fft for last direction 
        #fu[:] = fft(fu, axis=0)
        #return fu
