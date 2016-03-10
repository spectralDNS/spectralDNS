__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralDNS import config
from mpiFFT4py import *
from ..shen.shentransform import ShenDirichletBasis, ShenNeumannBasis, ShenBiharmonicBasis, SFTc
from ..shenGeneralBCs.shentransform import ShenBasis
from numpy import array, sum, meshgrid, mgrid, where, abs, pi, uint8, rollaxis, arange, log2

__all__ = ['setup']

def setupShen(N, L, MPI, float, complex, **kwargs):
    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad="GL")
    SN = ShenNeumannBasis(quad="GC")

    Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
    Nu = N[0]-2   # Number of velocity modes in Shen basis
    Nq = N[0]-3   # Number of pressure modes in Shen basis
    u_slice = slice(0, Nu)
    p_slice = slice(1, Nu)
    
    FST = FastShenFourierTransform(N, L, MPI)
    
    # Get grid for velocity points
    X = FST.get_local_mesh(ST)
    x0, x1, x2 = FST.get_mesh_dims(ST)

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

    H        = empty((3,)+FST.real_shape(), dtype=float)
    H0       = empty((3,)+FST.real_shape(), dtype=float)
    H1       = empty((3,)+FST.real_shape(), dtype=float)
    H_hat    = empty((3,)+FST.complex_shape(), dtype=complex)
    H_hat0   = empty((3,)+FST.complex_shape(), dtype=complex)
    H_hat1   = empty((3,)+FST.complex_shape(), dtype=complex)

    diff0   = empty((3,)+FST.complex_shape(), dtype=complex)
    Source  = empty((3,)+FST.real_shape(), dtype=float) 
    Sk      = empty((3,)+FST.complex_shape(), dtype=complex) 
    
    dealias = None
    if not config.dealias == "3/2-rule":
        dealias = FST.get_dealias_filter()
    K = FST.get_scaled_local_wavenumbermesh()
    K2 = K[1]*K[1]+K[2]*K[2]
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)

    del kwargs 
    return locals()


def setupShenKMM(N, L, MPI, float, complex, **kwargs):

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad="GL")
    SB = ShenBiharmonicBasis(quad="GL")

    Nu = N[0]-2   # Number of velocity modes in Shen basis
    Nb = N[0]-4   # Number of velocity modes in Shen biharmonic basis
    u_slice = slice(0, Nu)
    v_slice = slice(0, Nb)
    
    FST = FastShenFourierTransform(N, L, MPI)
    X = FST.get_local_mesh(ST)
    x0, x1, x2 = FST.get_mesh_dims(ST)

    dealias = None
    if not config.dealias == "3/2-rule":
        dealias = FST.get_dealias_filter()

    U     = empty((3,)+FST.real_shape(), dtype=float)
    U_hat = empty((3,)+FST.complex_shape(), dtype=complex)
    P     = empty(FST.real_shape(), dtype=float)
    P_hat = empty(FST.complex_shape(), dtype=complex)

    U0      = empty((3,)+FST.real_shape(), dtype=float)
    U_hat0  = empty((3,)+FST.complex_shape(), dtype=complex)
    
    # We're solving for:
    u = U_hat0[0]
    g = empty(FST.complex_shape(), dtype=complex)

    H        = empty((3,)+FST.real_shape(), dtype=float)
    H0       = empty((3,)+FST.real_shape(), dtype=float)
    H1       = empty((3,)+FST.real_shape(), dtype=float)
    H_hat    = empty((3,)+FST.complex_shape(), dtype=complex)
    H_hat0   = empty((3,)+FST.complex_shape(), dtype=complex)
    H_hat1   = empty((3,)+FST.complex_shape(), dtype=complex)
    
    U_tmp   = empty((3,)+FST.real_shape(), dtype=float)
    U_tmp2  = empty((3,)+FST.real_shape(), dtype=float)
    F_tmp   = empty((3,)+FST.complex_shape(), dtype=complex)
    F_tmp2  = empty((3,)+FST.complex_shape(), dtype=complex)

    dU      = empty((3,)+FST.complex_shape(), dtype=complex)
    hv      = empty(FST.complex_shape(), dtype=complex)
    hg      = empty(FST.complex_shape(), dtype=complex)
    diff0   = empty((3,)+FST.complex_shape(), dtype=complex)
    Source  = empty((3,)+FST.real_shape(), dtype=float) 
    Sk      = empty((3,)+FST.complex_shape(), dtype=complex)         

    K = FST.get_scaled_local_wavenumbermesh()
    K2 = K[1]*K[1]+K[2]*K[2]
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)
                
    del kwargs 
    return locals()


def setupShenMHD(N, L, MPI, float, complex, **kwargs):
    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad="GL")
    SN = ShenNeumannBasis(quad="GC")

    Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
    Nu = N[0]-2   # Number of velocity modes in Shen basis
    Nq = N[0]-3   # Number of pressure modes in Shen basis
    u_slice = slice(0, Nu)
    p_slice = slice(1, Nu)

    FST = FastShenFourierTransform(N, L, MPI)
    X = FST.get_local_mesh(ST)
    x0, x1, x2 = FST.get_mesh_dims(ST)

    dealias = None
    if not config.dealias == "3/2-rule":
        dealias = FST.get_dealias_filter()

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
    
    K = FST.get_scaled_local_wavenumbermesh()
    K2 = K[1]*K[1]+K[2]*K[2]
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)
    
    del kwargs 
    return locals()

def setupShenGeneralBCs(N, L, MPI, float, complex, **kwargs):
    # Get points and weights for Chebyshev weighted integrals
    BC1 = array([1,0,0, 1,0,0])
    BC2 = array([0,1,0, 0,1,0])
    ST = ShenBasis(BC1, quad="GL")
    SN = ShenBasis(BC2, quad="GC")
    Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
    Nu = N[0]-2   # Number of velocity modes in Shen basis
    Nq = N[0]-3   # Number of pressure modes in Shen basis
    u_slice = slice(0, Nu)
    p_slice = slice(1, Nu)

    FST = FastShenFourierTransform(N, L, MPI)
    X = FST.get_local_mesh(ST)
    x0, x1, x2 = FST.get_mesh_dims(ST)

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
    
    K = FST.get_scaled_local_wavenumbermesh()
    K2 = K[1]*K[1]+K[2]*K[2]
    K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)
    
    del kwargs 
    return locals()

class FastShenFourierTransform(slab_FFT):
    
    def __init__(self, N, L, MPI):        
        slab_FFT.__init__(self, N, L, MPI, "double")
        # Initialize intermediate MPI work arrays
        self.U_mpi2  = empty((self.num_processes, self.Np[0], self.Np[1], self.N[2]))
        self.UT      = empty((3, self.N[0], self.Np[1], self.N[2]))
        self.Upad_hatT = empty(self.complex_shape_padded_T(), dtype=self.complex)
        
    def complex_shape_padded_T(self):
        """The local shape of the transposed complex data padded in x and z directions"""
        return (self.Np[0], 3*self.N[1]/2, 3*self.N[2]/4+1)

    def real_shape_padded(self):
        """The local shape of the real data"""
        return (self.Np[0], 3*self.N[1]/2, 3*self.N[2]/2)
    
    def complex_shape_padded(self):
        return (self.N[0], 3*self.Np[1]/2, 3*self.N[2]/4+1)
    
    def copy_to_padded(self, fu, fp):
        fp[:, :self.N[1]/2, :self.Nf] = fu[:, :self.N[1]/2]
        fp[:, -(self.N[1]/2):, :self.Nf] = fu[:, self.N[1]/2:]
        return fp
    
    def copy_from_padded(self, fp, fu):
        fu[:, :self.N[1]/2] = fp[:, :self.N[1]/2, :self.Nf]
        fu[:, self.N[1]/2:] = fp[:, -(self.N[1]/2):, :self.Nf]
        return fu
    
    def get_mesh_dims(self, ST):
        return [self.get_mesh_dim(ST, i) for i in range(3)]
        
    def get_mesh_dim(self, ST, d):
        if d == 0:
            return ST.points_and_weights(self.N[0])[0]
        elif d == 1:
            return arange(self.N[1], dtype=self.float)*self.L[1]/self.N[1]
        elif d == 2:
            return arange(self.N[2], dtype=self.float)*self.L[2]/self.N[2]
    
    def get_local_mesh(self, ST):
        x0, x1, x2 = self.get_mesh_dims(ST)

        # Get grid for velocity points
        X = array(meshgrid(x0[self.rank*self.Np[0]:(self.rank+1)*self.Np[0]], 
                           x1, x2, indexing='ij'), dtype=self.float)
        return X
    
    def get_local_wavenumbermesh(self):
        kx = arange(self.N[0]).astype(self.float)
        ky = fftfreq(self.N[1], 1./self.N[1])[self.rank*self.Np[1]:(self.rank+1)*self.Np[1]]
        kz = fftfreq(self.N[2], 1./self.N[2])[:self.Nf]
        kz[-1] *= -1.0
        return array(meshgrid(kx, ky, kz, indexing='ij'), dtype=self.float) 

    def get_scaled_local_wavenumbermesh(self):
        K = self.get_local_wavenumbermesh()

        # scale with physical mesh size. 
        # This takes care of mapping the physical domain to a computational cube of size (2, 2pi, 2pi)
        # Note that first direction cannot be different from 2 (yet)
        Lp = array([2, 2*pi, 2*pi])/self.L
        for i in range(3):
            K[i] *= Lp[i] 
        return K
    
    def get_dealias_filter(self):
        """Filter for dealiasing nonlinear convection"""
        K = self.get_local_wavenumbermesh()
        kmax = 2./3.*(self.N/2+1)
        dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                        (abs(K[2]) < kmax[2]), dtype=uint8)
        return dealias

    def ifst_padded(self, fu, u, S):
        """Inverse Shen transform of x-direction, Fourier in y and z.
        
        fu is padded with zeros using the 3/2 rule before transforming to real space
        """
        self.Uc_hat[:] = S.ifst(fu, self.Uc_hat)
        self.comm.Alltoall([self.Uc_hat, self.mpitype], [self.U_mpi, self.mpitype])
        self.Uc_hatT[:] = rollaxis(self.U_mpi, 1).reshape(self.complex_shape_T())     
        self.Upad_hatT[:] = 0
        self.Upad_hatT = self.copy_to_padded(self.Uc_hatT, self.Upad_hatT)
        u[:] = irfft2(1.5**2*self.Upad_hatT, axes=(1,2))
        return u

    def fst_padded(self, u, fu, S):
        """Fast Shen transform of x-direction, Fourier transform of y and z
        
        u is of shape real_shape_padded. The output, fu, is normal complex_shape
        """   
        self.Upad_hatT[:] = rfft2(u, axes=(1,2))
        # cut the highest wavenumbers     
        self.Uc_hatT = self.copy_from_padded(self.Upad_hatT, self.Uc_hatT)
        self.U_mpi[:] = rollaxis(self.Uc_hatT.reshape(self.complex_shape_I()), 1)
        self.comm.Alltoall([self.U_mpi, self.mpitype], [self.Uc_hat, self.mpitype])
        fu = S.fst(self.Uc_hat/1.5**2, fu)
        return fu

    def fss_padded(self, u, fu, S):
        """Fast padded Shen scalar product of x-direction, Fourier transform of y and z
        
        u is of shape real_shape_padded. The output, fu, is normal complex_shape
        """        
        self.Upad_hatT[:] = rfft2(u, axes=(1,2))
        # cut the highest wavenumbers     
        self.Uc_hatT = self.copy_from_padded(self.Upad_hatT, self.Uc_hatT)
        self.U_mpi[:] = rollaxis(self.Uc_hatT.reshape(self.complex_shape_I()), 1)
        self.comm.Alltoall([self.U_mpi, self.mpitype], [self.Uc_hat, self.mpitype])
        fu = S.fastShenScalar(self.Uc_hat/1.5**2, fu)
        return fu
    
    def ifct_padded(self, fu, u, S):
        """Inverse Cheb transform of x-direction, Fourier in y and z
        
        fu is padded with zeros using the 3/2 rule before transforming to real space
        """
        self.Uc_hat[:] = S.ifct(fu, self.Uc_hat)
        self.comm.Alltoall([self.Uc_hat, self.mpitype], [self.U_mpi, self.mpitype])
        self.Uc_hatT[:] = rollaxis(self.U_mpi, 1).reshape(self.complex_shape_T())    
        self.Upad_hatT[:] = 0
        self.Upad_hatT = self.copy_to_padded(self.Uc_hatT, self.Upad_hatT)
        u[:] = irfft2(1.5**2*self.Upad_hatT, axes=(1,2))
        return u

    def fct_padded(self, u, fu, S):
        """Fast Shen transform of x-direction, Fourier transform of y and z
        
        u is of shape real_shape_padded. The output, fu, is normal complex_shape
        """        
        self.Upad_hatT[:] = rfft2(u, axes=(1,2))
        # cut the highest wavenumbers     
        self.Uc_hatT = self.copy_from_padded(self.Upad_hatT, self.Uc_hatT)
        self.U_mpi[:] = rollaxis(self.Uc_hatT.reshape(self.complex_shape_I()), 1)
        self.comm.Alltoall([self.U_mpi, self.mpitype], [self.Uc_hat, self.mpitype])
        fu = S.fct(self.Uc_hat/1.5**2, fu)
        return fu
    
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

    def fft(self, u, fu):
        """Fast Fourier transform of y and z"""
        self.Uc_hatT[:] = rfft2(u, axes=(1,2))
        self.U_mpi[:] = rollaxis(self.Uc_hatT.reshape(self.complex_shape_I()), 1)
        self.comm.Alltoall([self.U_mpi, self.mpitype], [fu, self.mpitype])
        return fu
    
    def ifft(self, fu, u):
        """Inverse Fourier transforms in y and z"""
        self.comm.Alltoall([fu, self.mpitype], [self.U_mpi, self.mpitype])
        self.Uc_hatT[:] = rollaxis(self.U_mpi, 1).reshape(self.complex_shape_T())
        u[:] = irfft2(self.Uc_hatT, axes=(1,2))
        return u
    
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

setup = {"IPCS": setupShen,
         "IPCSR": setupShen,
         "KMM": setupShenKMM,
         "KMMRK3": setupShenKMM,
         "IPCS_MHD": setupShenMHD,
         "IPCS_GeneralBCs": setupShenGeneralBCs}[config.solver]        
