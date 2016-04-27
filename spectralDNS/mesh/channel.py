__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralDNS import config
from mpiFFT4py import *
from ..shen.shentransform import ShenDirichletBasis, ShenNeumannBasis, ShenBiharmonicBasis, SFTc
from ..shenGeneralBCs.shentransform import ShenBasis
from numpy import array, ndarray, sum, meshgrid, mgrid, where, abs, pi, uint8, rollaxis, arange, conj

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
    F_tmp   = empty((3,)+FST.complex_shape(), dtype=complex)
    F_tmp2  = empty((3,)+FST.complex_shape(), dtype=complex)

    dU      = empty((3,)+FST.complex_shape(), dtype=complex)

    H_hat    = empty((3,)+FST.complex_shape(), dtype=complex)
    H_hat0   = empty((3,)+FST.complex_shape(), dtype=complex)
    H_hat1   = empty((3,)+FST.complex_shape(), dtype=complex)

    diff0   = empty((3,)+FST.complex_shape(), dtype=complex)
    Source  = empty((3,)+FST.real_shape(), dtype=float) 
    Sk      = empty((3,)+FST.complex_shape(), dtype=complex) 
    
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

    U     = empty((3,)+FST.real_shape(), dtype=float)
    U_hat = empty((3,)+FST.complex_shape(), dtype=complex)
    P     = empty(FST.real_shape(), dtype=float)
    P_hat = empty(FST.complex_shape(), dtype=complex)

    U0      = empty((3,)+FST.real_shape(), dtype=float)
    U_hat0  = empty((3,)+FST.complex_shape(), dtype=complex)
    
    # We're solving for:
    u = U_hat0[0]
    g = empty(FST.complex_shape(), dtype=complex)

    H_hat    = empty((3,)+FST.complex_shape(), dtype=complex)
    H_hat0   = empty((3,)+FST.complex_shape(), dtype=complex)
    H_hat1   = empty((3,)+FST.complex_shape(), dtype=complex)
    
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
    
    def __init__(self, N, L, MPI, padsize=1.5):
        slab_FFT.__init__(self, N, L, MPI, "double", padsize=padsize)
        
    def complex_shape_padded_T(self):
        """The local shape of the transposed complex data padded in x and z directions"""
        return (self.Np[0], int(self.padsize*self.N[1]), int(self.padsize*self.N[2]/2+1))

    def real_shape_padded(self):
        """The local shape of the real data"""
        return (self.Np[0], int(self.padsize*self.N[1]), int(self.padsize*self.N[2]))
    
    def complex_shape_padded(self):
        return (self.N[0], int(self.padsize*self.Np[1]), int(self.padsize*self.N[2]/2+1))
    
    def get_mesh_dims(self, ST):
        return [self.get_mesh_dim(ST, i) for i in range(3)]
        
    def real_local_slice(self, padded=False):
        if padded:
            return (slice(self.rank*self.Np[0], (self.rank+1)*self.Np[0], 1),
                    slice(0, int(self.padsize*self.N[1]), 1), 
                    slice(0, int(self.padsize*self.N[2]), 1))
        else:
            return (slice(self.rank*self.Np[0], (self.rank+1)*self.Np[0], 1),
                    slice(0, self.N[1], 1), 
                    slice(0, self.N[2], 1))
    
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
    
    def copy_to_padded(self, fu, fp):
        fp[:, :self.N[1]/2, :self.Nf] = fu[:, :self.N[1]/2]
        fp[:, -(self.N[1]/2):, :self.Nf] = fu[:, self.N[1]/2:]
        return fp
    
    def copy_from_padded(self, fp, fu):
        fu[:] = 0
        fu[:, :self.N[1]/2] = fp[:, :self.N[1]/2, :self.Nf]
        fu[:, self.N[1]/2:] = fp[:, -(self.N[1]/2):, :self.Nf]
        return fu
    
    def fss(self, u, fu, S, dealias=None):
        """Fast Shen scalar product of x-direction, Fourier transform of y and z"""
        
        # Intermediate work arrays
        Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0)]
        
        if not dealias == '3/2-rule':

            Uc_hatT[:] = rfft2(u, axes=(1,2))
            Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
            self.comm.Alltoall([Uc_mpi, self.mpitype], [Uc_hat, self.mpitype])
            fu = S.fastShenScalar(Uc_hat, fu)
            
        else:
            Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
            
            Upad_hatT[:] = rfft2(u/self.padsize**2, axes=(1,2))
            Uc_hatT = self.copy_from_padded(Upad_hatT, Uc_hatT)
            Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
            self.comm.Alltoall([Uc_mpi, self.mpitype], [Uc_hat, self.mpitype])
            fu = S.fastShenScalar(Uc_hat, fu)
            
        return fu

    def ifst(self, fu, u, S, dealias=None):
        """Inverse Shen transform of x-direction, Fourier in y and z"""
        
        Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0)]
        if dealias == '2/3-rule' and self.dealias.shape == (0,):
            self.dealias = self.get_dealias_filter()

        if not dealias == '3/2-rule':
            if dealias == '2/3-rule':
                fu *= self.dealias
                
            Uc_hat[:] = S.ifst(fu, Uc_hat)
            self.comm.Alltoall([Uc_hat, self.mpitype], [Uc_mpi, self.mpitype])
            Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
            u[:] = irfft2(Uc_hatT, axes=(1,2))
        
        else:
            Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
            
            Uc_hat[:] = S.ifst(fu, Uc_hat)
            self.comm.Alltoall([Uc_hat, self.mpitype], [Uc_mpi, self.mpitype])
            Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())     
            Upad_hatT = self.copy_to_padded(Uc_hatT, Upad_hatT)
            u[:] = irfft2(self.padsize**2*Upad_hatT, axes=(1,2))

        return u

    def fst(self, u, fu, S, dealias=None):
        """Fast Shen transform of x-direction, Fourier transform of y and z"""
        
        # Intermediate work arrays
        Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0)]

        if not dealias == '3/2-rule':
            Uc_hatT[:] = rfft2(u, axes=(1,2))
            Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
            self.comm.Alltoall([Uc_mpi, self.mpitype], [Uc_hat, self.mpitype])
            fu = S.fst(Uc_hat, fu)

        else:
            Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
            
            Upad_hatT[:] = rfft2(u/self.padsize**2, axes=(1,2))
            Uc_hatT = self.copy_from_padded(Upad_hatT, Uc_hatT)
            Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
            self.comm.Alltoall([Uc_mpi, self.mpitype], [Uc_hat, self.mpitype])
            fu = S.fst(Uc_hat, fu)

        return fu

    def fft(self, u, fu):
        """Fast Fourier transform of y and z"""
        # Intermediate work arrays
        Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]        
        Uc_hatT[:] = rfft2(u, axes=(1,2))
        Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
        self.comm.Alltoall([Uc_mpi, self.mpitype], [fu, self.mpitype])
        return fu
    
    def ifft(self, fu, u):
        """Inverse Fourier transforms in y and z"""
        Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]        
        self.comm.Alltoall([fu, self.mpitype], [Uc_mpi, self.mpitype])
        Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
        u[:] = irfft2(Uc_hatT, axes=(1,2))
        return u
    
    def fct(self, u, fu, S, dealias=None):
        """Fast Cheb transform of x-direction, Fourier transform of y and z"""
        
        # Intermediate work arrays
        Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0)]

        if not dealias == '3/2-rule':
            Uc_hatT[:] = rfft2(u, axes=(1,2))
            Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
            self.comm.Alltoall([Uc_mpi, self.mpitype], [Uc_hat, self.mpitype])
            fu = S.fct(Uc_hat, fu)
        
        else:
            Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
            Upad_hatT[:] = rfft2(u/self.padsize**2, axes=(1,2))
            Uc_hatT = self.copy_from_padded(Upad_hatT, Uc_hatT)
            Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
            self.comm.Alltoall([Uc_mpi, self.mpitype], [Uc_hat, self.mpitype])
            fu = S.fct(Uc_hat, fu)

        return fu

    def ifct(self, fu, u, S, dealias=None):
        """Inverse Cheb transform of x-direction, Fourier in y and z"""
        
        # Intermediate work arrays
        Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0)]

        if dealias == '2/3-rule' and self.dealias.shape == (0,):
            self.dealias = self.get_dealias_filter()
        
        if not dealias == '3/2-rule':
            if dealias == '2/3-rule':
                fu *= self.dealias

            Uc_hat[:] = S.ifct(fu, Uc_hat)
            self.comm.Alltoall([Uc_hat, self.mpitype], [Uc_mpi, self.mpitype])
            Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
            u[:] = irfft2(Uc_hatT, axes=(1,2))
        
        else:
            Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
            Uc_hat[:] = S.ifct(fu, Uc_hat)
            self.comm.Alltoall([Uc_hat, self.mpitype], [Uc_mpi, self.mpitype])
            Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())    
            Upad_hatT = self.copy_to_padded(Uc_hatT, Upad_hatT)
            u[:] = irfft2(self.padsize**2*Upad_hatT, axes=(1,2))

        return u

    def fct0(self, u, fu, S):
        """Fast Cheb transform of x-direction. No FFT, just align data in x-direction and do fct."""
        U_mpi2 = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.N[2]), self.float, 0)]
        UT = self.work_arrays[((self.N[0], self.Np[1], self.N[2]), self.float, 0)]
        
        U_mpi2[:] = rollaxis(u.reshape(self.Np[0], self.num_processes, self.Np[1], self.N[2]), 1)
        self.comm.Alltoall([U_mpi2, self.mpitype], [UT, self.mpitype])
        fu = S.fct(UT, fu)
        return fu

    def ifct0(self, fu, u, S):
        """Fast Cheb transform of x-direction. No FFT, just align data in x-direction and do ifct"""
        U_mpi2 = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.N[2]), self.float, 0)]
        UT = self.work_arrays[((self.N[0], self.Np[1], self.N[2]), self.float, 0)]

        UT = S.ifct(fu, UT)
        self.comm.Alltoall([UT, self.mpitype], [U_mpi2, self.mpitype])
        u[:] = rollaxis(U_mpi2, 1).reshape(u.shape)
        return u
    
    def chebDerivative_3D0(self, fj, u0, S):
        UT = self.work_arrays[((2, self.N[0], self.Np[1], self.N[2]), self.float, 0)]

        UT[0] = self.fct0(fj, UT[0], S)
        UT[1] = SFTc.chebDerivativeCoefficients_3D(UT[0], UT[1]) 
        u0[:] = self.ifct0(UT[1], u0, S)
        return u0

setup = {"IPCS": setupShen,
         "IPCSR": setupShen,
         "KMM": setupShenKMM,
         "KMMRK3": setupShenKMM,
         "IPCS_MHD": setupShenMHD,
         "IPCS_GeneralBCs": setupShenGeneralBCs}[config.solver]        
