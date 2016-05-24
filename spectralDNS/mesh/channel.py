__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpiFFT4py import *
from ..shen.shentransform import ShenDirichletBasis, ShenNeumannBasis, ShenBiharmonicBasis, SFTc
from ..shenGeneralBCs.shentransform import ShenBasis
from numpy import array, ndarray, sum, meshgrid, mgrid, where, abs, pi, uint8, rollaxis, arange, conj

__all__ = ['setup']

def setupShen(N, L, MPI, float, complex, config, **kwargs):
    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad="GL")
    SN = ShenNeumannBasis(quad="GC")

    Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
    Nu = N[0]-2   # Number of velocity modes in Shen basis
    Nq = N[0]-3   # Number of pressure modes in Shen basis
    u_slice = slice(0, Nu)
    p_slice = slice(1, Nu)
    
    FST = FastShenFourierTransform(N, L, MPI, dealias_cheb=config.dealias_cheb)
    
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
    
    work = work_arrays()

    del kwargs 
    return locals()


def setupShenKMM(N, L, MPI, float, complex, config, **kwargs):

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad="GL")
    SB = ShenBiharmonicBasis(quad="GL")

    Nu = N[0]-2   # Number of velocity modes in Shen basis
    Nb = N[0]-4   # Number of velocity modes in Shen biharmonic basis
    u_slice = slice(0, Nu)
    v_slice = slice(0, Nb)
    
    FST = FastShenFourierTransform(N, L, MPI, dealias_cheb=config.dealias_cheb)
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

    work = work_arrays()
                
    del kwargs 
    return locals()


def setupShenMHD(N, L, MPI, float, complex, config, **kwargs):
    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad="GL")
    SN = ShenNeumannBasis(quad="GC")

    Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
    Nu = N[0]-2   # Number of velocity modes in Shen basis
    Nq = N[0]-3   # Number of pressure modes in Shen basis
    u_slice = slice(0, Nu)
    p_slice = slice(1, Nu)

    FST = FastShenFourierTransform(N, L, MPI, dealias_cheb=config.dealias_cheb)
    X = FST.get_local_mesh(ST)
    x0, x1, x2 = FST.get_mesh_dims(ST)

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

    work = work_arrays()
    
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

    work = work_arrays()
    
    del kwargs 
    return locals()

class FastShenFourierTransform(slab_FFT):
    
    def __init__(self, N, L, MPI, padsize=1.5, dealias_cheb=False):
        slab_FFT.__init__(self, N, L, MPI, "double", padsize=padsize)
        self.dealias_cheb = dealias_cheb
        
    def complex_shape_padded_T(self):
        """The local shape of the transposed complex data padded in x and z directions"""
        if self.dealias_cheb:
            return (int(self.padsize*self.Np[0]), int(self.padsize*self.N[1]), int(self.padsize*self.N[2]/2+1))
        else:
            return (self.Np[0], int(self.padsize*self.N[1]), int(self.padsize*self.N[2]/2+1))

    def real_shape_padded(self):
        """The local shape of the real data"""
        if self.dealias_cheb:
            return (int(self.padsize*self.Np[0]), int(self.padsize*self.N[1]), int(self.padsize*self.N[2]))
        else:
            return (self.Np[0], int(self.padsize*self.N[1]), int(self.padsize*self.N[2]))
    
    def complex_shape_padded(self):
        if self.dealias_cheb:
            return (int(self.padsize*self.N[0]), int(self.padsize*self.Np[1]), int(self.padsize*self.N[2]/2+1))
        else:
            return (self.N[0], int(self.padsize*self.Np[1]), int(self.padsize*self.N[2]/2+1))
    
    def get_mesh_dims(self, ST):
        return [self.get_mesh_dim(ST, i) for i in range(3)]
        
    def real_local_slice(self, padded=False):
        if padded:
            if self.dealias_cheb:
                return (slice(self.rank*self.padsize*self.Np[0], (self.rank+1)*self.padsize*self.Np[0], 1),
                        slice(0, int(self.padsize*self.N[1]), 1), 
                        slice(0, int(self.padsize*self.N[2]), 1))
            else:
                return (slice(self.rank*self.Np[0], (self.rank+1)*self.Np[0], 1),
                        slice(0, int(self.padsize*self.N[1]), 1), 
                        slice(0, int(self.padsize*self.N[2]), 1))
            
        else:
            return (slice(self.rank*self.Np[0], (self.rank+1)*self.Np[0], 1),
                    slice(0, self.N[1], 1), 
                    slice(0, self.N[2], 1))
    
    def global_complex_shape_padded(self):
        """Global size of problem in complex wavenumber space"""
        if self.dealias_cheb:
            return (int(self.padsize*self.N[0]), int(self.padsize*self.N[1]), self.Nfp)
        else:
            return (self.N[0], int(self.padsize*self.N[1]), self.Nfp)
    
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
        kmax = 2./3.*(self.N/2)
        kmax[0] = 2./3*self.N[0] if self.dealias_cheb else self.N[0]         
        dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                        (abs(K[2]) < kmax[2]), dtype=uint8)
        return dealias
    
    def copy_to_padded(self, fu, fp):
        fp[:, self.ks, :self.Nf] = fu[:, :]
        # Make padding symmetrical
        #fp[:, -self.N[1]/2, 0] *= 0.5
        #fp[:, -self.N[1]/2, self.N[2]/2] *= 0.5
        #fp[:, self.N[1]/2, 0] = fp[:, -self.N[1]/2, 0]
        #fp[:, self.N[1]/2, self.N[2]/2] = fp[:, -self.N[1]/2, self.N[2]/2]
        return fp
    
    def copy_from_padded(self, fp, fu):
        fu[:] = fp[:, self.ks, :self.Nf]
        #fu[:, self.N[1]/2, 0] *= 2 # Because of symmetrical padding
        #fu[:, self.N[1]/2, self.N[2]/2] *= 2
        return fu
    
    def copy_from_padded_z(self, fp, fu):
        fu[:] = fp[:, :, :self.Nf]
        return fu
    
    def copy_to_padded_x(self, fu, fp):
        fp[:self.N[0]] = fu[:self.N[0]]
        return fp
    
    def copy_to_padded_y(self, fu, fp):
        fp[:, self.ks] = fu[:]
        #fp[:, -self.N[1]/2, 0] *= 0.5
        #fp[:, -self.N[1]/2, self.N[2]/2] *= 0.5
        #fp[:, self.N[1]/2, 0] = fp[:, -self.N[1]/2, 0]
        #fp[:, self.N[1]/2, self.N[2]/2] = fp[:, -self.N[1]/2, self.N[2]/2]
        return fp
    
    def copy_to_padded_z(self, fu, fp):
        fp[:, :, :self.Nf] = fu[:]
        return fp

    def fss(self, u, fu, S, dealias=None):
        """Fast Shen scalar product of x-direction, Fourier transform of y and z"""
        
        # Intermediate work arrays
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0)]
        Uc_mpi  = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
        
        if not dealias == '3/2-rule':

            Uc_hatT[:] = rfft2(u, axes=(1,2))
            Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
            self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
            fu = S.fastShenScalar(Uc_hat, fu)
            
        else:
            if not self.dealias_cheb:
                Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
                Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0)]
                
                Upad_hatT[:] = rfft(u/self.padsize, axis=2)
                Upad_hat_z = self.copy_from_padded_z(Upad_hatT, Upad_hat_z)
                Upad_hat_z[:] = fft(Upad_hat_z/self.padsize, axis=1)                
                Uc_hatT = self.copy_from_padded(Upad_hat_z, Uc_hatT)                
                
                Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
                fu = S.fastShenScalar(Uc_hat, fu)
            
            else:
                assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"
                assert u.shape == self.real_shape_padded()
                
                # Intermediate work arrays required for transform
                Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0)]
                Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1)]
                Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0)]
                Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0)]
                Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0)]

                # Do ffts and truncation in the padded y and z directions
                Upad_hat3[:] = rfft(u/self.padsize, axis=2)
                Upad_hat2 = self.copy_from_padded_z(Upad_hat3, Upad_hat2)
                Upad_hat2[:] = fft(Upad_hat2/self.padsize, axis=1)
                Upad_hat1 = self.copy_from_padded(Upad_hat2, Upad_hat1)
                
                # Transpose and commuincate data
                U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
                U_mpi[:] = rollaxis(Upad_hat1.reshape(self.complex_shape_padded_I()), 1)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Upad_hat, self.mpitype])
                
                # Perform fss of data in x-direction
                Upad_hat0 = S.fastShenScalar(Upad_hat, Upad_hat0)
                
                # Truncate to original complex shape
                fu[:] = Upad_hat0[:self.N[0]]
            
        return fu

    def ifst(self, fu, u, S, dealias=None):
        """Inverse Shen transform of x-direction, Fourier in y and z"""
        
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0)]
        Uc_mpi  = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
                
        if dealias == '2/3-rule' and self.dealias.shape == (0,):
            self.dealias = self.get_dealias_filter()

        if not dealias == '3/2-rule':
            if dealias == '2/3-rule':
                fu *= self.dealias
                
            Uc_hat = S.ifst(fu, Uc_hat)
            self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
            Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
            u[:] = irfft2(Uc_hatT, axes=(1,2))
        
        else:
            if not self.dealias_cheb:
                Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
                Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0)]
                
                Uc_hat = S.ifst(fu, Uc_hat)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
                Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())     
                
                Upad_hat_z = self.copy_to_padded_y(Uc_hatT, Upad_hat_z)
                Upad_hat_z[:] = ifft(self.padsize*Upad_hat_z, axis=1)
                Upad_hatT = self.copy_to_padded_z(Upad_hat_z, Upad_hatT)
                u[:] = irfft(self.padsize*Upad_hatT, axis=2)
                
            else:
                assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"            
            
                # Intermediate work arrays required for transform
                Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0)]
                Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1)]
                Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0)]
                Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0)]
                Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0)]
                
                # Expand in x-direction and perform ifst
                Upad_hat0 = self.copy_to_padded_x(fu, Upad_hat0)
                Upad_hat = S.ifst(Upad_hat0, Upad_hat) 
                
                # Communicate to distribute first dimension (like Fig. 2b but padded in x-dir)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Upad_hat, self.mpitype])
                
                # Transpose data and pad in y-direction before doing ifft. Now data is padded in x and y 
                U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
                Upad_hat1[:] = rollaxis(U_mpi, 1).reshape(Upad_hat1.shape)
                Upad_hat2 = self.copy_to_padded_y(Upad_hat1, Upad_hat2)
                Upad_hat2[:] = ifft(Upad_hat2*self.padsize, axis=1)
                
                # pad in z-direction and perform final irfft
                Upad_hat3 = self.copy_to_padded_z(Upad_hat2, Upad_hat3)
                u[:] = irfft(Upad_hat3*self.padsize, axis=2)

        return u

    def fst(self, u, fu, S, dealias=None):
        """Fast Shen transform of x-direction, Fourier transform of y and z"""
        
        # Intermediate work arrays
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0)]
        Uc_mpi  = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
        
        if not dealias == '3/2-rule':
            Uc_hatT[:] = rfft2(u, axes=(1,2))
            Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
            self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
            fu = S.fst(Uc_hat, fu)

        else:
            if not self.dealias_cheb:
                Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
                Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0)]
                
                Upad_hatT[:] = rfft(u/self.padsize, axis=2)
                Upad_hat_z = self.copy_from_padded_z(Upad_hatT, Upad_hat_z)
                Upad_hat_z[:] = fft(Upad_hat_z/self.padsize, axis=1)                
                Uc_hatT = self.copy_from_padded(Upad_hat_z, Uc_hatT)
                
                Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
                fu = S.fst(Uc_hat, fu)
            else:
                assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"
                assert u.shape == self.real_shape_padded()
                
                # Intermediate work arrays required for transform
                Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0)]
                Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1)]
                Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0)]
                Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0)]
                Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0)]

                # Do ffts and truncation in the padded y and z directions
                Upad_hat3[:] = rfft(u/self.padsize, axis=2)
                Upad_hat2 = self.copy_from_padded_z(Upad_hat3, Upad_hat2)
                Upad_hat2[:] = fft(Upad_hat2/self.padsize, axis=1)
                Upad_hat1 = self.copy_from_padded(Upad_hat2, Upad_hat1)
                
                # Transpose and commuincate data
                U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
                U_mpi[:] = rollaxis(Upad_hat1.reshape(self.complex_shape_padded_I()), 1)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Upad_hat, self.mpitype])
                
                # Perform fct of data in x-direction
                Upad_hat0 = S.fst(Upad_hat, Upad_hat0)
                
                # Truncate to original complex shape
                fu[:] = Upad_hat0[:self.N[0]]

        return fu
    
    def fct(self, u, fu, S, dealias=None):
        """Fast Cheb transform of x-direction, Fourier transform of y and z"""
        
        # Intermediate work arrays
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0)]
        Uc_mpi  = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
        
        if not dealias == '3/2-rule':
            Uc_hatT[:] = rfft2(u, axes=(1,2))
            Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
            self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
            fu = S.fct(Uc_hat, fu)
        
        else:
            if not self.dealias_cheb:
                Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
                Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0)]
                
                Upad_hatT[:] = rfft(u/self.padsize, axis=2)
                Upad_hat_z = self.copy_from_padded_z(Upad_hatT, Upad_hat_z)
                Upad_hat_z[:] = fft(Upad_hat_z/self.padsize, axis=1)
                Uc_hatT = self.copy_from_padded(Upad_hat_z, Uc_hatT)
                
                Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.complex_shape_I()), 1)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
                fu = S.fct(Uc_hat, fu)

            else:
                assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"
                assert u.shape == self.real_shape_padded()
                                
                # Intermediate work arrays required for transform
                Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0)]
                Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1)]
                Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0)]
                Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0)]
                Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0)]

                # Do ffts and truncation in the padded y and z directions
                Upad_hat3[:] = rfft(u/self.padsize, axis=2)
                Upad_hat2 = self.copy_from_padded_z(Upad_hat3, Upad_hat2)
                Upad_hat2[:] = fft(Upad_hat2/self.padsize, axis=1)
                Upad_hat1 = self.copy_from_padded(Upad_hat2, Upad_hat1)
                
                # Transpose and commuincate data
                U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
                U_mpi[:] = rollaxis(Upad_hat1.reshape(self.complex_shape_padded_I()), 1)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Upad_hat, self.mpitype])
                
                # Perform fct of data in x-direction
                Upad_hat0 = S.fct(Upad_hat, Upad_hat0)
                
                # Truncate to original complex shape
                fu[:] = Upad_hat0[:self.N[0]]            

        return fu

    def ifct(self, fu, u, S, dealias=None):
        """Inverse Cheb transform of x-direction, Fourier in y and z"""
        
        # Intermediate work arrays
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0)]
        Uc_mpi  = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
        
        if dealias == '2/3-rule' and self.dealias.shape == (0,):
            self.dealias = self.get_dealias_filter()
        
        if not dealias == '3/2-rule':
            if dealias == '2/3-rule':
                fu *= self.dealias

            Uc_hat[:] = S.ifct(fu, Uc_hat)
            self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_mpi, self.mpitype])
            Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
            u[:] = irfft2(Uc_hatT, axes=(1,2))
        
        else:
            if not self.dealias_cheb:
                Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
                Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0)]
                
                Uc_hat[:] = S.ifct(fu, Uc_hat)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
                Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())

                Upad_hat_z = self.copy_to_padded_y(Uc_hatT, Upad_hat_z)
                Upad_hat_z[:] = ifft(self.padsize*Upad_hat_z, axis=1)
                Upad_hatT = self.copy_to_padded_z(Upad_hat_z, Upad_hatT)
                u[:] = irfft(self.padsize*Upad_hatT, axis=2)
                
            else:
                assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"            
            
                # Intermediate work arrays required for transform
                Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0)]
                Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1)]
                Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0)]
                Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0)]
                Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0)]
                
                # Expand in x-direction and perform ifst
                Upad_hat0 = self.copy_to_padded_x(fu, Upad_hat0)
                Upad_hat = S.ifct(Upad_hat0, Upad_hat)
                
                # Communicate to distribute first dimension (like Fig. 2b but padded in x-dir)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Upad_hat, self.mpitype])
                
                # Transpose data and pad in y-direction before doing ifft. Now data is padded in x and y 
                U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
                Upad_hat1[:] = rollaxis(U_mpi, 1).reshape(Upad_hat1.shape)
                Upad_hat2 = self.copy_to_padded_y(Upad_hat1, Upad_hat2)
                Upad_hat2[:] = ifft(Upad_hat2*self.padsize, axis=1)
                
                # pad in z-direction and perform final irfft
                Upad_hat3 = self.copy_to_padded_z(Upad_hat2, Upad_hat3)
                u[:] = irfft(Upad_hat3*self.padsize, axis=2)

        return u

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
         "IPCS_GeneralBCs": setupShenGeneralBCs}
