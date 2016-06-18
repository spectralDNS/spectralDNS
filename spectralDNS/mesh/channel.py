__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpiFFT4py import *
from ..shen.shentransform import ShenDirichletBasis, ShenNeumannBasis, ShenBiharmonicBasis, SFTc
from ..shenGeneralBCs.shentransform import ShenBasis
from numpy import array, ndarray, sum, meshgrid, mgrid, where, abs, pi, uint8, rollaxis, arange, conj
from collections import defaultdict
from ..optimization import *

__all__ = ['setup']

def setupShen(MPI, config, **kwargs):
    # Get points and weights for Chebyshev weighted integrals
    params = config.params
    ST = ShenDirichletBasis(quad="GL", threads=params.threads, planner_effort=params.planner_effort["dct"])
    SN = ShenNeumannBasis(quad="GC", threads=params.threads, planner_effort=params.planner_effort["dct"])

    Nf = params.N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nq = params.N[0]-3   # Number of pressure modes in Shen basis
    u_slice = slice(0, Nu)
    p_slice = slice(1, Nu)
    
    FST = slab_shen_r2c(params.N, params.L, MPI, threads=params.threads,
                                   communication=params.communication,
                                   planner_effort=params.planner_effort,
                                   dealias_cheb=params.dealias_cheb)
    
    float, complex, mpitype = datatypes("double")
    
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


def setupShenKMM(MPI, config, **kwargs):
    params = config.params

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad="GC", threads=params.threads, planner_effort=params.planner_effort["dct"])
    SB = ShenBiharmonicBasis(quad="GC", threads=params.threads, planner_effort=params.planner_effort["dct"])
    
    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nb = params.N[0]-4   # Number of velocity modes in Shen biharmonic basis
    u_slice = slice(0, Nu)
    v_slice = slice(0, Nb)
    
    FST = slab_shen_r2c(params.N, params.L, MPI, threads=params.threads, 
                        communication=params.communication,
                        planner_effort=params.planner_effort,
                        dealias_cheb=params.dealias_cheb)
    
    float, complex, mpitype = datatypes("double")
    
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


def setupShenMHD(MPI, config, **kwargs):
    params = config.params

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad="GL", threads=params.threads, planner_effort=params.planner_effort["dct"])
    SN = ShenNeumannBasis(quad="GC", threads=params.threads, planner_effort=params.planner_effort["dct"])

    Nf = params.N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nq = params.N[0]-3   # Number of pressure modes in Shen basis
    u_slice = slice(0, Nu)
    p_slice = slice(1, Nu)

    FST = slab_shen_r2c(params.N, params.L, MPI, threads=params.threads, 
                        communication=params.communication,
                        planner_effort=params.planner_effort,
                        dealias_cheb=params.dealias_cheb)
    
    float, complex, mpitype = datatypes("double")
    
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
    
    # FIXME remove work arrays
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

def setupShenGeneralBCs(MPI, config, **kwargs):
    params = config.params
    # Get points and weights for Chebyshev weighted integrals
    BC1 = array([1,0,0, 1,0,0])
    BC2 = array([0,1,0, 0,1,0])
    ST = ShenBasis(BC1, quad="GL", threads=params.threads, planner_effort=params.planner_effort["dct"])
    SN = ShenBasis(BC2, quad="GC", threads=params.threads, planner_effort=params.planner_effort["dct"])
    
    Nf = params.N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nq = params.N[0]-3   # Number of pressure modes in Shen basis
    u_slice = slice(0, Nu)
    p_slice = slice(1, Nu)

    FST = slab_shen_r2c(params.N, params.L, MPI, threads=params.threads, 
                        communication=params.communication,
                        planner_effort=params.planner_effort,
                        dealias_cheb=params.dealias_cheb)
    
    float, complex, mpitype = datatypes("double")
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

    #FIXME Remove work arrays
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

class slab_shen_r2c(slab_r2c):
    
    def __init__(self, N, L, MPI, padsize=1.5, threads=1, communication='Alltoallw', dealias_cheb=False,
                 planner_effort=defaultdict(lambda: "FFTW_MEASURE", {"dct": "FFTW_EXHAUSTIVE"})):
        slab_r2c.__init__(self, N, L, MPI, "double", padsize=padsize, threads=threads,
                          communication=communication, planner_effort=planner_effort)
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
    
    @staticmethod
    @optimizer
    def copy_from_padded(fp, fu, N, axis=0):
        if axis == 1:
            fu[:, :N[1]/2] = fp[:, :N[1]/2, :(N[2]/2+1)]
            fu[:, N[1]/2:] = fp[:, -N[1]/2:, :(N[2]/2+1)]
            #fu[:, self.N[1]/2, 0] *= 2 # Because of symmetrical padding
            #fu[:, self.N[1]/2, self.N[2]/2] *= 2
        elif axis == 2:
            fu[:] = fp[:, :, :(N[2]//2+1)]
            
        return fu
    
    @staticmethod
    @optimizer
    def copy_to_padded(fu, fp, N, axis=0):
        if axis == 0:
            fp[:N[0]] = fu[:N[0]]
        elif axis == 1:
            fp[:, :N[1]/2] = fu[:, :N[1]/2]
            fp[:, -N[1]/2:] = fu[:, N[1]/2:]
        elif axis == 2:
            fp[:, :, :(N[2]/2+1)] = fu[:]        
        return fp
    
    #@profile
    def forward(self, u, fu, fun, dealias=None):
        
        # Intermediate work arrays
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0, False)]
        
        if self.num_processes == 1:
            
            if not dealias == '3/2-rule':
                assert u.shape == self.real_shape()
                
                Uc_hat = rfft2(u, Uc_hat, axes=(1,2), threads=self.threads, planner_effort=self.planner_effort['rfft2'])
                fu = fun(Uc_hat, fu)
                
            else:
                if not self.dealias_cheb:
                    Upad_hat = self.work_arrays[(self.complex_shape_padded(), self.complex, 0, False)]
                    Upad_hat_z = self.work_arrays[((self.N[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0, False)]
                
                    Upad_hat = rfft(u/self.padsize, Upad_hat, overwrite_input=True, axis=2, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                    Upad_hat_z = slab_shen_r2c.copy_from_padded(Upad_hat, Upad_hat_z, self.N, 2)
                    Upad_hat_z = fft(Upad_hat_z/self.padsize, Upad_hat_z, axis=1, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['fft'])   
                    Uc_hat = slab_shen_r2c.copy_from_padded(Upad_hat_z, Uc_hat, self.N, 1)
                    fu = fun(Uc_hat, fu)
                else:
                    # Intermediate work arrays required for transform
                    Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
                    Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1, False)]
                    Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0, False)]
                    Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]

                    # Do ffts and truncation in the padded y and z directions
                    Upad_hat3 = rfft(u/self.padsize, Upad_hat3, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                    Upad_hat2 = slab_shen_r2c.copy_from_padded(Upad_hat3, Upad_hat2, self.N, 2)
                    Upad_hat2 = fft(Upad_hat2/self.padsize, Upad_hat2, axis=1, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['fft'])
                    Upad_hat = slab_shen_r2c.copy_from_padded(Upad_hat2, Upad_hat, self.N, 1)
                    
                    # Perform fst of data in x-direction
                    Upad_hat0 = fun(Upad_hat, Upad_hat0)
                    
                    # Truncate to original complex shape
                    fu[:] = Upad_hat0[:self.N[0]]
            return fu

        if not dealias == '3/2-rule':
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]

            if self.communication == 'alltoall':
                Uc_mpi  = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
                Uc_hatT = rfft2(u, Uc_hatT, axes=(1,2), threads=self.threads, planner_effort=self.planner_effort['rfft2'])
                Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
                
            elif self.communication == 'Alltoallw':
                if len(self._subarraysA) == 0:
                    self._subarraysA, self._subarraysB, self._counts_displs = self.get_subarrays()
                    
                # Do 2 ffts in y-z directions on owned data
                Uc_hatT = rfft2(u, Uc_hatT, axes=(1,2), threads=self.threads, planner_effort=self.planner_effort['rfft2'])

                self.comm.Alltoallw(
                    [Uc_hatT, self._counts_displs, self._subarraysB],
                    [Uc_hat,  self._counts_displs, self._subarraysA])
                
            fu = fun(Uc_hat, fu)
            
        else:
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
            if not self.dealias_cheb:
                Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0, False)]
                Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0, False)]
                
                Upad_hatT = rfft(u/self.padsize, Upad_hatT, overwrite_input=True, axis=2, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                Upad_hat_z = slab_shen_r2c.copy_from_padded(Upad_hatT, Upad_hat_z, self.N, 2)
                Upad_hat_z = fft(Upad_hat_z/self.padsize, Upad_hat_z, axis=1, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['fft'])   
                Uc_hatT = slab_shen_r2c.copy_from_padded(Upad_hat_z, Uc_hatT, self.N, 1)                
                
                if self.communication == 'alltoall':
                    Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
                    self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
                elif self.communication == 'Alltoallw':
                    self.comm.Alltoallw(
                        [Uc_hatT, self._counts_displs, self._subarraysB],
                        [Uc_hat,  self._counts_displs, self._subarraysA])
                    
                fu = fun(Uc_hat, fu)
            
            else:
                assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"
                assert u.shape == self.real_shape_padded()
                
                # Intermediate work arrays required for transform
                Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
                Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1, False)]
                Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
                Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0, False)]
                Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]

                # Do ffts and truncation in the padded y and z directions
                Upad_hat3 = rfft(u/self.padsize, Upad_hat3, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                Upad_hat2 = slab_shen_r2c.copy_from_padded(Upad_hat3, Upad_hat2, self.N, 2)
                Upad_hat2 = fft(Upad_hat2/self.padsize, Upad_hat2, axis=1, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['fft'])
                Upad_hat1 = slab_shen_r2c.copy_from_padded(Upad_hat2, Upad_hat1, self.N, 1)
                
                if self.communication == 'alltoall':
                    # Transpose and commuincate data
                    U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
                    U_mpi[:] = rollaxis(Upad_hat1.reshape(self.complex_shape_padded_I()), 1)
                    self.comm.Alltoall(self.MPI.IN_PLACE, [Upad_hat, self.mpitype])
                    
                elif self.communication == 'Alltoallw':
                    if len(self._subarraysA_pad) == 0:
                        self._subarraysA_pad, self._subarraysB_pad, self._counts_displs = self.get_subarrays(padsize=self.padsize)
                    
                    self.comm.Alltoallw(
                        [Upad_hat1, self._counts_displs, self._subarraysB_pad],
                        [Upad_hat,  self._counts_displs, self._subarraysA_pad])
                    
                # Perform fst of data in x-direction
                Upad_hat0 = fun(Upad_hat, Upad_hat0)
                
                # Truncate to original complex shape
                fu[:] = Upad_hat0[:self.N[0]]
            
        return fu

    #@profile
    def backward(self, fu, u, fun, dealias=None):
        
        Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0, False)]
        Uc_mpi  = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
                
        if dealias == '2/3-rule' and self.dealias.shape == (0,):
            self.dealias = self.get_dealias_filter()
            
        if self.num_processes == 1:
            if not dealias == '3/2-rule':
                if dealias == '2/3-rule':
                    fu *= self.dealias
                    
                Uc_hat = fun(fu, Uc_hat)
                u = irfft2(Uc_hat, u, axes=(1,2), overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft2'])
            
            else:
                if not self.dealias_cheb:
                    Upad_hat = self.work_arrays[(self.complex_shape_padded(), self.complex, 0)]
                    Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0)]
                    
                    Uc_hat = fun(fu, Uc_hat)
                    Upad_hat_z = slab_shen_r2c.copy_to_padded(Uc_hat, Upad_hat_z, self.N, 1)
                    Upad_hat_z = ifft(self.padsize*Upad_hat_z, Upad_hat_z, axis=1, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['ifft'])
                    Upad_hat = slab_shen_r2c.copy_to_padded(Upad_hat_z, Upad_hat, self.N, 2)
                    u = irfft(self.padsize*Upad_hat, u, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft'])                
            return u

        if not dealias == '3/2-rule':
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
            if dealias == '2/3-rule':
                fu *= self.dealias
                
            Uc_hat = fun(fu, Uc_hat)
            
            if self.communication == 'alltoall':
                self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
                Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())                
                #Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0, False)]
                #self.comm.Alltoall([Uc_hat, self.mpitype], [Uc_mpi, self.mpitype])
                #Uc_hatT = np.rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
                
            elif self.communication == 'Alltoallw':
                if len(self._subarraysA) == 0:
                    self._subarraysA, self._subarraysB, self._counts_displs = self.get_subarrays()
                Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
                self.comm.Alltoallw(
                    [Uc_hat, self._counts_displs, self._subarraysA],
                    [Uc_hatT,  self._counts_displs, self._subarraysB])
            
            u = irfft2(Uc_hatT, u, axes=(1,2), overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft2'])
        
        else:
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
            if not self.dealias_cheb:
                Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
                Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0)]
                
                Uc_hat = fun(fu, Uc_hat)
                if self.communication == 'alltoall':
                    self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
                    Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
                    
                elif self.communication == 'Alltoallw':
                    if len(self._subarraysA) == 0:
                        self._subarraysA, self._subarraysB, self._counts_displs = self.get_subarrays()
                        
                    self.comm.Alltoallw(
                        [Uc_hat, self._counts_displs, self._subarraysA],
                        [Uc_hatT,  self._counts_displs, self._subarraysB])
                
                Upad_hat_z = slab_shen_r2c.copy_to_padded(Uc_hatT, Upad_hat_z, self.N, 1)
                Upad_hat_z = ifft(self.padsize*Upad_hat_z, Upad_hat_z, axis=1, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['ifft'])
                Upad_hatT = slab_shen_r2c.copy_to_padded(Upad_hat_z, Upad_hatT, self.N, 2)
                u = irfft(self.padsize*Upad_hatT, u, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft'])
                
            else:
                assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"            
            
                # Intermediate work arrays required for transform
                Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
                Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1)]
                Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
                Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0)]
                Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0)]
                
                # Expand in x-direction and perform ifst
                Upad_hat0 = slab_shen_r2c.copy_to_padded(fu, Upad_hat0, self.N, 0)
                Upad_hat = fun(Upad_hat0, Upad_hat) 
                
                if self.communication == 'alltoall':
                    # Communicate to distribute first dimension (like Fig. 2b but padded in x-dir)
                    self.comm.Alltoall(self.MPI.IN_PLACE, [Upad_hat, self.mpitype])
                    
                    # Transpose data and pad in y-direction before doing ifft. Now data is padded in x and y 
                    U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
                    Upad_hat1[:] = rollaxis(U_mpi, 1).reshape(Upad_hat1.shape)
                    
                elif self.communication == 'Alltoallw':
                    if len(self._subarraysA_pad) == 0:
                        self._subarraysA_pad, self._subarraysB_pad, self._counts_displs = self.get_subarrays(padsize=self.padsize)
                    self.comm.Alltoallw(
                        [Upad_hat,  self._counts_displs, self._subarraysA_pad],
                        [Upad_hat1, self._counts_displs, self._subarraysB_pad])
                
                Upad_hat2 = slab_shen_r2c.copy_to_padded(Upad_hat1, Upad_hat2, self.N, 1)
                Upad_hat2 = ifft(Upad_hat2*self.padsize, Upad_hat2, axis=1, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['ifft'])
                
                # pad in z-direction and perform final irfft
                Upad_hat3 = slab_shen_r2c.copy_to_padded(Upad_hat2, Upad_hat3, self.N, 2)
                u = irfft(Upad_hat3*self.padsize, u, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft'])

        return u        

    def fss(self, u, fu, S, dealias=None):
        """Fast Shen scalar product of x-direction, Fourier transform of y and z"""
        fu = self.forward(u, fu, S.fastShenScalar, dealias=dealias)            
        return fu

    def fst(self, u, fu, S, dealias=None):
        """Fast Shen transform of x-direction, Fourier transform of y and z"""
        fu = self.forward(u, fu, S.fst, dealias=dealias)            
        return fu
    
    def fct(self, u, fu, S, dealias=None):
        """Fast Cheb transform of x-direction, Fourier transform of y and z"""
        fu = self.forward(u, fu, S.fct, dealias=dealias)            
        return fu
        
    def ifst(self, fu, u, S, dealias=None):
        """Inverse Shen transform of x-direction, Fourier in y and z"""
        u = self.backward(fu, u, S.ifst, dealias=dealias)
        return u

    def ifct(self, fu, u, S, dealias=None):
        """Inverse Cheb transform of x-direction, Fourier in y and z"""
        u = self.backward(fu, u, S.ifct, dealias=dealias)
        return u

    def fft(self, u, fu):
        """Fast Fourier transform of y and z"""
        # Intermediate work arrays
        Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hatT = rfft2(u, Uc_hatT, axes=(1,2), threads=self.threads, planner_effort=self.planner_effort['rfft2'])
        Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
        self.comm.Alltoall([Uc_mpi, self.mpitype], [fu, self.mpitype])
        return fu
    
    def ifft(self, fu, u):
        """Inverse Fourier transforms in y and z"""
        Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]        
        self.comm.Alltoall([fu, self.mpitype], [Uc_mpi, self.mpitype])
        Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
        u = irfft2(Uc_hatT, u, axes=(1,2), threads=self.threads, planner_effort=self.planner_effort['irfft2'])
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
        u0 = self.ifct0(UT[1], u0, S)
        return u0

setup = {"IPCS": setupShen,
         "IPCSR": setupShen,
         "KMM": setupShenKMM,
         "KMMRK3": setupShenKMM,
         "IPCS_MHD": setupShenMHD,
         "IPCS_GeneralBCs": setupShenGeneralBCs}
