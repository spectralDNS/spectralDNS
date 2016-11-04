import numpy as np
from numpy import array, zeros_like, sum, hstack, meshgrid, abs, pi, uint8, \
    rollaxis, arange
from numpy.polynomial import chebyshev as n_cheb
from mpiFFT4py import dct, work_arrays, Slab_R2C, fftfreq, rfftfreq, rfft2, \
    irfft2, rfft, irfft, fft, ifft
from . import SFTc
import scipy.sparse.linalg as la
from ..shen.la import TDMA, PDMA
from ..shen.Matrices import BBBmat
from ..optimization import optimizer
from collections import defaultdict
import decimal
from mpi4py import MPI

"""
Fast transforms for pure Chebyshev basis or 
Shen's Chebyshev basis: 

  For homogeneous Dirichlet boundary conditions:
 
    phi_k = T_k - T_{k+2}
    
  For homogeneous Neumann boundary conditions:
    
    phi_k = T_k - (k/k+2)**2 * T_{k+2}
    
  For Biharmonic basis with both homogeneous Dirichlet
  and Neumann:
  
    phi_k = T_k - 2(k+2)/(k+3)*T_{k+2} + (k+1)/(k+3)*T_{k+4}

Use either Chebyshev-Gauss (GC) or Gauss-Lobatto (GL)
points in real space.

The ChebyshevTransform may be used to compute derivatives
through fast Chebyshev transforms.

"""
float, complex = np.float64, np.complex128
pi, zeros, ones = np.pi, np.zeros, np.ones
work = work_arrays()

class ChebyshevTransform(object):
    
    def __init__(self, quad="GL", threads=1, planner_effort="FFTW_MEASURE"): 
        self.quad = quad
        self.threads = threads
        self.planner_effort = planner_effort
        self.points = zeros(0)
        self.weights = zeros(0)
                
    def points_and_weights(self, N):
        self.N = N
        if self.quad == "GL":
            points = (n_cheb.chebpts2(N)[::-1]).astype(float)
            #points = (n_cheb.chebpts2(N)).astype(float)
            weights = zeros(N)+pi/(N-1)
            weights[0] /= 2
            weights[-1] /= 2

        elif self.quad == "GC":
            points, weights = n_cheb.chebgauss(N)
            #points = points[::-1]
            points = points.astype(float)
            weights = weights.astype(float)
            
        return points, weights
    
    def init(self, N):
        """Vandermonde matrix is used just for verification"""
        self.points, self.weights = self.points_and_weights(N)
        # Build Vandermonde matrix.
        self.V = n_cheb.chebvander(self.points, N-1).T
        
    def chebDerivativeCoefficients(self, fk, fj):
        SFTc.chebDerivativeCoefficients(fk, fj)  
        return fj

    def chebDerivative_3D(self, fj, fd):
        fk = work[(fj, 0)]
        fkd = work[(fj, 1)]
        fk = self.fct(fj, fk)
        fkd = SFTc.chebDerivativeCoefficients_3D(fk, fkd)
        fd = self.ifct(fkd, fd)
        return fd
    
    def fastChebDerivative(self, fj, fd):
        """Compute derivative of fj at the same points."""
        fk = work[(fj, 0)]
        fkd = work[(fj, 1)]
        fk = self.fct(fj, fk)
        fkd = self.chebDerivativeCoefficients(fk, fkd)
        fd  = self.ifct(fkd, fd)
        return fd
        
    @staticmethod
    @optimizer
    def scale_fct(cj, quad):
        N = cj.shape[0]
        if quad == 'GC':
            cj /= N
            cj[0] /= 2

        elif quad == 'GL':
            cj /= (N-1)
            cj[0] /= 2
            cj[-1] /= 2

        return cj
        
    #@profile
    def fct(self, fj, cj):
        """Fast Chebyshev transform."""
        if self.quad == "GC":
            cj = dct(fj, cj, type=2, axis=0, threads=self.threads, planner_effort=self.planner_effort)  
                
        elif self.quad == "GL":
            cj = dct(fj, cj, type=1, axis=0, threads=self.threads, planner_effort=self.planner_effort)
        
        cj = ChebyshevTransform.scale_fct(cj, self.quad)
        return cj

    @staticmethod
    @optimizer
    def scale_ifct(cj, fk, quad):
        if quad == 'GC':
            cj *= 0.5
            cj += fk[0]/2
        elif quad == 'GL':
            cj *= 0.5
            cj += fk[0]/2
            cj[::2] += fk[-1]/2
            cj[1::2] -= fk[-1]/2
        return cj
    
    #@profile
    def ifct(self, fk, cj):
        """Inverse fast Chebyshev transform."""
        if self.quad == "GC":
            cj = dct(fk, cj, type=3, axis=0, threads=self.threads, planner_effort=self.planner_effort)
        
        elif self.quad == "GL":
            cj = dct(fk, cj, type=1, axis=0, threads=self.threads, planner_effort=self.planner_effort)
        
        cj = ChebyshevTransform.scale_ifct(cj, fk, self.quad)
        return cj
    
    #@profile
    def fastChebScalar(self, fj, fk, fast_transform=True):
        """Fast Chebyshev scalar product."""
        N = fj.shape[0]
        if fast_transform:
            if self.quad == "GC":
                fk = dct(fj, fk, type=2, axis=0, threads=self.threads, planner_effort=self.planner_effort)
                fk *= (pi/(2*N))
                            
            elif self.quad == "GL":
                fk = dct(fj, fk, type=1, axis=0, threads=self.threads, planner_effort=self.planner_effort)
                fk *= (pi/(2*(N-1)))
        else:
            if not self.points.shape == (N,): self.init(N)
            fk[:] = np.dot(self.V, fj*self.weights)

        return fk

    def slice(self, N):
        return slice(0, N)

class ShenDirichletBasis(ChebyshevTransform):
    """Shen basis for Dirichlet boundary conditions
    
    args:
        quad        ('GL', 'GC')  Gauss-Lobatto or Gauss-Chebyshev points
        threads          1        Number of threads used by pyfftw
        planner_effort            Planner effort for FFTs. 
        bc             (a, b)     Boundary conditions at x=(1,-1)

    """
    
    
    def __init__(self, quad="GL", threads=1, planner_effort="FFTW_MEASURE",
                 bc=(0., 0.)):
        ChebyshevTransform.__init__(self, quad=quad, planner_effort=planner_effort)
        self.N = -1
        self.Solver = TDMA(quad, False)
        self.bc = bc
        
    def init(self, N):
        """Vandermonde matrix is used just for verification"""
        self.points, self.weights = self.points_and_weights(N)
        # Build Vandermonde matrix.
        self.V = np.zeros((N, N))
        V = n_cheb.chebvander(self.points, N-1)
        self.V[:N-2, :] = V[:, :-2].T - V[:, 2:].T
        self.V[N-2, :] = 0.5*(V[:, 0] + V[:, 1])
        self.V[N-1, :] = 0.5*(V[:, 0] - V[:, 1])

    def wavenumbers(self, N):
        if isinstance(N, tuple):
            if len(N) == 1:
                N = N[0]
        if isinstance(N, int): 
            return np.arange(N-2).astype(float)
        
        else:
            kk = np.mgrid[:N[0]-2, :N[1], :N[2]].astype(float)
            return kk[0]

    #@profile
    def fastShenScalar(self, fj, fk, fast_transform=True):
                
        if fast_transform:
            fk = self.fastChebScalar(fj, fk)
            #c0 = 0.5*(fk[0] + fk[1])
            #c1 = 0.5*(fk[0] - fk[1])
            fk[:-2] -= fk[2:]
            #fk[-2] = c0
            #fk[-1] = c1
        else:
            if not self.points.shape == (fj.shape[0],): self.init(fj.shape[0])
            fk[:] = np.dot(self.V, fj*self.weights)

        fk[-2:] = 0     # Last two not used, so set to zero. Even for nonhomogeneous bcs, where they are technically non-zero
        return fk
        
    #@profile
    def ifst(self, fk, fj, fast_transform=True):
        """Fast inverse Shen transform
        
        Transform needs to take into account that phi_k = T_k - T_{k+2}
        fk contains Shen coefficients in the first fk.shape[0]-2 positions
        """
        if fast_transform:
            w_hat = work[(fk, 0)]
            w_hat[:-2] = fk[:-2]
            w_hat[2:] -= fk[:-2] 
            w_hat[0] += 0.5*(self.bc[0] + self.bc[1])
            w_hat[1] += 0.5*(self.bc[0] - self.bc[1])
            
            fj = self.ifct(w_hat, fj)
            return fj
        
        else:
            if not self.points.shape == (fj.shape[0],): self.init(fj.shape[0])
            return np.dot(self.V.T, fk[:-2])

    #@profile
    def fst(self, fj, fk):
        """Fast Shen transform
        """
        fk = self.fastShenScalar(fj, fk)
        fk[0] -= pi/2*(self.bc[0] + self.bc[1])
        fk[1] -= pi/4*(self.bc[0] - self.bc[1])
        fk = self.Solver(fk)
        fk[-2] = self.bc[0]
        fk[-1] = self.bc[1]
        return fk
    
    def slice(self, N):
        return slice(0, N-2)

class ShenNeumannBasis(ShenDirichletBasis):
    
    def __init__(self, quad="GC", threads=1, planner_effort="FFTW_MEASURE"): 
        ShenDirichletBasis.__init__(self, quad, threads, planner_effort)
        self.factor = None        
        self.k = None
        self.Solver = TDMA(quad, True)
            
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        k = self.wavenumbers(N)
        # Build Vandermonde matrix. Note! N points in real space gives N-3 bases in spectral space
        self.V = n_cheb.chebvander(self.points, N-3).T - ((k/(k+2))**2)[:, np.newaxis]*n_cheb.chebvander(self.points, N-1)[:, 2:].T
        self.V = self.V[1:, :]

    def set_factor_array(self, v):
        recreate = False
        if isinstance(self.factor, np.ndarray):
            if not self.factor.shape == v.shape:
                recreate = True
            
        if self.factor is None:
            recreate = True
            
        if recreate:
            if len(v.shape)==3:
                k = self.wavenumbers(v.shape)                
            elif len(v.shape)==1:
                k = self.wavenumbers(v.shape[0])
            self.factor = (k[1:]/(k[1:]+2))**2

    def fastShenScalar(self, fj, fk, fast_transform=True):
        """Fast Shen scalar product.
        Chebyshev transform taking into account that phi_k = T_k - (k/(k+2))**2*T_{k+2}
        Note, this is the non-normalized scalar product
        """        
        if fast_transform:
            self.set_factor_array(fk)
            fk = self.fastChebScalar(fj, fk)
            fk[1:-2] -= self.factor * fk[3:]
            fk[0] = 0

        else:
            if not self.points.shape == (fj.shape[0],): self.init(fj.shape[0])
            fk[1:-2] = np.dot(self.V, fj*self.weights)
            
        fk[-2:] = 0
        return fk
    
    def ifst(self, fk, fj):
        """Fast inverse Shen scalar transform
        """
        w_hat = work[(fk, 0)]
        self.set_factor_array(fk)
        w_hat[1:-2] = fk[1:-2]
        w_hat[3:] -= self.factor*fk[1:-2]
        fj = self.ifct(w_hat, fj)
        return fj

    def slice(self, N):
        return slice(1, N-2)

class ShenBiharmonicBasis(ShenDirichletBasis):
    
    def __init__(self, quad="GC", threads=1, planner_effort="FFTW_MEASURE"):
        ShenDirichletBasis.__init__(self, quad, threads, planner_effort)
        self.factor1 = zeros(0)
        self.factor2 = zeros(0)
        self.Solver = PDMA(quad)
        
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        k = self.wavenumbers(N)
        # Build Vandermonde matrix.
        self.V = n_cheb.chebvander(self.points, N-5).T - (2*(k+2)/(k+3))[:, np.newaxis]*n_cheb.chebvander(self.points, N-3)[:, 2:].T + ((k+1)/(k+3))[:, np.newaxis]*n_cheb.chebvander(self.points, N-1)[:, 4:].T
        
    def wavenumbers(self, N):
        if isinstance(N, tuple):
            if len(N) == 1:
                N = N[0]
        if isinstance(N, int): 
            return np.arange(N-4).astype(float)
        
        else:
            kk = np.mgrid[:N[0]-4, :N[1], :N[2]].astype(float)
            return kk[0]
        
    def set_factor_arrays(self, v):
        if not self.factor1.shape == v[:-4].shape:
            if len(v.shape)==3:
                k = self.wavenumbers(v.shape)                
            elif len(v.shape)==1:
                k = self.wavenumbers(v.shape[0])
                #k = np.array(map(decimal.Decimal, np.arange(v.shape[0]-4)))
                
            self.factor1 = (-2*(k+2)/(k+3)).astype(float)
            self.factor2 = ((k+1)/(k+3)).astype(float)

    def fastShenScalar(self, fj, fk, fast_transform=True):
        """Fast Shen scalar product.
        """        
        if fast_transform:
            self.set_factor_arrays(fk)
            Tk = work[(fk, 0)]
            Tk = self.fastChebScalar(fj, Tk)
            fk[:] = Tk[:]
            fk[:-4] += self.factor1 * Tk[2:-2]
            fk[:-4] += self.factor2 * Tk[4:]

        else:
            if not self.points.shape == (fj.shape[0],): self.init(fj.shape[0])
            fk[:-4] = np.dot(self.V, fj*self.weights)
            
        fk[-4:] = 0
        return fk
    
    @staticmethod
    @optimizer
    def set_w_hat(w_hat, fk, f1, f2):
        w_hat[:-4] = fk[:-4]
        w_hat[2:-2] += f1*fk[:-4]
        w_hat[4:]   += f2*fk[:-4]
        return w_hat
    
    def ifst(self, fk, fj):
        """Fast inverse Shen scalar transform
        """
        w_hat = work[(fk, 0)]
        self.set_factor_arrays(fk)
        w_hat = ShenBiharmonicBasis.set_w_hat(w_hat, fk, self.factor1, self.factor2)
        fj = self.ifct(w_hat, fj)
        return fj
    
    def slice(self, N):
        return slice(0, N-4)

class SlabShen_R2C(Slab_R2C):
    
    def __init__(self, N, L, comm, padsize=1.5, threads=1, communication='Alltoallw', dealias_cheb=False,
                 planner_effort=defaultdict(lambda: "FFTW_MEASURE", {"dct": "FFTW_EXHAUSTIVE"})):
        Slab_R2C.__init__(self, N, L, comm, "double", padsize=padsize, threads=threads,
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
        
    def real_local_slice(self, padsize=1):
        if self.dealias_cheb:
            return (slice(self.rank*padsize*self.Np[0], (self.rank+1)*padsize*self.Np[0], 1),
                    slice(0, int(padsize*self.N[1]), 1), 
                    slice(0, int(padsize*self.N[2]), 1))
        else:
            return (slice(self.rank*self.Np[0], (self.rank+1)*self.Np[0], 1),
                    slice(0, int(padsize*self.N[1]), 1), 
                    slice(0, int(padsize*self.N[2]), 1))
    
    def global_complex_shape(self, padsize=1.0):
        """Global size of problem in complex wavenumber space"""
        if self.dealias_cheb:
            return (int(self.padsize*self.N[0]), int(self.padsize*self.N[1]),
                    int(padsize*self.N[2]/2+1))
        else:
            return (self.N[0], int(self.padsize*self.N[1]),
                    int(padsize*self.N[2]/2+1))
    
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
        X = array(meshgrid(x0[int(self.rank*self.Np[0]):int((self.rank+1)*self.Np[0])], 
                           x1, x2, indexing='ij'), dtype=self.float)
        return X
    
    def get_local_wavenumbermesh(self):
        kx = arange(self.N[0]).astype(self.float)
        ky = fftfreq(self.N[1], 1./self.N[1])[int(self.rank*self.Np[1]):int((self.rank+1)*self.Np[1])]
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
            #fu[:, N[1]/2, 0] *= 2 # Because of symmetrical padding
            #fu[:, N[1]/2, N[2]/2] *= 2
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
                
                    Upad_hat = rfft(u, Upad_hat, axis=2, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                    Upad_hat_z = SlabShen_R2C.copy_from_padded(Upad_hat, Upad_hat_z, self.N, 2)
                    Upad_hat_z[:] = fft(Upad_hat_z, axis=1, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['fft'])   
                    Uc_hat = SlabShen_R2C.copy_from_padded(Upad_hat_z, Uc_hat, self.N, 1)
                    fu = fun(Uc_hat/self.padsize**2, fu)
                else:
                    # Intermediate work arrays required for transform
                    Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
                    Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1, False)]
                    Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0, False)]
                    Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]

                    # Do ffts and truncation in the padded y and z directions
                    Upad_hat3 = rfft(u, Upad_hat3, axis=2, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                    Upad_hat2 = SlabShen_R2C.copy_from_padded(Upad_hat3, Upad_hat2, self.N, 2)
                    Upad_hat2[:] = fft(Upad_hat2, axis=1, threads=self.threads, planner_effort=self.planner_effort['fft'])
                    Upad_hat = SlabShen_R2C.copy_from_padded(Upad_hat2, Upad_hat, self.N, 1)
                    
                    # Perform fst of data in x-direction
                    Upad_hat0 = fun(Upad_hat, Upad_hat0)
                    
                    # Truncate to original complex shape
                    fu[:] = Upad_hat0[:self.N[0]]/self.padsize**2
            return fu

        if not dealias == '3/2-rule':
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]

            if self.communication == 'alltoall':
                Uc_mpi  = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
                Uc_hatT = rfft2(u, Uc_hatT, axes=(1,2), threads=self.threads, planner_effort=self.planner_effort['rfft2'])
                Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
                self.comm.Alltoall(MPI.IN_PLACE, [Uc_hat, self.mpitype])
                
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
            Uc_mpi  = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
            if not self.dealias_cheb:
                Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0, False)]
                Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0, False)]
                
                Upad_hatT = rfft(u, Upad_hatT, axis=2, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                Upad_hat_z = SlabShen_R2C.copy_from_padded(Upad_hatT, Upad_hat_z, self.N, 2)
                Upad_hat_z[:] = fft(Upad_hat_z, axis=1, threads=self.threads, planner_effort=self.planner_effort['fft'])   
                Uc_hatT = SlabShen_R2C.copy_from_padded(Upad_hat_z, Uc_hatT, self.N, 1)                
                
                if self.communication == 'alltoall':
                    Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
                    self.comm.Alltoall(MPI.IN_PLACE, [Uc_hat, self.mpitype])
                
                elif self.communication == 'Alltoallw':
                    if len(self._subarraysA) == 0:
                        self._subarraysA, self._subarraysB, self._counts_displs = self.get_subarrays()
                    self.comm.Alltoallw(
                        [Uc_hatT, self._counts_displs, self._subarraysB],
                        [Uc_hat,  self._counts_displs, self._subarraysA])
                    
                fu = fun(Uc_hat/self.padsize**2, fu)
            
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
                Upad_hat3 = rfft(u, Upad_hat3, axis=2, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                Upad_hat2 = SlabShen_R2C.copy_from_padded(Upad_hat3, Upad_hat2, self.N, 2)
                Upad_hat2[:] = fft(Upad_hat2, axis=1, threads=self.threads, planner_effort=self.planner_effort['fft'])
                Upad_hat1 = SlabShen_R2C.copy_from_padded(Upad_hat2, Upad_hat1, self.N, 1)
                
                if self.communication == 'alltoall':
                    # Transpose and commuincate data
                    U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
                    U_mpi[:] = rollaxis(Upad_hat1.reshape(self.complex_shape_padded_I()), 1)
                    self.comm.Alltoall(MPI.IN_PLACE, [Upad_hat, self.mpitype])
                    
                elif self.communication == 'Alltoallw':
                    if len(self._subarraysA_pad) == 0:
                        self._subarraysA_pad, self._subarraysB_pad, self._counts_displs = self.get_subarrays(padsize=self.padsize)
                    
                    self.comm.Alltoallw(
                        [Upad_hat1, self._counts_displs, self._subarraysB_pad],
                        [Upad_hat,  self._counts_displs, self._subarraysA_pad])
                    
                # Perform fst of data in x-direction
                Upad_hat0 = fun(Upad_hat, Upad_hat0)
                
                # Truncate to original complex shape
                fu[:] = Upad_hat0[:self.N[0]]/self.padsize**2
            
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
                    
                    Uc_hat = fun(fu*self.padsize**2, Uc_hat)
                    Upad_hat_z = SlabShen_R2C.copy_to_padded(Uc_hat, Upad_hat_z, self.N, 1)
                    Upad_hat_z[:] = ifft(Upad_hat_z, axis=1, threads=self.threads, planner_effort=self.planner_effort['ifft'])
                    Upad_hat = SlabShen_R2C.copy_to_padded(Upad_hat_z, Upad_hat, self.N, 2)
                    u = irfft(Upad_hat, u, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft']) 
                    
                else:
                    # Intermediate work arrays required for transform
                    Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
                    Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1)]
                    Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
                    Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0)]
                    Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0)]
                    
                    # Expand in x-direction and perform ifst
                    Upad_hat0 = SlabShen_R2C.copy_to_padded(fu*self.padsize**2, Upad_hat0, self.N, 0)
                    Upad_hat = fun(Upad_hat0, Upad_hat) 
                    
                    Upad_hat2 = SlabShen_R2C.copy_to_padded(Upad_hat, Upad_hat2, self.N, 1)
                    Upad_hat2[:] = ifft(Upad_hat2, axis=1, threads=self.threads, planner_effort=self.planner_effort['ifft'])
                    
                    # pad in z-direction and perform final irfft
                    Upad_hat3 = SlabShen_R2C.copy_to_padded(Upad_hat2, Upad_hat3, self.N, 2)
                    u = irfft(Upad_hat3, u, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft'])
                    
            return u

        if not dealias == '3/2-rule':
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
            if dealias == '2/3-rule':
                fu *= self.dealias
                
            Uc_hat = fun(fu, Uc_hat)
            
            if self.communication == 'alltoall':
                self.comm.Alltoall(MPI.IN_PLACE, [Uc_hat, self.mpitype])
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
                
                Uc_hat = fun(fu*self.padsize**2, Uc_hat)
                if self.communication == 'alltoall':
                    self.comm.Alltoall(MPI.IN_PLACE, [Uc_hat, self.mpitype])
                    Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
                    
                elif self.communication == 'Alltoallw':
                    if len(self._subarraysA) == 0:
                        self._subarraysA, self._subarraysB, self._counts_displs = self.get_subarrays()
                        
                    self.comm.Alltoallw(
                        [Uc_hat, self._counts_displs, self._subarraysA],
                        [Uc_hatT,  self._counts_displs, self._subarraysB])
                
                Upad_hat_z = SlabShen_R2C.copy_to_padded(Uc_hatT, Upad_hat_z, self.N, 1)
                Upad_hat_z[:] = ifft(Upad_hat_z, axis=1, threads=self.threads, planner_effort=self.planner_effort['ifft'])
                Upad_hatT = SlabShen_R2C.copy_to_padded(Upad_hat_z, Upad_hatT, self.N, 2)
                u = irfft(Upad_hatT, u, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft'])
                
            else:
                assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"            
            
                # Intermediate work arrays required for transform
                Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
                Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1)]
                Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
                Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0)]
                Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0)]
                
                # Expand in x-direction and perform ifst
                Upad_hat0 = SlabShen_R2C.copy_to_padded(fu*self.padsize**2, Upad_hat0, self.N, 0)
                Upad_hat = fun(Upad_hat0, Upad_hat) 
                
                if self.communication == 'alltoall':
                    # Communicate to distribute first dimension (like Fig. 2b but padded in x-dir)
                    self.comm.Alltoall(MPI.IN_PLACE, [Upad_hat, self.mpitype])
                    
                    # Transpose data and pad in y-direction before doing ifft. Now data is padded in x and y 
                    U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
                    Upad_hat1[:] = rollaxis(U_mpi, 1).reshape(Upad_hat1.shape)
                    
                elif self.communication == 'Alltoallw':
                    if len(self._subarraysA_pad) == 0:
                        self._subarraysA_pad, self._subarraysB_pad, self._counts_displs = self.get_subarrays(padsize=self.padsize)
                    self.comm.Alltoallw(
                        [Upad_hat,  self._counts_displs, self._subarraysA_pad],
                        [Upad_hat1, self._counts_displs, self._subarraysB_pad])
                
                Upad_hat2 = SlabShen_R2C.copy_to_padded(Upad_hat1, Upad_hat2, self.N, 1)
                Upad_hat2[:] = ifft(Upad_hat2, axis=1, threads=self.threads, planner_effort=self.planner_effort['ifft'])
                
                # pad in z-direction and perform final irfft
                Upad_hat3 = SlabShen_R2C.copy_to_padded(Upad_hat2, Upad_hat3, self.N, 2)
                u = irfft(Upad_hat3, u, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft'])

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
    
    def dx(self, u, quad):
        """Compute integral of u over domain"""
        uu = sum(u, axis=(1,2))
        c = zeros(self.N[0])
        self.comm.Gather(uu, c)
        if self.rank == 0:
            if quad == 'GL':
                ak = zeros_like(c)
                ak = dct(c, ak, 1, axis=0)
                ak /= (self.N[0]-1)
                w = arange(0, self.N[0], 1, dtype=float)
                w[2:] = 2./(1-w[2:]**2)
                w[0] = 1
                w[1::2] = 0
                return sum(ak*w)*self.L[1]*self.L[2]/self.N[1]/self.N[2]
            
            elif quad == 'GC':
                d = zeros(self.N[0])
                k = 2*(1 + arange((self.N[0]-1)//2))
                d[::2] = (2./self.N[0])/hstack((1., 1.-k*k))
                w = zeros_like(d)
                w = dct(d, w, type=3, axis=0)
                return sum(c*w)*self.L[1]*self.L[2]/self.N[1]/self.N[2]                
        else:
            return 0    

    
if __name__ == "__main__":
    from sympy import Symbol, sin, cos
    N = 16
    #a = np.random.random((N, N, N/2+1))+1j*np.random.random((N, N, N/2+1))
    #af = zeros((N, N, N/2+1), dtype=a.dtype)
    #a[0,:,:] = 0
    #a[-1,:,:] = 0
    #a = np.random.random(N)+np.random.random(N)*1j 
    #af = zeros(N, dtype=np.complex)
    #a[0] = 0
    #a[-1] = 0
    x = Symbol("x")
    f = (1-x**2)*cos(pi*x)    
    CT = ChebyshevTransform(quad="GL")
    points, weights = CT.points_and_weights(N)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)    
    u0 = zeros(N, dtype=float)
    u0 = CT.fct(fj, u0)
    u1 = u0.copy()
    u1 = CT.ifct(u0, u1)
    assert np.allclose(u1, fj)    
        
    ST = ShenDirichletBasis(quad="GC")
    points, weights = ST.points_and_weights(N)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)    
    u0 = zeros(N, dtype=float)
    u0 = ST.fst(fj, u0)
    u1 = u0.copy()
    u1 = ST.ifst(u0, u1)
    assert np.allclose(u1, fj)
    
    SN = ShenNeumannBasis(quad="GL")
    points, weights = SN.points_and_weights(N)
    f = cos(pi*x)  # A function with f(+-1) = 0 and f'(+-1) = 0
    fj = np.array([f.subs(x, j) for j in points], dtype=float)
    fj -= np.dot(fj, weights)/weights.sum()
    u0 = SN.fst(fj, u0)
    u1 = u0.copy()
    u1 = SN.ifst(u0, u1)
    assert np.allclose(u1, fj)

    N = 30
    SB = ShenBiharmonicBasis(quad="GL")
    points, weights = SB.points_and_weights(N)
    f = (1-x**2)*sin(2*pi*x)    
    fj = np.array([f.subs(x, j) for j in points], dtype=float)    
    u0 = zeros(N, dtype=float)
    u0 = SB.fst(fj, u0)
    u1 = u0.copy()
    u1 = SB.ifst(u0, u1)
    assert np.allclose(u1, fj)
