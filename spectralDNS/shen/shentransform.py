import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from mpiFFT4py import dct, work_arrays
import SFTc
import scipy.sparse.linalg as la
from spectralDNS.shen.la import TDMA, PDMA
from spectralDNS.shen.Matrices import BBBmat
from spectralDNS.optimization import optimizer
import decimal

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
    
    def __init__(self, quad="GL", fast_transform=True, threads=1, planner_effort="FFTW_MEASURE"): 
        self.quad = quad
        self.fast_transform = fast_transform
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
    def fastChebScalar(self, fj, fk):
        """Fast Chebyshev scalar product."""
        N = fj.shape[0]
        if self.fast_transform:
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
    
    def __init__(self, quad="GL", fast_transform=True, threads=1, planner_effort="FFTW_MEASURE"):
        ChebyshevTransform.__init__(self, quad=quad, fast_transform=fast_transform,
                                    planner_effort=planner_effort)
        self.N = -1
        self.Solver = TDMA(quad, False)
        
    def init(self, N):
        """Vandermonde matrix is used just for verification"""
        self.points, self.weights = self.points_and_weights(N)
        # Build Vandermonde matrix. Note! N points in real space gives N-2 bases in spectral space
        self.V = n_cheb.chebvander(self.points, N-3).T - n_cheb.chebvander(self.points, N-1)[:, 2:].T

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
    def fastShenScalar(self, fj, fk):
                
        if self.fast_transform:
            fk = self.fastChebScalar(fj, fk)
            fk[:-2] -= fk[2:]
        else:
            if not self.points.shape == (fj.shape[0],): self.init(fj.shape[0])
            fk[:-2] = np.dot(self.V, fj*self.weights)
        fk[-2:] = 0     # Last two not used by Shen
        return fk
        
    #@profile
    def ifst(self, fk, fj):
        """Fast inverse Shen transform
        Transform needs to take into account that phi_k = T_k - T_{k+2}
        fk contains Shen coefficients in the first fk.shape[0]-2 positions
        """
        if self.fast_transform:
            w_hat = work[(fk, 0)]
            w_hat[:-2] = fk[:-2] 
            w_hat[2:] -= fk[:-2] 
            #w_hat[:2] = fk[:2]
            #w_hat[2:] = fk[2:]-fk[:-2]  
            
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
        fk = self.Solver(fk)
        return fk
    
    def slice(self, N):
        return slice(0, N-2)

class ShenNeumannBasis(ShenDirichletBasis):
    
    def __init__(self, quad="GC", fast_transform=True, threads=1, planner_effort="FFTW_MEASURE"): 
        ShenDirichletBasis.__init__(self, quad, fast_transform, threads, planner_effort)
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

    def fastShenScalar(self, fj, fk):
        """Fast Shen scalar product.
        Chebyshev transform taking into account that phi_k = T_k - (k/(k+2))**2*T_{k+2}
        Note, this is the non-normalized scalar product
        """        
        if self.fast_transform:
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
    
    def __init__(self, quad="GC", fast_transform=True, threads=1, planner_effort="FFTW_MEASURE"):
        ShenDirichletBasis.__init__(self, quad, fast_transform, threads, planner_effort)
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

    def fastShenScalar(self, fj, fk):
        """Fast Shen scalar product.
        """        
        if self.fast_transform:
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
    
    #@profile
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
        
    ST = ShenDirichletBasis(quad="GC", fast_transform=True)
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
    SB = ShenBiharmonicBasis(quad="GL", fast_transform=False)
    points, weights = SB.points_and_weights(N)
    f = (1-x**2)*sin(2*pi*x)    
    fj = np.array([f.subs(x, j) for j in points], dtype=float)    
    u0 = zeros(N, dtype=float)
    u0 = SB.fst(fj, u0)
    u1 = u0.copy()
    u1 = SB.ifst(u0, u1)
    assert np.allclose(u1, fj)
    

