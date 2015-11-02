import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from ..fft.wrappyfftw import dct
from cbcdns import config
import SFTc
from ..shen.Helmholtz import TDMA

"""
Fast transforms for pure Chebyshev basis or 
Shen's Chebyshev basis: 

  For homogeneous Dirichlet boundary conditions:
 
    phi_k = T_k - T_{k+2}
    
  For homogeneous Neumann boundary conditions:
    
    phi_k = T_k - (k/k+2)**2 * T_{k+2}

Use either Chebyshev-Gauss (GC) or Gauss-Lobatto (GL)
points in real space.

The ChebyshevTransform may be used to compute derivatives
through fast Chebyshev transforms.

"""
float, complex = np.float64, np.complex128
pi, zeros, ones = np.pi, np.zeros, np.ones

class ChebyshevTransform(object):
    
    def __init__(self, quad="GL"): 
        self.quad = quad
        
    def points_and_weights(self, N):
        self.N = N
        if self.quad == "GL":
            points = (n_cheb.chebpts2(N)[::-1]).astype(float)
            weights = zeros(N)+pi/(N-1)
            weights[0] /= 2
            weights[-1] /= 2

        elif self.quad == "GC":
            points, weights = n_cheb.chebgauss(N)
            points = points.astype(float)
            weights = weights.astype(float)
            
        return points, weights
        
    def chebDerivativeCoefficients(self, fk, fj):
        SFTc.chebDerivativeCoefficients(fk, fj)  
        return fj

    def chebDerivative_3D(self, fj, fd, fk, fkd):
        fk = self.fct(fj, fk)
        fkd = SFTc.chebDerivativeCoefficients_3D(fk, fkd)
        fd = self.ifct(fl, fd)
        return fd
    
    def fastChebDerivative(self, fj, fd, fk, fkd):
        """Compute derivative of fj at the same points."""
        fk = self.fct(fj, fk)
        fkd = self.chebDerivativeCoefficients(fk, fkd)
        fd  = self.ifct(fkd, fd)
        return fd
        
    def fct(self, fj, cj):
        """Fast Chebyshev transform."""
        N = fj.shape[0]
        if self.quad == "GC":
            cj[:] = dct(fj, type=2, axis=0)            
            cj /= N
            cj[0] /= 2
                
        elif self.quad == "GL":
            cj[:] = dct(fj, type=1, axis=0)/(N-1)
            cj[0] /= 2
            cj[-1] /= 2
            
        return cj

    #@profile
    def ifct(self, fk, cj):
        """Inverse fast Chebyshev transform."""
        if self.quad == "GC":
            cj[:] = 0.5*dct(fk, type=3, axis=0)
            cj += 0.5*fk[0]
        
        elif self.quad == "GL":
            cj[:] = 0.5*dct(fk, type=1, axis=0)
            cj += 0.5*fk[0]
            cj[::2] += 0.5*fk[-1]
            cj[1::2] -= 0.5*fk[-1]

        return cj
    
    def fastChebScalar(self, fj, fk):
        """Fast Chebyshev scalar product."""
        N = fj.shape[0]
        if self.quad == "GC":
            fk[:] = dct(fj, type=2, axis=0)*pi/(2*N)
        
        elif self.quad == "GL":
            fk[:] = dct(fj, type=1, axis=0)*pi/(2*(N-1))
        return fk

class ShenDirichletBasis(ChebyshevTransform):
    
    def __init__(self, quad="GL", fast_transform=True):
        self.quad = quad
        self.fast_transform = fast_transform
        self.points = None
        self.weights = None
        self.N = -1
        self.ck = None
        self.w_hat = None
        self.TDMASolver = None
        
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

    def fastShenScalar(self, fj, fk):
                
        if self.fast_transform:
            fk = self.fastChebScalar(fj, fk)
            fk[:-2] -= fk[2:]
        else:
            if self.points is None: self.init(fj.shape[0])
            fk[:-2] = np.dot(self.V, fj*self.weights)
        return fk
        
    def ifst(self, fk, fj):
        """Fast inverse Shen transform
        Transform needs to take into account that phi_k = T_k - T_{k+2}
        fk contains Shen coefficients in the first fk.shape[0]-2 positions
        """
        if self.fast_transform:
            N = len(fj)
            if self.w_hat is None:
                self.w_hat = fk.copy()
                
            self.w_hat[:] = 0
            self.w_hat[:-2] = fk[:-2] 
            self.w_hat[2:] -= fk[:-2] 
            fj = self.ifct(self.w_hat, fj)
            return fj    
        
        else:
            if not self.points: self.init(fj.shape[0])
            return np.dot(self.V.T, fk[:-2])

    def fst(self, fj, fk):
        """Fast Shen transform
        """
        fk = self.fastShenScalar(fj, fk)
        
        N = fj.shape[0]
        if self.TDMASolver is None:
            self.TDMASolver = TDMA(N, self.quad, False)
        fk = self.TDMASolver(fk)
        
        #if self.ck is None:
            #if self.quad == "GC":
                #self.ck = ones(N-2, int)
                #self.ck[0] = 2
                
            #elif self.quad == "GL":
                #self.ck = ones(N-2, int) 
                #self.ck[0] = 2
                #self.ck[-1] = 2  # Note!! Shen paper has only ck[0] = 2, not ck[-1] = 2. 
            
            #self.a = ones(N-4)*(-pi/2)
            #self.b = pi/2*(self.ck+1)
            #self.c = self.a.copy()
            #self.bc = self.b.copy()
            
        #if len(fk.shape) == 3:
            #fk[:-2] = SFTc.TDMA_3D(self.a, self.b, self.bc, self.c, fk[:-2])

        #elif len(fk.shape) == 1:
            #fk[:-2] = SFTc.TDMA_1D(self.a, self.b, self.c, fk[:-2])
            
        return fk
    

class ShenNeumannBasis(ShenDirichletBasis):
    
    def __init__(self, quad="GC"): 
        ShenDirichletBasis.__init__(self, quad)
        self.factor = None        
            
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        k = self.wavenumbers(N)
        # Build Vandermonde matrix. Note! N points in real space gives N-3 bases in spectral space
        self.V = n_cheb.chebvander(self.points, N-3).T - ((k/(k+2))**2)[:, np.newaxis]*n_cheb.chebvander(self.points, N-1)[:, 2:].T
        self.V = self.V[1:, :]

    def fastShenScalar(self, fj, fk):
        """Fast Shen scalar product.
        Chebyshev transform taking into account that phi_k = T_k - (k/(k+2))**2*T_{k+2}
        Note, this is the non-normalized scalar product

        """
        
        if self.fast_transform:
            k  = self.wavenumbers(fj.shape)
            fk = self.fastChebScalar(fj, fk)
            fk[:-2] -= ((k/(k+2))**2) * fk[2:]

        else:
            if self.points is None: self.init(fj.shape[0])
            fk = np.dot(self.V, fj*self.weights)
        return fk

    def ifst(self, fk, fj):
        """Fast inverse Shen scalar transform
        """
        if self.w_hat is None:
            if len(fk.shape)==3:
                k = self.wavenumbers(fk.shape)                
            elif len(fk.shape)==1:
                k = self.wavenumbers(fk.shape[0])
                
            self.w_hat = fk.copy()
            self.factor = (k[1:]/(k[1:]+2))**2
        self.w_hat[:] = 0
        self.w_hat[1:-2] = fk[1:-2]
        self.w_hat[3:] -= self.factor*fk[1:-2]
        fj = self.ifct(self.w_hat, fj)
        return fj
        
    def fst(self, fj, fk):
        """Fast Shen transform.
        """
        fk = self.fastShenScalar(fj, fk)
        N = fj.shape[0]
        if self.TDMASolver is None:
            self.TDMASolver = TDMA(N, self.quad, True)
        fk = self.TDMASolver(fk)

        #if self.ck is None:
            #k = self.wavenumbers(N)
            #self.ck = ones(N-3)
            #if self.quad == "GL": 
                #ck[-1] = 2 # Note not the first since basis phi_0 is not included        
            #self.a = ones(N-5)*(-pi/2)*(k[1:-2]/(k[1:-2]+2))**2
            #self.b = pi/2*(1+self.ck*(k[1:]/(k[1:]+2))**4)
            #self.c = self.a.copy()
            #self.bc = self.b.copy()
            
        #if len(fk.shape) == 3:
            #fk[1:-2] = SFTc.TDMA_3D(self.a, self.b, self.bc, self.c, fk[1:-2])

        #elif len(fk.shape) == 1:
            #fk[1:-2] = SFTc.TDMA_1D(self.a, self.b, self.c, fk[1:-2])

        return fk
    
if __name__ == "__main__":
    N = 8
    a = np.random.random((N, N, N/2+1))+1j*np.random.random((N, N, N/2+1))
    af = zeros((N, N, N/2+1), dtype=a.dtype)
    a[0,:,:] = 0
    a[-1,:,:] = 0
    #a = np.random.random(N)+np.random.random(N)*1j 
    #af = zeros(N, dtype=np.complex)
    #a[0] = 0
    #a[-1] = 0
    
    ST = ShenDirichletBasis(quad="GL")
    af = ST.fst(a, af) 
    a0 = a.copy()
    a0 = ST.ifst(af, a0)
    assert np.allclose(a0, a)
