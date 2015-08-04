import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from scipy.fftpack import dct, idct
import SFTc

"""
Fast transforms for pure Chebyshev basis or 
Shen's Chebyshev basis: 

  For homogeneous Dirichlet boundary conditions:
 
    phi_k = T_k - T_{k+2}
    
  For homogeneous Neumann boundary conditions:
    
    phi_k = T_k - (k/k+2)**2 * T_{k+2}

Use either Chebyshev-Gauss or Gauss-Lobatto points for
Dirichlet basis, but only Chebyshev-Gauss for Neumann.

The ChebyshevTransform may be used to compute derivatives
through fast Chebyshev transforms.

"""

dct1 = dct
def dct(x, i, axis=0):
    if np.iscomplexobj(x):
        xreal = dct1(x.real, i, axis=axis)
        ximag = dct1(x.imag, i, axis=axis)
        return xreal + ximag*1j
    else:
        return dct1(x, i, axis=axis)

class ChebyshevTransform(object):
    
    def __init__(self, quad="GC"): 
        self.quad = quad
        
    def chebDerivativeCoefficients(self, fk):
        N = fk.shape[0]-1
        fl = fk.copy()
        SFTc.chebDerivativeCoefficients(fk, fl)
        #fl[-1] = 0
        #fl[-2] = 2*N*fk[-1]
        #for k in range(N-2, 0, -1):
            #fl[k] = 2*(k+1)*fk[k+1]+fl[k+2]
        #fl[0] = fk[1] + 0.5*fl[2]
                        
        return fl
    
    def fastChebDerivative(self, fj):
        fk = self.fastChebTrans(fj)
        fl = self.chebDerivativeCoefficients(fk)
        df  = self.ifastChebTrans(fl)
        return df
        
    def fct(self, a):
        return self.fastChebTrans(a)
    
    def ifct(self, a):
        return self.ifastChebTrans(a)

    def fastChebTrans(self, fj):
        N = fj.shape[0]
        if self.quad == "GL":
            cj = dct(fj, 2, axis=0)
            cj /= N
            cj[0] /= 2
                
        elif self.quad == "GC":
            cj = dct(fj, 1, axis=0)/(N-1)
            cj[0] /= 2
            cj[-1] /= 2
            
        return cj

    def ifastChebTrans(self, fk):
        if self.quad == "GL":
            cj = 0.5*dct(fk, 3, axis=0)
            cj += 0.5*fk[0]
        
        elif self.quad == "GC":
            cj = 0.5*dct(fk, 1, axis=0)
            cj += 0.5*fk[0]
            cj[::2] += 0.5*fk[-1]
            cj[1::2] -= 0.5*fk[-1]

        return cj
    
    def fastChebScalar(self, fj):
        N = fj.shape[0]
        if self.quad == "GL":
            return dct(fj, 2, axis=0)*np.pi/(2*N)
        
        elif self.quad == "GC":
            return dct(fj, 1, axis=0)*np.pi/(2*(N-1))
        

class ShenDirichletBasis(ChebyshevTransform):
    
    def __init__(self, quad="GC", fast_transform=True):
        self.quad = quad
        self.fast_transform = fast_transform
        self.points = None
        self.weights = None
        self.N = -1
        
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        # Build Vandermonde matrix. Note! N points in real space gives N-2 bases in spectral space
        self.V = n_cheb.chebvander(self.points, N-3).T - n_cheb.chebvander(self.points, N-1)[:, 2:].T

    def points_and_weights(self, N):
        self.N = N
        if self.quad == "GC":
            points = n_cheb.chebpts2(N)[::-1]
            weights = np.zeros((N))+np.pi/(N-1)
            weights[0] /= 2
            weights[-1] /= 2

        elif self.quad == "GL":
            points, weights = n_cheb.chebgauss(N)
            
        return points, weights
    
    def wavenumbers(self, N, dim=1):
        if dim == 1:            
            return np.arange(N-2).astype(np.float)
        
        else:
            kk = np.mgrid[:N-2, :N, :N/2+1].astype(float)
            return kk[0]
        
    def fastShenScalar(self, fj):
                
        if self.fast_transform:
            u_k = self.fastChebScalar(fj)
            u_k[:-2] -= u_k[2:]
            return u_k[:-2]
        else:
            if not self.points: self.init(fj.shape[0])
            return np.dot(self.V, fj*self.weights)
        
    def ifastShenScalar(self, fk):
        """Fast inverse Shen scalar product
        Transform needs to take into account that phi_k = T_k - T_{k+2}
        
        """
        if self.fast_transform:
            N = len(fk)+2
            if len(fk.shape) == 3:
                w_hat = np.zeros((fk.shape[0]+2, fk.shape[1], fk.shape[2]), dtype=fk.dtype)
                w_hat[:-2] = fk
                w_hat[2:-2] -= fk[:-2]
            if len(fk.shape) == 1:
                w_hat = np.zeros(N)                
                w_hat[:-2] = fk - np.hstack([0, 0, fk[:-2]])    
            
            w_hat[-2] = -fk[-2]
            w_hat[-1] = -fk[-1]
            fj = self.ifastChebTrans(w_hat)            
            return fj    
        
        else:
            if not self.points: self.init(fj.shape[0])
            return np.dot(self.V.T, fk)

    def fst(self, fj):
        """Fast Shen transform
        """
        cj = self.fastShenScalar(fj)
        
        N = fj.shape[0]
        if self.quad == "GL":
            ck = np.ones(N-2); ck[0] = 2
            
        elif self.quad == "GC":
            ck = np.ones(N-2); ck[0] = 2; ck[-1] = 2  # Note!! Shen paper has only ck[0] = 2, not ck[-1] = 2. For Gauss points ck[-1] = 1, but not here! 
            
        a = np.ones(N-4)*(-np.pi/2)
        b = np.pi/2*(ck+1)
        c = a.copy()
        if len(cj.shape) == 3:
            bc = b.copy()
            cj = SFTc.TDMA_3D_complex(a, b, bc, c, cj)

        elif len(cj.shape) == 1:
            cj = SFTc.TDMA_1D(a, b, c, cj)
            
        return cj
    
    def ifst(self, fj):
        """Inverse fast Shen transform
        """
        return self.ifastShenScalar(fj)
    

class ShenNeumannBasis(ShenDirichletBasis):
    
    def __init__(self): 
        ShenDirichletBasis.__init__(self, "GC")
            
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        k = self.wavenumbers(N)
        # Build Vandermonde matrix. Note! N points in real space gives N-3 bases in spectral space
        self.V = n_cheb.chebvander(self.points, N-3).T - ((k/(k+2))**2)[:, np.newaxis]*n_cheb.chebvander(self.points, N-1)[:, 2:].T
        self.V = self.V[1:, :]

    def fastShenScalar(self, fj):
        """Fast Shen scalar product on cos(j*pi/N).
        Chebyshev transform taking into account that phi_k = T_k - (k/(k+2))**2*T_{k+2}
        Note, this is the non-normalized scalar product

        """
        N = fj.shape[0]
        k = self.wavenumbers(N, dim=len(fj.shape))
        ck = dct(fj, 1, axis=0)
        ck *= (np.pi/((N-1)*2))
        ck[:-2] -= ((k/(k+2))**2) * ck[2:]
        return ck[1:-2]

    def ifastShenScalar(self, fk):
        """Fast inverse Shen scalar product
        """
        N = fk.shape[0]+3
        k = self.wavenumbers(N, dim=len(fk.shape))
        if len(fk.shape)==3:
            w_hat = np.zeros((N, fk.shape[1], fk.shape[2]), dtype=fk.dtype)
            w_hat[1:-2] = fk
            w_hat[3:] -= (k[1:]/(k[1:]+2))**2*fk
            
        if len(fk.shape)==1:
            w_hat = np.zeros(N)       
            w_hat[1:-2] = fk
            w_hat[3:] -= (k[1:]/(k[1:]+2))**2*fk

        fj = 0.5*dct(w_hat, 1, axis=0)
        fj[::2] += 0.5*w_hat[-1]
        fj[1::2] -= 0.5*w_hat[-1]
        return fj
        
    def fst(self, fj):
        """Fast Shen transform.
        """
        cj = self.fastShenScalar(fj)
        N = fj.shape[0]
        k = self.wavenumbers(N)
        ck = np.ones(N-3); ck[-1] = 2 # Note not the first since basis phi_0 is not included        
        a = np.ones(N-5)*(-np.pi/2)*(k[1:-2]/(k[1:-2]+2))**2
        b = np.pi/2*(1+(k[1:]/(k[1:]+2))**4)
        c = a.copy()
        if len(cj.shape) == 3:
            bc = b.copy()
            cj = SFTc.TDMA_3D_complex(a, b, bc, c, cj)

        elif len(cj.shape) == 1:
            cj = SFTc.TDMA_1D(a, b, c, cj)

        return cj
    
if __name__ == "__main__":
    N = 8
    #a = np.random.random((N, N, N/2+1))+1j*np.random.random((N, N, N/2+1))
    #a = np.random.random((N, N, N/2+1))
    a = np.random.random(N)
    #a[0,:,:] = 0
    #a[-1,:,:] = 0
    a[0] = 0
    a[-1] = 0
    
    ST = ShenDirichletBasis()
    fst1 = ST.ifst(ST.fst(a))
    assert np.allclose(fst1, a)
        
     