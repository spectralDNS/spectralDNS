import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from scipy.fftpack import dct, idct
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import SFTc
from numpy import linalg, inf

"""
Fast transforms for pure Chebyshev basis or 
Shen's Chebyshev basis: 

  phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},

where a_k and b_k are calculated from 

   1) a_minus U(-1) + b_minus U(-1) = c_minus 
and
   2) a_plus U(1) + b_plus U(1) = c_plus

The array BC = [a_minus, b_minus, c_minus, a_plus, b_plus, c_plus] that determines the 
boundary conditions must be given. The code automatically calculates a_k and b_k, and it gives the
Shen transform.

In particular, for homogeneous Dirichlet boundary conditions we have:
 
    a_k = 0  and  b_k = -1
    
For homogeneous Neumann boundary conditions:
    
     a_k = 0  and  b_k = -(k/k+2)**2 
     
For Robin boundary conditions:

     a_k = \pm 4*(k+1)/((k+1)**2 + (k+2)**2)  and  
     b_k = -(k**2 + (k+1)**2)/((k+1)**2 + (k+2)**2)

Here a_k is positive for Dirichlet BC at x = -1 and Neumann BC at x = +1,
and it is negative for Neumann BC at x = -1 and Dirichlet BC at x = +1.

Use either Chebyshev-Gauss (GC) or Gauss-Lobatto (GL)
points in real space.

The ChebyshevTransform may be used to compute derivatives
through fast Chebyshev transforms.
"""
pi, zeros, ones = np.pi, np.zeros, np.ones

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
        
    def points_and_weights(self, N):
        self.N = N
        if self.quad == "GL":
            points = n_cheb.chebpts2(N)[::-1]
            weights = np.zeros((N))+np.pi/(N-1)
            weights[0] /= 2
            weights[-1] /= 2
        elif self.quad == "GC":
            points, weights = n_cheb.chebgauss(N)
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
            cj = dct(fj, 2, axis=0)
            cj /= N
            cj[0] /= 2        
        elif self.quad == "GL":
            cj = dct(fj, 1, axis=0)/(N-1)
            cj[0] /= 2
            cj[-1] /= 2
        return cj

    def ifct(self, fk, cj):
        """Inverse fast Chebyshev transform."""
        if self.quad == "GC":
            cj = 0.5*dct(fk, 3, axis=0)
            cj += 0.5*fk[0]
        elif self.quad == "GL":
            cj = 0.5*dct(fk, 1, axis=0)
            cj += 0.5*fk[0]
            cj[::2] += 0.5*fk[-1]
            cj[1::2] -= 0.5*fk[-1]
        return cj
    
    def fastChebScalar(self, fj, fk):
        """Fast Chebyshev scalar product."""
        N = fj.shape[0]
        if self.quad == "GC":
            fk = dct(fj, 2, axis=0)*np.pi/(2*N)
        elif self.quad == "GL":
            fk = dct(fj, 1, axis=0)*np.pi/(2*(N-1))
        return fk

class ShenBasis(ChebyshevTransform):

    def __init__(self, BC, quad="GC"):
        self.quad = quad
        self.BC = BC
        self.points = None
        self.weights = None
        self.N = -1
        
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)

    def wavenumbers(self, N):
        if isinstance(N, tuple):
            if len(N) == 1:
                N = N[0]
        if isinstance(N, int): 
            return np.arange(N-2).astype(np.float)
        else:
            kk = np.mgrid[:N[0]-2, :N[1], :N[2]].astype(float)
            return kk[0]

    def chebNormalizationFactor(self, N, quad):
        if self.quad == "GC":
            ck = ones(N[0]-2); ck[0] = 2
        elif self.quad == "GL":
            ck = ones(N[0]-2); ck[0] = 2; ck[-1] = 2
        return ck
    
    def shenCoefficients(self, k, BC):
        """
        Shen basis functions given by
        phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},
        satisfy the imposed boundary conditions for a unique set of {a_k, b_k}.  
        """
        am = BC[0]; bm = BC[1]; cm = BC[2]
        ap = BC[3]; bp = BC[4]; cp = BC[5]
        
        detk = 2*am*ap + ((k + 1.)**2 + (k + 2.)**2)*(am*bp - ap*bm) - 2.*bm*bp*(k + 1.)**2*(k + 2.)**2

        Aa = am - bm*(k + 2.)**2; Ab= -ap - bp*(k + 2.)**2  
        Ac = am - bm*(k + 1.)**2; Ad= ap + bp*(k + 1.)**2
        
        y1 = -ap - bp*k**2 + cp; y2= -am + bm*k**2 + cm/((-1)**k) 
        
        ak = (1./detk)*(Aa*y1 + Ab*y2)
        bk = (1./detk)*(Ac*y1 + Ad*y2)
        
        return ak, bk

    def fastShenScalar(self, fj, fk):
        """Fast Shen scalar product 
        B u_hat = sum_{j=0}{N} u_j phi_k(x_j) w_j,
        for Shen basis functions given by
        phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2}
        """
        k  = self.wavenumbers(fj.shape)
        fk = self.fastChebScalar(fj, fk)
        ak, bk = self.shenCoefficients(k, self.BC)
        
        fk_tmp = fk
        fk[:-2] = fk_tmp[:-2] + ak*fk_tmp[1:-1] + bk*fk_tmp[2:]

        return fk

    def ifst(self, fk, fj):
        """Fast inverse Shen scalar transform for general BC.
        """
        if len(fk.shape)==3:
            k = self.wavenumbers(fk.shape)
            w_hat = np.zeros(fk.shape, dtype=fk.dtype)
        elif len(fk.shape)==1:
            k = self.wavenumbers(fk.shape[0])
            w_hat = np.zeros(fk.shape[0])
        ak, bk = self.shenCoefficients(k, self.BC)
        w_hat[:-2] = fk[:-2]
        w_hat[1:-1] += ak*fk[:-2]
        w_hat[2:]   += bk*fk[:-2]
            
        if self.BC[0]==0 and self.BC[1]==1 and self.BC[2]==0 and self.BC[3]==0 and self.BC[4]==1 and self.BC[5]==0:
            w_hat[0] = 0.0
        fj = self.ifct(w_hat, fj)
        return fj
        
    def fst(self, fj, fk):
        """Fast Shen transform for general BC.
        """
        fk = self.fastShenScalar(fj, fk)
        N = fj.shape[0]
        k = self.wavenumbers(N) 
        k1 = self.wavenumbers(N+1) 
        ak, bk = self.shenCoefficients(k, self.BC)
        ak1, bk1 = self.shenCoefficients(k1, self.BC)
        
        if self.quad == "GC":
            ck = ones(N-2); ck[0] = 2
        elif self.quad == "GL":
            ck = ones(N-2); ck[0] = 2; ck[-1] = 2  
        
        a = (pi/2)*(ck + ak**2 + bk**2)
        b = ones(N-3)*(pi/2)*(ak[:-1] + ak1[1:-1]*bk[:-1])
        c = ones(N-4)*(pi/2)* bk[:-2]

        if len(fk.shape) == 3:
            if self.BC[0]==0 and self.BC[1]==1 and self.BC[2]==0 and self.BC[3]==0 and self.BC[4]==1 and self.BC[5]==0:
                fk[1:-2] = SFTc.PDMA_3D_complex(a[1:], b[1:], c[1:], fk[1:-2])
            else:
                fk[:-2] = SFTc.PDMA_3D_complex(a, b, c, fk[:-2])
        elif len(fk.shape) == 1:
            if self.BC[0]==0 and self.BC[1]==1 and self.BC[2]==0 and self.BC[3]==0 and self.BC[4]==1 and self.BC[5]==0:
                fk[1:-2] = SFTc.PDMA_1D(a[1:], b[1:], c[1:], fk[1:-2])
            else:
                fk[:-2] = SFTc.PDMA_1D(a, b, c, fk[:-2])
        
        return fk    

  
if __name__ == "__main__":
    
    N = 2**6
    BC = np.array([0,1,0, 1,0,0])
    af = np.zeros(N, dtype=np.complex)
    SR = ShenBasis(BC, quad="GC")
    pointsr, weightsr = SR.points_and_weights(N)
    x = pointsr
    a = x -(8./13.)*(-1 + 2*x**2) - (5./13.)*(-3*x + 4*x**3) # Chebyshev polynomial that satisfies the Robin BC
    
    af = SR.fst(a, af)
    a0 = a.copy()
    a0 = SR.ifst(af, a0)
    print("Error in Shen-Robin transform: {}".format(linalg.norm((a - a0), inf))) 
    # Out: Error in Shen-Robin transform: 4.57966997658e-16
    assert np.allclose(a0, a)
