import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from scipy.fftpack import dct, idct
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import SFTc
from numpy import linalg, inf
import time

"""
Fast transforms for pure Chebyshev basis or 
Shen's Chebyshev basis: 

  phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},

where for homogeneous Dirichlet boundary conditions:
 
    a_k = 0  and  b_k = -1
    
For homogeneous Neumann boundary conditions:
    
     a_k = 0  and  b_k = -(k/k+2)**2 
     
For Robin/mixed boundary conditions:

     a_k = \pm 4*(k+1)/((k+1)**2 + (k+2)**2)  and  
     b_k = -(k**2 + (k+1)**2)/((k+1)**2 + (k+2)**2)

a_k is positive for Dirichlet BC at x = -1 and Neumann BC at x = +1 (DN),
and it is negative for Neumann BC at x = -1 and Dirichlet BC at x = +1 (ND)

It is therefore possible to choose DN boundary conditions (BC = "DN")
or ND boundary conditions (BC = "ND").

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
        if self.quad == "GC":
            points = n_cheb.chebpts2(N)[::-1]
            weights = np.zeros((N))+np.pi/(N-1)
            weights[0] /= 2
            weights[-1] /= 2

        elif self.quad == "GL":
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
        if self.quad == "GL":
            cj = dct(fj, 2, axis=0)
            cj /= N
            cj[0] /= 2
                
        elif self.quad == "GC":
            cj = dct(fj, 1, axis=0)/(N-1)
            cj[0] /= 2
            cj[-1] /= 2
            
        return cj

    def ifct(self, fk, cj):
        """Inverse fast Chebyshev transform."""
        if self.quad == "GL":
            cj = 0.5*dct(fk, 3, axis=0)
            cj += 0.5*fk[0]
        
        elif self.quad == "GC":
            cj = 0.5*dct(fk, 1, axis=0)
            cj += 0.5*fk[0]
            cj[::2] += 0.5*fk[-1]
            cj[1::2] -= 0.5*fk[-1]

        return cj
    
    def fastChebScalar(self, fj, fk):
        """Fast Chebyshev scalar product."""
        N = fj.shape[0]
        if self.quad == "GL":
            fk = dct(fj, 2, axis=0)*np.pi/(2*N)
        
        elif self.quad == "GC":
            fk = dct(fj, 1, axis=0)*np.pi/(2*(N-1))
        return fk

class ShenDirichletBasis(ChebyshevTransform):
    
    def __init__(self, quad="GC", fast_transform=True):
        self.quad = quad
        self.fast_transform = fast_transform
        self.points = None
        self.weights = None
        self.N = -1
        
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
            return np.arange(N-2).astype(np.float)
        
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
            if len(fk.shape) == 3:
                w_hat = np.zeros(fj.shape, dtype=fk.dtype)
                w_hat[:-2] = fk[:-2]
                w_hat[2:] -= fk[:-2]
            if len(fk.shape) == 1:
                w_hat = np.zeros(N)                
                w_hat[:-2] = fk[:-2] 
                w_hat[2:] -= fk[:-2] 
                #- np.hstack([0, 0, fk[:-2]])    
                #w_hat[-2] = -fk[-2]
                #w_hat[-1] = -fk[-1]
            fj = self.ifct(w_hat, fj)
            return fj    
        
        else:
            if not self.points: self.init(fj.shape[0])
            return np.dot(self.V.T, fk[:-2])

    def fst(self, fj, fk):
        """Fast Shen transform
        """
        fk = self.fastShenScalar(fj, fk)
        
        N = fj.shape[0]
        if self.quad == "GL":
            ck = np.ones(N-2); ck[0] = 2
            
        elif self.quad == "GC":
            ck = np.ones(N-2); ck[0] = 2; ck[-1] = 2  # Note!! Shen paper has only ck[0] = 2, not ck[-1] = 2. For Gauss points ck[-1] = 1, but not here! 
            
        a = np.ones(N-4)*(-np.pi/2)
        b = np.pi/2*(ck+1)
        c = a.copy()
        if len(fk.shape) == 3:
            bc = b.copy()
            fk[:-2] = SFTc.TDMA_3D_complex(a, b, bc, c, fk[:-2])

        elif len(fk.shape) == 1:
            fk[:-2] = SFTc.TDMA_1D(a, b, c, fk[:-2])
            
        return fk
    

class ShenNeumannBasis(ShenDirichletBasis):
    
    def __init__(self, quad="GL"): 
        ShenDirichletBasis.__init__(self, quad)
            
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
        if len(fk.shape)==3:
            k = self.wavenumbers(fk.shape)
            w_hat = np.zeros(fk.shape, dtype=fk.dtype)
            w_hat[1:-2] = fk[1:-2]
            w_hat[3:] -= (k[1:]/(k[1:]+2))**2*fk[1:-2]
            
        elif len(fk.shape)==1:
            k = self.wavenumbers(fk.shape[0])
            w_hat = np.zeros(fk.shape[0])    
            w_hat[1:-2] = fk[1:-2]
            w_hat[3:] -= (k[1:]/(k[1:]+2))**2*fk[1:-2]

        fj = self.ifct(w_hat, fj)
        return fj
        
    def fst(self, fj, fk):
        """Fast Shen transform.
        """
        fk = self.fastShenScalar(fj, fk)
        N = fj.shape[0]
        k = self.wavenumbers(N)
        ck = np.ones(N-3)
        if self.quad == "GC": ck[-1] = 2 # Note not the first since basis phi_0 is not included        
        a = np.ones(N-5)*(-np.pi/2)*(k[1:-2]/(k[1:-2]+2))**2
        b = np.pi/2*(1+ck*(k[1:]/(k[1:]+2))**4)
        c = a.copy()
        if len(fk.shape) == 3:
            bc = b.copy()
            fk[1:-2] = SFTc.TDMA_3D_complex(a, b, bc, c, fk[1:-2])
	    
        elif len(fk.shape) == 1:
            fk[1:-2] = SFTc.TDMA_1D(a, b, c, fk[1:-2])

        return fk


class ShenRobinBasis(ShenDirichletBasis):
    
    def __init__(self, quad="GL", BC = "ND"): 
	self.BC = BC
        ShenDirichletBasis.__init__(self, quad)
               
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        k = self.wavenumbers(N)

    def shenCoefficients(self, k):
	"""
	Shen basis functions given by
	phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},
        satisfy the imposed Robin (mixed) boundary conditions for a unique set of {a_k, b_k}.  
	"""
	if self.BC == "ND":
	    ak = -4*(k+1)/((k+1)**2 + (k+2)**2)
	elif self.BC == "DN":
	    ak = 4*(k+1)/((k+1)**2 + (k+2)**2)
	bk = -((k**2 + (k+1)**2)/((k+1)**2 + (k+2)**2))
        return ak, bk
    
    def fastShenScalar(self, fj, fk):
        """Fast Shen scalar product 
        B u_hat = sum_{j=0}{N} u_j phi_k(x_j) w_j,
        for Shen basis functions given by
	phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2}
        """
        if self.fast_transform:
            k  = self.wavenumbers(fj.shape)
            fk = self.fastChebScalar(fj, fk)
            ak, bk = self.shenCoefficients(k)
            
            fk_tmp = fk
            fk[:-2] = fk_tmp[:-2] + ak*fk_tmp[1:-1] + bk*fk_tmp[2:]

        return fk

    def ifst(self, fk, fj):
        """Fast inverse Shen scalar transform for Robin BC.
        """
        if len(fk.shape)==3:
            k = self.wavenumbers(fk.shape)
            w_hat = np.zeros(fk.shape, dtype=fk.dtype)
        elif len(fk.shape)==1:
	    k = self.wavenumbers(fk.shape[0])
            w_hat = np.zeros(fk.shape[0])
	ak, bk = self.shenCoefficients(k)
	w_hat[:-2] = fk[:-2]
	w_hat[1:-1] += ak*fk[:-2]
	w_hat[2:]   += bk*fk[:-2]
            
        fj = self.ifct(w_hat, fj)
        return fj
        
    def fst(self, fj, fk):
        """Fast Shen transform for Robin BC.
        """
        fk = self.fastShenScalar(fj, fk)
        N = fj.shape[0]
        k = self.wavenumbers(N) 
        k1 = self.wavenumbers(N+1) 
        ak, bk = self.shenCoefficients(k)
        ak1, bk1 = self.shenCoefficients(k1)
        
        if self.quad == "GL":
            ck = ones(N-2); ck[0] = 2
        elif self.quad == "GC":
            ck = ones(N-2); ck[0] = 2; ck[-1] = 2  
        
        a = (pi/2)*(ck + ak**2 + bk**2)
        b = ones(N-3)*(pi/2)*(ak[:-1] + ak1[1:-1]*bk[:-1])
        c = ones(N-4)*(pi/2)* bk[:-2]
        
        """
        Here we use splu to solve  B u = f_k,
        where B is the pentadiagonal mass matrix and f_k is the Shen scalar product 
        """
        if len(fk.shape) == 3:
	    fk[:-2] = SFTc.PDMA_3D_complex(a, b, c, fk[:-2])
   
        elif len(fk.shape) == 1:
	    fk[:-2] = SFTc.PDMA_1D(a, b, c, fk[:-2])	    
	return fk    
    
if __name__ == "__main__":
    
    N = 2**12
    af = np.zeros(N, dtype=np.complex)
    SR = ShenRobinBasis(quad="GC", BC="ND")
    pointsr, weightsr = SR.points_and_weights(N)
    x = pointsr
    a = x -(8./13.)*(-1 + 2*x**2) - (5./13.)*(-3*x + 4*x**3) # Chebyshev polynomial that satisfies the Robin BC
    
    af = SR.fst(a, af)
    a0 = a.copy()
    a0 = SR.ifst(af, a0)
    print "Error in Shen-Robin transform: ",linalg.norm((a - a0), inf) 
    assert np.allclose(a0, a)
     