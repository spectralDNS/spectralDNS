# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:36:11 2015

@author: Diako Darian
"""

import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from scipy.fftpack import dct, idct
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
from scipy.sparse.linalg import splu
#import SFTc 
from TDMA import TDMA_1D, TDMA_3D_complex
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

class ElsasserPlusBasis(ChebyshevTransform):
    
    def __init__(self, quad="GC"):
        self.quad = quad
        self.points = None
        self.weights = None
        self.N = -1

    def wavenumbers(self, N):
        if isinstance(N, tuple):
            if len(N) == 1:
                N = N[0]
        if isinstance(N, int): 
            return np.arange(N-2).astype(np.float)
        
        else:
            kk = np.mgrid[:N[0]-2, :N[1], :N[2]].astype(float)
            return kk[0]

    def shenCoefficients(self, k):
	"""
	Shen coeffiecient for the Elsässer variable z^{+} 
	from the basis functions given by
	phi_k = 2*T_k + 2*a_k*T_{k+2}.  
	"""
	ak = -2*(k**2+2*k+2)/((k+2)**2)
	return ak

    def fastShenScalar(self, fj, fk):
        """
        Fast Shen scalar product 
        B z^{+}_hat = sum_{j=0}{N} ^{+}_j phi_k(x_j) w_j,
        for Shen basis functions given by
        phi_k = 2*T_k + 2*a_k*T_{k+2}
        """
        k  = self.wavenumbers(fj.shape)
	fk = self.fastChebScalar(fj, fk)
	ak = self.shenCoefficients(k)
	
	fk_tmp = fk
	fk[:-2] = 2*fk_tmp[:-2] + ak*fk_tmp[2:]

        return fk
        
    def ifst(self, fk, fj):
	"""Fast inverse Shen scalar transform the Elsässer variable z^{+}.
	"""
	if len(fk.shape)==3:
	    k = self.wavenumbers(fk.shape)
	    w_hat = zeros(fk.shape, dtype=fk.dtype)
	elif len(fk.shape)==1:
	    k = self.wavenumbers(fk.shape[0])
	    w_hat = zeros(fk.shape[0])
	ak = self.shenCoefficients(k)
	w_hat[:-2] = 2*fk[:-2]
	w_hat[2:] += ak*fk[:-2]

	fj = self.ifct(w_hat, fj)
	return fj
    
    def fst(self, fj, fk):
	"""Fast Shen transform for the Elsässer variable z^{+}.
	"""
	fk = self.fastShenScalar(fj, fk)
	N = fj.shape[0]
	k = self.wavenumbers(N)  
	ak = self.shenCoefficients(k)
	
	if self.quad == "GL":
	    ck = ones(N-2); ck[0] = 2
	elif self.quad == "GC":
	    ck = ones(N-2); ck[0] = 2; ck[-1] = 2  

	a = ones(N-4)*(pi*ak[:-2])
	b = (pi/2.)*(4.0*ck+ak**2)
	c = a.copy()
	
	if len(fk.shape) == 3:
	    bc = b.copy()
	    fk[:-2] = TDMA_3D_complex(a, b, bc, c, fk[:-2])
	elif len(fk.shape) == 1:
	    fk[:-2] = TDMA_1D(a, b, c, fk[:-2])

	return fk  
            
class ElsasserMinusBasis(ElsasserPlusBasis):
    
    def __init__(self, quad="GL"): 
        ElsasserPlusBasis.__init__(self, quad)
            
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        k = self.wavenumbers(N)
        # Build Vandermonde matrix. Note! N points in real space gives N-3 bases in spectral space
        self.V = n_cheb.chebvander(self.points, N-3).T - ((k/(k+2))**2)[:, np.newaxis]*n_cheb.chebvander(self.points, N-1)[:, 2:].T
        self.V = self.V[1:, :]
        
    def shenCoefficients(self, k):
	"""
	Shen coeffiecient for the Elsässer variable z^{-} 
	from the basis functions given by
	phi_k = b_k*T_{k+2}.  
	"""
	bk = (-4.*(k+1.))/((k+2.)**2)
	
	return bk

    def fastShenScalar(self, fj, fk):
	"""
	Fast Shen scalar product 
	B z^{-}_hat = sum_{j=0}{N} z^{-}_j phi_k(x_j) w_j,
	for Shen basis functions given by
	phi_k = b_k*T_{k+2}
	"""
	
	k  = self.wavenumbers(fj.shape)
	fk = self.fastChebScalar(fj, fk)
	bk = self.shenCoefficients(k)
	
	fk_tmp = fk
	fk[:-2] = bk*fk_tmp[2:]

	return fk

    def ifst(self, fk, fj):
	"""Fast inverse Shen scalar transform for the Elsässer variable z^{-}.
	"""
	if len(fk.shape)==3:
	    k = self.wavenumbers(fk.shape)
	    w_hat = zeros(fk.shape, dtype=fk.dtype)
	elif len(fk.shape)==1:
	    k = self.wavenumbers(fk.shape[0])
	    w_hat = zeros(fk.shape[0])
	bk = self.shenCoefficients(k)
	w_hat[2:] = bk*fk[:-2]
	
	fj = self.ifct(w_hat, fj)
	return fj
        
    def fst(self, fj, fk):
	"""Fast Shen transform for the Elsässer variable z^{-}.
	"""
	fk = self.fastShenScalar(fj, fk)
	N = fj.shape[0]
	k = self.wavenumbers(N)  
	bk = self.shenCoefficients(k)

	a = ones(N-2)*(pi/2.)*bk**2
	
	if len(fk.shape) == 3:
	    for i in range(fk.shape[0]-2):
		for j in range(fk.shape[1]):
		    for k in range(fk.shape[2]):
			fk[i, j, k] = fk[i, j, k]/a[i]
	elif len(fk.shape) == 1:
	    for i in range(N-2):
		fk[i] = fk[i]/a[i]    
	return fk  
    
if __name__ == "__main__":
    
    N = 2**8
    am = np.zeros(N, dtype=np.complex)
    SM = ElsasserMinusBasis(quad="GL")
    pointsr, weightsr = SM.points_and_weights(N)    
    x = pointsr
    v = -(8./9.)*(-3.*x + 4.*x**3)
    am = SM.fst(v, am)
    v0 = np.zeros(N)
    v0 = SM.ifst(am, v0)
    
    #ap = np.zeros(N, dtype=np.complex)
    #SP = ElsasserPlusBasis(quad="GL")    
    #pointsr, weightsr = SP.points_and_weights(N)    
    #x = pointsr
    #u = -(40./9.)*x**3 + (16./3.)*x
    #u0 = np.zeros(N)
    #ap = SP.fst(u, ap)
    #u0 = SP.ifst(ap, u0)
    
    print "Error in 1D ElsasserMinusBasis transforms = ", linalg.norm(v-v0)  
    #print "Error in 1D ElsasserPlusBasis transforms = ", linalg.norm(u-u0) 