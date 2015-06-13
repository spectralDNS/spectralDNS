from numpy.polynomial import chebyshev as n_cheb
from sympy import chebyshevt, Symbol, sin, cos, pi, lambdify, sqrt as Sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.sparse import diags
import scipy.sparse.linalg as la
from scipy.fftpack import dct, idct
import TDMA

"""
Solve Poisson equation on (-1, 1) with homogeneous bcs

    \nabla^2 u = f, u(\pm 1) = 0
    
Use basis \phi_k = T_k - T_{k+2}, where T_k is k'th Chebyshev 
polynomial of first kind. Solve using spectral Galerkin and the
weighted L_w norm (u, v)_w = \int_{-1}^{1} u v / \sqrt(1-x^2) dx

    (\nabla^ u, \phi_k)_w = (f, \phi_k)_w 
    

"""

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
u = (1-x**2)**2*cos(np.pi*x)*(x-0.25)**2
f = u.diff(x, 2) 

# Choices
banded = True
fast_transform = True

N = 20
k = np.arange(N-2)
points, weights = n_cheb.chebgauss(N)

# Note! N points in real space gives N-2 bases in spectral space

# Build Vandermonde matrix for Gauss-Cheb points and weights
V = n_cheb.chebvander(points, N-3).T - n_cheb.chebvander(points, N-1)[:, 2:].T

# Gauss-Chebyshev quadrature to compute rhs
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

def fastShenTrans(fj):
    """Fast Shen transform on cos(j*pi/N).
    """
    cj = forwardtransform(fj, fast_transform)
    ck = np.ones(N-2); ck[0] = 2    
    a = np.ones(N-4)*(-np.pi/2)
    b = np.pi/2*(ck+1)
    c = a.copy()
    cj = TDMA.TDMA_offset(a, b, c, cj)
    return cj

def ifastShenTrans(fj):
    """Inverse fast Shen transform on cos(j*pi/N).
    """
    cj = backtransform(fj, fast_transform)
    return cj

def forwardtransform(u, fast=True):
    if fast:
        # Chebyshev transform taking into account that phi_k = T_k - T_{k+2}
        # 
        #   u_k = (u, T_k - T_{k+2})_w, for k = 0, 1, ..., N-1
        
        #u_k = idct(u, 3, axis=0, norm=None)*np.pi/2./len(u)
        u_k = dct(u, 2, axis=0, norm=None)*np.pi/2./len(u)
        #u_k = real(fft(u, axis=0, norm=None))*np.pi/2./len(u)
        u_k[:-2] -= u_k[2:]
        u_k = u_k[:-2]
    
    else:
        # Alternatively using Vandermonde matrix
        u_k = np.dot(V, u*weights)
        
    return u_k

f_hat = forwardtransform(fj, fast_transform)

#@profile
def solve(banded=True):
    
    if banded:
        A = np.zeros((N-2, N-2))
        A[-1, :] = -2*np.pi*(k+1)*(k+2)
        for i in range(2, N-2, 2):
            A[-i-1, i:] = -4*np.pi*(k[:-i]+1)
        uk_hat = solve_banded((0, N-3), A, f_hat)
        
    else:
        aij = [-2*np.pi*(k+1)*(k+2)]
        for i in range(2, N-2, 2):
            aij.append(np.array(-4*np.pi*(k[:-i]+1)))    
        A = diags(aij, range(0, N-2, 2))
        uk_hat = la.spsolve(A, f_hat)

    return uk_hat, A

uk_hat, A = solve(banded)

# Back transform
#@profile
def backtransform(u_hat, fast=True):
    if fast:
        # Transform needs to take into account that phi_k = T_k - T_{k+2}
        #
        #   u = sum_{k=0}^{N-1} u_k T_k - sum_{k=0}^{N-1} u_k T_{k+2}
        #   
        w_hat = np.zeros(N)
        w_hat[:-2] = u_hat - np.hstack([0, 0, u_hat[:-2]])
        w_hat[-2] -= u_hat[-2]
        w_hat[-1] -= u_hat[-1]
        
        # Chebyshev transform
        w_hat[1:] /= 2  
        #uq = dct(w_hat, type=3, axis=0)
        uq = idct(w_hat, type=2, axis=0)
    
    else:
        # Alternatively using Vandermonde matrix
        uq = np.dot(V.T, uk_hat)
        
    return uq

uq = backtransform(uk_hat, fast_transform)

print "Error in transforms = ", np.linalg.norm(ifastShenTrans(fastShenTrans(uq))-uq)

plt.figure(); plt.plot(points, [u.subs(x, i) for i in points]); plt.title("U")    
plt.figure(); plt.plot(points, uq - np.array([u.subs(x, h) for h in points])); plt.title("Error")
plt.show()
