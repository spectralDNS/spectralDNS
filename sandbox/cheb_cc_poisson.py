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
# Get points and weights for Chebyshev weighted integrals
points = n_cheb.chebpts2(N)[::-1]
weights = np.zeros((N))+np.pi/(N-1)
weights[0] /= 2
weights[-1] /= 2

# Note! N points in real space gives N-2 bases in spectral space

# Build Vandermonde matrix for Lobatto points and weights
V = n_cheb.chebvander(points, N-3).T - n_cheb.chebvander(points, N-1)[:, 2:].T

# Gauss-Chebyshev quadrature to compute rhs
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

def fastChebTrans(fj):
    cj = dct(fj, 1)
    cj /= (len(fj)-1)
    cj[0] /= 2
    cj[-1] /= 2
    return cj

#@profile
def fastShenTrans(fj):
    """Fast Shen transform on cos(j*pi/N).
    """
    cj = forwardtransform(fj, fast_transform)
    ck = np.ones(N-2); ck[0] = 2; ck[-1] = 2  # Note!! Shen paper has only ck[0] = 2, not ck[-1] = 2. For Gauss points ck[-1] = 1, but not here! 
    #B = np.zeros((5, N))
    #B[0, 2:] = -np.pi/2
    #B[2, :]  = np.pi/2*(ck+1)
    #B[4, :-2] = -np.pi/2
    #f_hat = solve_banded((2, 2), B, cj)
    
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

def fastShenScalar(fj):
    """Fast Shen scalar product on cos(j*pi/N).
       Chebyshev transform taking into account that phi_k = T_k - T_{k+2}
       Note, this is the non-normalized scalar product

    """
    cj = dct(fj, 1)
    cj *= (np.pi/((len(fj)-1)*2))
    cj[:-2] -= cj[2:]
    return cj[:-2]

def ifastShenScalar(fk):
    """Fast inverse Shen scalar product
    Transform needs to take into account that phi_k = T_k - T_{k+2}
    
    """
    w_hat = np.zeros(len(fk)+2)
    w_hat[:-2] = fk - np.hstack([0, 0, fk[:-2]])    
    w_hat[-2] = -fk[-2]
    w_hat[-1] = -fk[-1]
    fj = 0.5*dct(w_hat, 1)
    fj += 0.5*w_hat[0]
    fj[::2] += 0.5*w_hat[-1]
    fj[1::2] -= 0.5*w_hat[-1]

    return fj    

def forwardtransform(u, fast=True):
    if fast:
        
        u_k = fastShenScalar(u)
    else:
        # Alternatively using Vandermonde matrix
        u_k = np.dot(V, u*weights) 
        
    return u_k

fj = forwardtransform(fj, fast_transform)

#@profile
def solve(banded=True):
    
    if banded:
        A = np.zeros((N-2, N-2))
        A[-1, :] = -2*np.pi*(k+1)*(k+2)
        for i in range(2, N-2, 2):
            A[-i-1, i:] = -4*np.pi*(k[:-i]+1)
        uk_hat = solve_banded((0, N-3), A, fj)
        
    else:
        aij = [-2*np.pi*(k+1)*(k+2)]
        for i in range(2, N-2, 2):
            aij.append(np.array(-4*np.pi*(k[:-i]+1)))    
        A = diags(aij, range(0, N-2, 2))
        uk_hat = la.spsolve(A, fj)

    return uk_hat, A

uk_hat, A = solve(banded)

# Back transform
#@profile
def backtransform(u_hat, fast=True):
    if fast:
        uq = ifastShenScalar(u_hat)
    
    else:
        # Alternatively using Vandermonde matrix
        uq = np.dot(V.T, u_hat)
        
    return uq

uq = backtransform(uk_hat, fast_transform)

print "Error in transforms = ", np.linalg.norm(ifastShenTrans(fastShenTrans(uq))-uq)

plt.figure(); plt.plot(points, [u.subs(x, i) for i in points]); plt.title("U")    
plt.figure(); plt.plot(points, uq - np.array([u.subs(x, h) for h in points])); plt.title("Error")
plt.show()
