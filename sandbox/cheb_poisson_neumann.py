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
Solve Poisson equation on (-1, 1) with homogeneous Neumann bcs

    \nabla^2 u = f, u'(\pm 1) = 0
    
Use basis \phi_k = T_k - (k/(k+2))**2 * T_{k+2}, where T_k is k'th Chebyshev 
polynomial of first kind. Solve using spectral Galerkin and the
weighted L_w norm (u, v)_w = \int_{-1}^{1} u v / \sqrt(1-x^2) dx

    (\nabla^ u, \phi_k)_w = (f, \phi_k)_w 
    

"""

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
u = cos(np.pi*x)
f = u.diff(x, 2) 

N = 50
k = np.arange(N-2).astype(np.float)
# Get points and weights for Chebyshev weighted integrals
points = n_cheb.chebpts2(N)[::-1]
weights = np.zeros((N))+np.pi/(N-1)
weights[0] /= 2
weights[-1] /= 2

# Build Vandermonde matrix for Lobatto points and weights
V = n_cheb.chebvander(points, N-3).T - ((k/(k+2))**2)[:, np.newaxis]*n_cheb.chebvander(points, N-1)[:, 2:].T
V = V[1:, :] # Not using k=0

def fastShenTransNeumann(fj):
    """Fast Shen transform on cos(j*pi/N).
    """
    cj = fastShenScalarNeumann(fj)
    ck = np.ones(N-3); ck[-1] = 2 # Note not the first since basis phi_0 is not included
    a = np.ones(N-5)*(-np.pi/2)*(k[1:-2]/(k[1:-2]+2))**2
    b = np.pi/2*(1+(k[1:]/(k[1:]+2))**4)
    c = a.copy()
    cj = TDMA.TDMA_1D(a, b, c, cj)
    return cj

def ifastShenTransNeumann(fj):
    """Inverse fast Shen transform on cos(j*pi/N).
    """
    cj = ifastShenScalarNeumann(fj)
    return cj

def fastShenScalarNeumann(fj):
    """Fast Shen scalar product on cos(j*pi/N).
       Chebyshev transform taking into account that phi_k = T_k - (k/(k+2))**2*T_{k+2}
       Note, this is the non-normalized scalar product

    """
    ck = dct(fj, 1)
    ck *= (np.pi/((N-1)*2))
    ck[:-2] -= (k[:]/(k[:]+2))**2 * ck[2:]
    return ck[1:-2]

def ifastShenScalarNeumann(fk):
    """Fast inverse Shen scalar product
    """
    w_hat = np.zeros(N)
    f_hat = np.zeros(N-2)
    f_hat[1:] = fk
    w_hat[:-2] = f_hat - np.hstack([0, 0, (k[:-2]/(k[:-2]+2))**2*f_hat[:-2]])    
    w_hat[-2] = -(k[-2]/(k[-2]+2))**2*f_hat[-2]
    w_hat[-1] = -(k[-1]/(k[-1]+2))**2*f_hat[-1]
    fj = 0.5*dct(w_hat, 1)
    fj[::2] += 0.5*w_hat[-1]
    fj[1::2] -= 0.5*w_hat[-1]

    return fj

# Gauss-Chebyshev quadrature to compute rhs
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points
fj -= np.dot(fj, weights)/weights.sum()

f_k = np.dot(V, fj*weights)

def solve_neumann(f):    

    A = np.zeros((N-2, N-2))
    A[-1, :] = -2*np.pi*(k+1)*k**2/(k+2)
    for i in range(2, N-3, 2):
        A[-i-1, i:] = -4*np.pi*(k[:-i]+i)**2*(k[:-i]+1)/(k[:-i]+2)**2
        
    uk_hat = solve_banded((0, N-4), A[1:, 1:], f)
        
    return uk_hat

f_hat = solve_neumann(f_k)

ff = ifastShenTransNeumann(f_hat)

print "Error in transforms = ", np.linalg.norm(ifastShenTransNeumann(fastShenTransNeumann(ff))-ff)

u_exact = np.array([u.subs(x, i) for i in points])
u_exact -= np.dot(u_exact, weights)/weights.sum()
plt.figure(); plt.plot(points, u_exact); plt.title("U")    
plt.figure(); plt.plot(points, ff - u_exact); plt.title("Error")
plt.show()
