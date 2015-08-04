from numpy.polynomial import chebyshev as n_cheb
from sympy import chebyshevt, Symbol, sin, cos, pi, lambdify, sqrt as Sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.sparse import diags
import scipy.sparse.linalg as la
from scipy.fftpack import dct, idct
from shentransform import ShenNeumannBasis 

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

N = 40
banded = False
ST = ShenNeumannBasis()
points, weights = ST.points_and_weights(N)
k = ST.wavenumbers(N)

# Gauss-Chebyshev quadrature to compute rhs (zero mean)
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points
fj -= np.dot(fj, weights)/weights.sum()

def solve_neumann(f):    

    if banded:
        A = np.zeros((N-2, N-2))
        A[-1, :] = -2*np.pi*(k+1)*k**2/(k+2)
        for i in range(2, N-3, 2):
            A[-i-1, i:] = -4*np.pi*(k[:-i]+i)**2*(k[:-i]+1)/(k[:-i]+2)**2
            
        uk_hat = solve_banded((0, N-4), A[1:, 1:], f)

    else:
        aij = [-2*np.pi*(k[1:]+1)*k[1:]**2/(k[1:]+2)]
        for i in range(2, N-2, 2):
            aij.append(-4*np.pi*(k[1:-i]+i)**2*(k[1:-i]+1)/(k[1:-i]+2)**2)    
        A = diags(aij, range(0, N-3, 2))
        uk_hat = la.spsolve(A, f)
        
    uk_hat = zeros(N-3)
    diag = -2*np.pi*(k+1)*k**2/(k+2) 
    uk_hat[-2:] = f[-2:]/diag[-2:]
    while 
        
    return uk_hat, A

f_hat = ST.fastShenScalar(fj)
uk_hat, A = solve_neumann(f_hat)
uj = ST.ifst(uk_hat)

print "Error in transforms = ", np.linalg.norm(ST.ifst(ST.fst(fj))-fj)

u_exact = np.array([u.subs(x, i) for i in points])
u_exact -= np.dot(u_exact, weights)/weights.sum()
plt.figure(); plt.plot(points, u_exact); plt.title("U")    
plt.figure(); plt.plot(points, uj - u_exact); plt.title("Error")
plt.show()
