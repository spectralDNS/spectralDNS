from numpy.polynomial import chebyshev as n_cheb
from sympy import chebyshevt, Symbol, sin, cos, pi, lambdify, sqrt as Sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.sparse import diags
import scipy.sparse.linalg as la
from shentransform import ShenTransform

"""
Solve Poisson equation on (-1, 1) with homogeneous bcs

    \nabla^2 u = f, u(\pm 1) = 0
    
Use Shen basis \phi_k = T_k - T_{k+2}, where T_k is k'th Chebyshev 
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
N = 20

ST = ShenTransform(quad="GC")
points, weights = ST.points_and_weights(N) 

# Gauss-Chebyshev quadrature to compute rhs
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

#@profile
def solve(fk, banded=True):
    
    N = len(fk)+2
    k = ST.wavenumbers()
    if banded:
        A = np.zeros((N-2, N-2))
        A[-1, :] = -2*np.pi*(k+1)*(k+2)
        for i in range(2, N-2, 2):
            A[-i-1, i:] = -4*np.pi*(k[:-i]+1)
        uk_hat = solve_banded((0, N-3), A, fk)
        
    else:
        aij = [-2*np.pi*(k+1)*(k+2)]
        for i in range(2, N-2, 2):
            aij.append(np.array(-4*np.pi*(k[:-i]+1)))    
        A = diags(aij, range(0, N-2, 2))
        uk_hat = la.spsolve(A, fk)

    return uk_hat

f_hat = ST.fastShenScalar(fj)
uk_hat = solve(f_hat, banded)
uq = ST.ifastShenScalar(uk_hat)

plt.figure(); plt.plot(points, [u.subs(x, i) for i in points]); plt.title("U")    
plt.figure(); plt.plot(points, uq - np.array([u.subs(x, h) for h in points])); plt.title("Error")
plt.show()
