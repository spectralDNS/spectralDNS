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
fast_transform = False

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

def fastChebScalar(fj):
    return dct(fj, 1)*np.pi/(2*(len(fj)-1))

def fastChebTrans(fj):
    cj = dct(fj, 1)/(len(fj)-1)
    cj[0] /= 2
    cj[-1] /= 2
    return cj

def ifastChebTrans(fj):
    cj = 0.5*dct(fj, 1)
    cj += 0.5*fj[0]
    cj[::2] += 0.5*fj[-1]
    cj[1::2] -= 0.5*fj[-1]
    return cj

#@profile
def fastShenTrans(fj):
    """Fast Shen transform on cos(j*pi/N).
    """
    cj = fastShenScalar(fj)
    ck = np.ones(N-2); ck[0] = 2; ck[-1] = 2  # Note!! Shen paper has only ck[0] = 2, not ck[-1] = 2. For Gauss points ck[-1] = 1, but not here! 
    #B = np.zeros((5, N))
    #B[0, 2:] = -np.pi/2
    #B[2, :]  = np.pi/2*(ck+1)
    #B[4, :-2] = -np.pi/2
    #f_hat = solve_banded((2, 2), B, cj)
    
    a = np.ones(N-4)*(-np.pi/2)
    b = np.pi/2*(ck+1)
    c = a.copy()
    cj = TDMA.TDMA_1D(a, b, c, cj)
    return cj

def ifastShenTrans(fj):
    """Inverse fast Shen transform on cos(j*pi/N).
    """
    return ifastShenScalar(fj)

def fastShenScalar(fj):
    """Fast Shen scalar product on cos(j*pi/N).
       Chebyshev transform taking into account that phi_k = T_k - T_{k+2}
       Note, this is the non-normalized scalar product

    """
    if fast_transform:
        cj = fastChebScalar(fj)
        cj[:-2] -= cj[2:]
        return cj[:-2]
    else:
        return np.dot(V, fj*weights)         

def chebDerivativeCoefficients(f_k):
    N = len(f_k)-1
    f_1 = f_k.copy()
    f_1[-1] = 0
    f_1[-2] = 2*N*f_k[-1]
    for k in range(N-2, 0, -1):
        f_1[k] = 2*(k+1)*f_k[k+1]+f_1[k+2]
    f_1[0] = f_k[1] + 0.5*f_1[2]
    return f_1

def fastChebDerivative(fj):
    f_k = fastChebTrans(fj)
    f_1 = chebDerivativeCoefficients(f_k)
    df = ifastChebTrans(f_1)
    return df
     
def ifastShenScalar(fk):
    """Fast inverse Shen scalar product
    Transform needs to take into account that phi_k = T_k - T_{k+2}
    
    """
    if fast_transform:
        w_hat = np.zeros(len(fk)+2)
        w_hat[:-2] = fk - np.hstack([0, 0, fk[:-2]])    
        w_hat[-2] = -fk[-2]
        w_hat[-1] = -fk[-1]
        fj = ifastChebTrans(w_hat)
        return fj    
    
    else:
        return np.dot(V.T, fk)

#@profile
def solve(fj, banded=True):
    
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

fj = fastShenScalar(fj)
uk_hat, A = solve(fj, banded)
uq = ifastShenScalar(uk_hat)

print "Error in transforms = ", np.linalg.norm(ifastShenTrans(fastShenTrans(uq))-uq)

plt.figure(); plt.plot(points, [u.subs(x, i) for i in points]); plt.title("U")    
plt.figure(); plt.plot(points, uq - np.array([u.subs(x, h) for h in points])); plt.title("Error")
plt.show()
