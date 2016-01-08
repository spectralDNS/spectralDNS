from numpy.polynomial import chebyshev as n_cheb
from sympy import chebyshevt, Symbol, sin, cos, pi, lambdify, sqrt as Sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded, lu_factor, lu_solve
from scipy.sparse import diags
import scipy.sparse.linalg as la
from cbcdns.shen.shentransform import ShenBiharmonicBasis
from cbcdns.shen.Matrices import ABBmat, BBBmat, SBBmat
from cbcdns.shen.Helmholtz import Biharmonic
from scipy.linalg import solve
"""
Solve biharmonic equation on (-1, 1)

    \nabla^4 u - a \nabla^2 u + b u = f, u(\pm 1) = u'(\pm 1) = 0

where a and b are some integer wavenumbers.

The equation to be solved for is

    (\nabla^4 u, \phi_k)_w - a(\nabla^2 u, \phi_k)_w + b(u, \phi_k)_w = (f, \phi_k)_w 
    
    (A - aC + bB) u = f
"""

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
u = sin(2*pi*x)**2
a = 1.0
b = 1.0
f = -u.diff(x, 4) + a*u.diff(x, 2) + b*u

SD = ShenBiharmonicBasis("GC", True)
N = 128
points, weights = SD.points_and_weights(N) 

uj = np.array([u.subs(x, j) for j in points], dtype=float)
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

solver = Biharmonic(N, a, b, quad=SD.quad)

f_hat = np.zeros(N)
f_hat = SD.fastShenScalar(fj, f_hat)
u_hat = np.zeros(N)
u_hat = solver(u_hat, f_hat)

u1 = np.zeros(N)
u1 = SD.ifst(u_hat, u1)

fr = np.random.randn(N)
fr[-4:] = 0
fr_hat = np.zeros(N)
fr2 = np.zeros(N)

fr_hat = SD.fst(fr, fr_hat)
fr = SD.ifst(fr_hat, fr)
fr_hat = SD.fst(fr, fr_hat)
fr2 = SD.ifst(fr_hat, fr2)

assert np.allclose(fr2, fr)

assert np.allclose(u1, uj)

alfa = np.ones((4,4))
beta = np.ones((4,4))
solver = Biharmonic(N, alfa, beta, quad=SD.quad)
f_hat = f_hat.repeat(16).reshape((N, 4, 4))+f_hat.repeat(16).reshape((N, 4, 4))*1j
u_hat = u_hat.repeat(16).reshape((N, 4, 4))+u_hat.repeat(16).reshape((N, 4, 4))*1j
u_hat = solver(u_hat, f_hat)
u1 = np.zeros((N, 4, 4))
u1 = SD.ifst(u_hat, u1)
uj = uj.repeat(16).reshape((N, 4, 4))
assert np.allclose(u1, uj)


