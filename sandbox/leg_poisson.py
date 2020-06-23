from numpy.polynomial import legendre as leg
from sympy import legendre, Symbol, sin, cos, lambdify
import numpy as np
import sys
from pylab import plot, spy, show, figure
from spectralDNS.shen.shentransform import ShenLegendreDirichletBasis
from spectralDNS.shen.Matrices import ShenMatrix

"""
Solve Helmholtz equation on (-1, 1) with homogeneous bcs

    -\nabla^2 u + kx^2u = f, u(\pm 1) = 0

where kx is some integer wavenumber.

Use Shen's Legendre basis

  \phi_k = L_{k} - L_{k+2}, k = 0, ..., N-2

where L_k is k'th Legendre polynomial. Solve using spectral Galerkin and the
l2 norm (u, v) = \int_{-1}^{1} u v dx

The equation to be solved for is

    (\nabla u, \nabla \phi_k)_w + kx^2(u, phi_k)_w = (f, \phi_k)_w

    Au + kx^2*Bu = f

"""

# Use sympy to compute a rhs, given an analytical solution

a = 0
b = 0

# Some exact solution
x = Symbol("x")
u = (1.0-x**2)**2*cos(np.pi*4*x)*(x-0.25)**3 + b*(1 + x)/2. + a*(1 - x)/2.
kx = np.sqrt(5)
f = -u.diff(x, 2) + kx**2*u
fl = lambdify(x, f, "numpy")
ul = lambdify(x, u, "numpy")

n = 64

ST = ShenLegendreDirichletFunctionSpace()

# Legendre-Gauss nodes and weights
points, w = ST.points_and_weights(n, 'LG')

# Mass Matrix
M = ShenMatrix({}, n, (ST, 0), (ST, 0))

# Stiffness Matrix
K = ShenMatrix({}, n, (ST, 1), (ST, 1))

# Legendre-Gauss quadrature to compute rhs
fj = fl(points)
uj = ul(points)

# Compute rhs
f_hat = np.zeros(n)
f_hat = ST.scalar_product(fj, f_hat)

# Assemble Helmholtz
A = K + kx**2*M

AA = A.diags().toarray()
u_hat = np.zeros(n)
u_hat[:-2] = np.linalg.solve(AA, f_hat[:-2])

uq = np.zeros(n)
uq = ST.backward(u_hat, uq)

assert np.allclose(uq, uj)


