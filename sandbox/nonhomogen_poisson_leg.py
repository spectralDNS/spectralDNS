from numpy.polynomial import legendre as leg
from sympy import legendre, Symbol, sin, cos, lambdify
import numpy as np
import sys
from pylab import plot, spy, show, figure

"""
Solve Helmholtz equation on (-1, 1) with homogeneous bcs

    -\nabla^2 u + kx^2u = f, u(1) = a, u(-1) = b

where kx is some integer wavenumber.

Use Shen's Legendre basis

  \phi_{0} = 0.5*(L_0 - L_1)
  \phi_k = L_{k-1} - L_{k+1}, k = 1, ..., N-1
  \phi_{N} = 0.5*(L_0 + L_1)

where L_k is k'th Legendre polynomial. Solve using spectral Galerkin and the
l2 norm (u, v) = \int_{-1}^{1} u v dx

The equation to be solved for is

    (\nabla u, \nabla \phi_k)_w + kx^2(u, phi_k)_w = (f, \phi_k)_w

    Au + kx^2*Bu = f

"""

# Use sympy to compute a rhs, given an analytical solution

a = -0.5
b = 1.5

# Some exact solution
x = Symbol("x")
u = (1.0-x**2)**2*cos(np.pi*4*x)*(x-0.25)**3 + b*(1 + x)/2. + a*(1 - x)/2.
kx = np.sqrt(5)
f = -u.diff(x, 2) + kx**2*u
fl = lambdify(x, f, "numpy")
ul = lambdify(x, u, "numpy")

n = 32
domain = sys.argv[-1] if len(sys.argv) == 2 else "C1"

# Chebyshev-Gauss nodes and weights
points, w = leg.leggauss(n+1)

# Chebyshev Vandermonde matrix
V = leg.legvander(points, n)

scl=1.0
# First derivative matrix zero padded to (n+1)x(n+1)
D1 = np.zeros((n+1,n+1))
D1[:-1,:] = leg.legder(np.eye(n+1), 1, scl=scl)

Vx  = np.dot(V, D1)

# Matrix of trial functions
P = np.zeros((n+1,n+1))

P[:,0] = (V[:,0] - V[:,1])/2
P[:,1:-1] = V[:,:-2] - V[:,2:]
P[:,-1] = (V[:,0] + V[:,1])/2

# Matrix of first derivatives of trial functions
Px = np.zeros((n+1,n+1))
Px[:,0] = (Vx[:,0] - Vx[:,1])/2
Px[:,1:-1] = Vx[:,:-2] - Vx[:,2:]
Px[:,-1] = (Vx[:,0] + Vx[:,1])/2

############################ Solve first in one domain ########################

# Mass Matrix
M = np.dot(w*P.T, P)

# First derivative Matrix
C = np.dot(w*P.T, Px)

# Stiffness Matrix
K = np.dot(w*Px.T, Px)

# Gauss-Chebyshev quadrature to compute rhs
fj = fl(points)
uj = ul(points)

# Compute rhs
f_hat = np.dot(w*P.T, fj)

# Assemble Helmholtz
A = K + kx**2*M

# Simply indent zeros to set boundary conditions
A[0,:] = 0
A[0, 0] = 1
A[-1, :] = 0
A[-1, -1] = 1
f_hat[0] = a
f_hat[-1] = b

u_hat = np.linalg.solve(A, f_hat)

uq = np.dot(P, u_hat)

assert np.allclose(uq, uj)

# Alternatively, solve only for j=1,2,...N-1
f_hat = np.dot(w*P.T, fj)
A = K + kx**2*M
f_hat[1] -= kx**2*(M[0,1]*a + M[-1,1]*b)
f_hat[2] -= kx**2*(M[0,2]*a + M[-1,2]*b)

u_hat2 = np.zeros_like(u_hat)
u_hat2[1:-1] = np.linalg.solve(A[1:-1, 1:-1], f_hat[1:-1]) # Note, f_hat[0] and f_hat[-1] never used
u_hat2[0] = a
u_hat2[-1] = b

uq2 = np.dot(P, u_hat2)

assert np.allclose(uq2, uj, 0, 1e-4)


####################### Now with domain [-1, 0] x [0, 1] ######################

# Assemble Helmholtz
scl = 2.0

xx = np.array([0.5*points-0.5+i for i in range(2)])
fj = fl(xx)
uj = ul(xx)

# Assemble Helmholtz
A = K*scl**2 + kx**2*M

A_t = np.zeros((2*(n+1)-1, 2*(n+1)-1))
A_t[0, 0] = 1
A_t[1:n, :n+1] = A[1:-1, :]
A_t[n+1:2*n, n:] = A[1:-1, :]
A_t[-1, -1] = 1

f_hat = np.zeros(2*(n+1)-1)
f_hat[:(n+1)] = np.dot(w*P.T, fj[0])
f_hat[n:] += np.dot(w*P.T, fj[1])

f_hat[0] = a
f_hat[-1] = b

A_t[n] = 0
A_t[n, :n+1] = A[-1, :]
A_t[n, n:] += A[0, :]

u_hat = np.linalg.solve(A_t, f_hat)

uq = np.zeros((2, n+1))
uq[0] = np.dot(P, u_hat[:(n+1)])
uq[1] = np.dot(P, u_hat[n:])

assert np.allclose(uq, uj)


########## Now do four domains [-1, -0.5] x [-0.5, 0] x [0, 0.5] x [0, 1] ######

# Assemble Helmholtz
scl = 4.0

fj = np.zeros((4, (n+1)))
uj = np.zeros((4, (n+1)))
xx = np.array([0.25*points+(2*i-3)*0.25 for i in range(4)])
fj = fl(xx)
uj = ul(xx)

# Assemble Helmholtz
A = K*scl**2 + kx**2*M

A_t = np.zeros((4*(n+1)-3, 4*(n+1)-3))
A_t[0, 0] = 1
A_t[1:n+1, :n+1] = A[1:]
A_t[n:2*n+1, n:2*n+1] += A[:]
A_t[2*n:3*n+1, 2*n:3*n+1] += A[:]
A_t[3*n:4*n, 3*n:4*n+1] += A[:-1]
A_t[-1, -1] = 1

f_hat = np.zeros(4*n+1)
f_hat[:(n+1)] = np.dot(w*P.T, fj[0])
f_hat[n:2*n+1] += np.dot(w*P.T, fj[1])
f_hat[2*n:3*n+1] += np.dot(w*P.T, fj[2])
f_hat[3*n:4*n+1] += np.dot(w*P.T, fj[3])

f_hat[0] = a
f_hat[-1] = b

u_hat = np.linalg.solve(A_t, f_hat)

uq = np.array([np.dot(P, u_hat[n*i:(i+1)*n+1]) for i in range(4)])

assert np.allclose(uq, uj)

plot(xx.flatten(), uq.flatten())
figure()
spy(A_t, precision=1e-8)
show()
