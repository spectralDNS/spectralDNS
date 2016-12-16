from numpy.polynomial import chebyshev as n_cheb
from sympy import chebyshevt, Symbol, sin, cos
import numpy as np
import sys

"""
Solve Helmholtz equation on (-1, 1) with homogeneous bcs

    -\nabla^2 u + kx^2u = f, u(1) = a, u(-1) = b

where kx is some integer wavenumber.

Use Shen basis

  \phi_k = T_k - T_{k+2}, k=0, 1, ..., N-2
  \phi_{N-1} = 0.5*(T_0 + T_1)
  \phi_{N} = 0.5*(T_0 - T_1)

where T_k is k'th Chebyshev
polynomial of first kind. Solve using spectral Galerkin and the
weighted L_w norm (u, v)_w = \int_{-1}^{1} u v / \sqrt(1-x^2) dx
The equation to be solved for is

    -(\nabla^2 u, \phi_k)_w + kx^2(u, phi_k)_w = (f, \phi_k)_w

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

n = 24
domain = sys.argv[-1] if len(sys.argv) == 2 else "C1"

# Chebyshev-Gauss nodes and weights
points, w = n_cheb.chebgauss(n+1)
points = points[::-1]

# Chebyshev Vandermonde matrix
V = n_cheb.chebvander(points, n)

scl=1.0
# First derivative matrix zero padded to (n+1)x(n+1)
D1 = np.zeros((n+1,n+1))
D1[:-1,:] = n_cheb.chebder(np.eye(n+1), 1, scl=scl)

# Second derivative matrix zero padded to (n+1)x(n+1)
D2 = np.zeros((n+1,n+1))
D2[:-2,:] = n_cheb.chebder(np.eye(n+1), 2, scl=scl)

Vx  = np.dot(V, D1)
Vxx = np.dot(V, D2)

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

# Matrix of second derivatives of trial functions
Pxx = np.zeros((n+1,n+1))
Pxx[:,0] = (Vxx[:,0] - Vxx[:,1])/2
Pxx[:,1:-1] = Vxx[:,:-2] - Vxx[:,2:]
Pxx[:,-1] = (Vxx[:,0] + Vxx[:,1])/2


# Mass Matrix
M = np.dot(w*P.T, P)

# First derivative Matrix
C = np.dot(w*P.T, Px)

# Stiffness Matrix
K = -np.dot(w*P.T, Pxx)

# Gauss-Chebyshev quadrature to compute rhs
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points
uj = np.array([u.subs(x, j) for j in points], dtype=float)

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

#assert np.allclose(uq, uj)

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


# Now do two domains [-1, 0] x [0, 1]
# Assemble Helmholtz
scl = 2.0 # Map domains to [-1, 1]

fj = np.zeros((2, (n+1)))
uj = np.zeros((2, (n+1)))
fj[0] = np.array([f.subs(x, 0.5*(j-1)) for j in points], dtype=float)
fj[1] = np.array([f.subs(x, 0.5*(j+1)) for j in points], dtype=float)
uj[0] = np.array([u.subs(x, 0.5*(j-1)) for j in points], dtype=float)
uj[1] = np.array([u.subs(x, 0.5*(j+1)) for j in points], dtype=float)

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

# Use continuity across
left = -np.arange(n+1)*4.0
left[0], left[-1] = -0.5, 0.5
right = -np.arange(n+1)*4.0
right[1::2] *= -1
right[0], right[-1] = -0.5, 0.5

A_t[n, :] = 0
if domain == "C1":
    A_t[n, :n+1] = left
    A_t[n, n:] -= right
elif domain == "Ceven":
    A_t[n, :n+1:2] = left[::2]
    A_t[n, n::2] -= right[::2]
else:
    A_t[n, 1:n+1:2] = left[1::2]
    A_t[n, n+1::2] -= right[1::2]

f_hat[n] = 0

u_hat = np.linalg.solve(A_t, f_hat)

uq = np.zeros((2, n+1))
uq[0] = np.dot(P, u_hat[:(n+1)])
uq[1] = np.dot(P, u_hat[n:])

x1 = np.hstack([0.5*points-0.5,0.5*points+0.5])
assert np.allclose(uq, uj)

# Now do four domains [-1, -0.5] x [-0.5, 0] x [0, 0.5] x [0, 1]
# Assemble Helmholtz
scl = 4.0

fj = np.zeros((4, (n+1)))
uj = np.zeros((4, (n+1)))
fj[0] = np.array([f.subs(x, 0.25*j-0.75) for j in points], dtype=float)
fj[1] = np.array([f.subs(x, 0.25*j-0.25) for j in points], dtype=float)
fj[2] = np.array([f.subs(x, 0.25*j+0.25) for j in points], dtype=float)
fj[3] = np.array([f.subs(x, 0.25*j+0.75) for j in points], dtype=float)
uj[0] = np.array([u.subs(x, 0.25*j-0.75) for j in points], dtype=float)
uj[1] = np.array([u.subs(x, 0.25*j-0.25) for j in points], dtype=float)
uj[2] = np.array([u.subs(x, 0.25*j+0.25) for j in points], dtype=float)
uj[3] = np.array([u.subs(x, 0.25*j+0.75) for j in points], dtype=float)


# Assemble Helmholtz
A = K*scl**2 + kx**2*M

A_t = np.zeros((4*(n+1)-3, 4*(n+1)-3))
A_t[0, 0] = 1

A_t[1:n, :n+1] = A[1:-1, :]
A_t[n+1:2*n, n:2*n+1] = A[1:-1, :]
A_t[2*n+1:3*n, 2*n:3*n+1] = A[1:-1, :]
A_t[3*n+1:4*n, 3*n:4*n+1] = A[1:-1, :]

A_t[-1, -1] = 1

f_hat = np.zeros(4*n+1)
f_hat[:(n+1)] = np.dot(w*P.T, fj[0])
f_hat[n:2*n+1] += np.dot(w*P.T, fj[1])
f_hat[2*n:3*n+1] += np.dot(w*P.T, fj[2])
f_hat[3*n:4*n+1] += np.dot(w*P.T, fj[3])

f_hat[0] = a
f_hat[-1] = b

# Use continuity across
left = -np.arange(n+1)*4.0
left[0], left[-1] = -0.5, 0.5
right = -np.arange(n+1)*4.0
right[1::2] *= -1
right[0], right[-1] = -0.5, 0.5

A_t[n, :] = 0
A_t[2*n, :] = 0
A_t[3*n, :] = 0

A_t[n, :n+1] = left
A_t[n, n:2*n+1] -= right
A_t[2*n, n:2*n+1] = left
A_t[2*n, 2*n:3*n+1] -= right
A_t[3*n, 2*n:3*n+1] = left
A_t[3*n, 3*n:] -= right

f_hat[n] = 0
f_hat[2*n] = 0
f_hat[3*n] = 0

u_hat = np.linalg.solve(A_t, f_hat)

uq = np.zeros((4, n+1))
uq[0] = np.dot(P, u_hat[:(n+1)])
uq[1] = np.dot(P, u_hat[n:2*n+1])
uq[2] = np.dot(P, u_hat[2*n:3*n+1])
uq[3] = np.dot(P, u_hat[3*n:4*n+1])

xx = np.hstack([0.25*points+(2*i-3)*0.25 for i in range(4)])

assert np.allclose(uq, uj)

from pylab import *
plot(xx, uq.flatten())

spy(A_t, precision=1e-8)
show()

#M = M
#K = K*scl**2
#A = K + kx**2*M

## Alternatively, solve only for j=1,2,...N-1
#f_hat = np.zeros((2, (n+1)))
#f_hat[0] = np.dot(w*P.T, fj[0])*da
#f_hat[1] = np.dot(w*P.T, fj[1])*da

#A = K + kx**2*M
#f_hat[0, 1] -= kx**2*M[0, 1]*a
#f_hat[0, 2] -= kx**2*M[0, 2]*a
#f_hat[1, 1] -= kx**2*M[-1,1]*b
#f_hat[1, 2] -= kx**2*M[-1,2]*b

#AA = np.zeros((2*(n-1)+1, 2*(n-1)+1))
#AA[:n, :n] = A[1:-1, 1:-1]
#AA[n, :n+1] = A[-1, :]
#AA[n, n:] += A[0, :]
#AA[n+1:, n+1:] = A[1:-1, 1:-1]

#AA[0, n] += kx**2*M[-1, 1]
#AA[1, n] += kx**2*M[-1, 2]
#AA[n+1, n] += kx**2*M[0, 1]
#AA[n+2, n] += kx**2*M[0, 2]

#u_hat = np.zeros(2*(n+1)-1)
#ff = np.zeros(2*(n+1))
#ff[1:(n+1)] = f_hat[0, 1:]
#ff[n:-1] = f_hat[1, :-1]
#u_hat[1:-1] = np.linalg.solve(AA, ff)
#u_hat[0] = a
#u_hat[-1] = b

#uq2 = np.dot(P, u_hat2)

#assert np.allclose(uq2, uj)

#scl=2.0
## First derivative matrix zero padded to (n+1)x(n+1)
#D1 = np.zeros((n+1,n+1))
#D1[:-1,:] = n_cheb.chebder(np.eye(n+1), 1, scl=scl)

## Second derivative matrix zero padded to (n+1)x(n+1)
#D2 = np.zeros((n+1,n+1))
#D2[:-2,:] = n_cheb.chebder(np.eye(n+1), 2, scl=scl)

#Vx  = np.dot(V, D1)
#Vxx = np.dot(V, D2)

## Matrix of trial functions
#P = np.zeros((n+1,n+1))

#P[:,0] = (V[:,0] - V[:,1])/2
#P[:,1:-1] = V[:,:-2] - V[:,2:]
#P[:,-1] = (V[:,0] + V[:,1])/2

## Matrix of first derivatives of trial functions
#Px = np.zeros((n+1,n+1))
#Px[:,0] = (Vx[:,0] - Vx[:,1])/2
#Px[:,1:-1] = Vx[:,:-2] - Vx[:,2:]
#Px[:,-1] = (Vx[:,0] + Vx[:,1])/2

## Matrix of second derivatives of trial functions
#Pxx = np.zeros((n+1,n+1))
#Pxx[:,0] = (Vxx[:,0] - Vxx[:,1])/2
#Pxx[:,1:-1] = Vxx[:,:-2] - Vxx[:,2:]
#Pxx[:,-1] = (Vxx[:,0] + Vxx[:,1])/2


#elif domain == "Ceven":
    #A_t[n, :n+1:2] = left[::2]
    #A_t[n, n:2*n+1:2] -= right[::2]
    #A_t[2*n, n:2*n+1:2] = left[::2]
    #A_t[2*n, 2*n:3*n+1:2] -= right[::2]
    #A_t[3*n, 2*n:3*n+1:2] = left[::2]
    #A_t[3*n, 3*n::2] -= right[::2]

#elif domain == "Codd":
    #A_t[n, 1:n+1:2] = left[1::2]
    #A_t[n, n+1:2*n+1:2] -= right[1::2]
    #A_t[2*n, n+1:2*n+1:2] = left[1::2]
    #A_t[2*n, 2*n+1:3*n+1:2] -= right[1::2]
    #A_t[3*n, 2*n+1:3*n+1:2] = left[1::2]
    #A_t[3*n, 3*n+1::2] -= right[1::2]

#elif domain == "Cevenodd":
    #A_t[n, :n+1:2] = left[::2]
    #A_t[n, n:2*n+1:2] -= right[::2]
    #A_t[2*n, n+1:2*n+1:2] = left[1::2]
    #A_t[2*n, 2*n+1:3*n+1:2] -= right[1::2]
    #A_t[3*n, 2*n:3*n+1:2] = left[::2]
    #A_t[3*n, 3*n::2] -= right[::2]

#elif domain == "Coddeven":
    #A_t[n, 1:n+1:2] = left[1::2]
    #A_t[n, n+1:2*n+1:2] -= right[1::2]
    #A_t[2*n, n:2*n+1:2] = left[::2]
    #A_t[2*n, 2*n:3*n+1:2] -= right[::2]
    #A_t[3*n, 2*n+1:3*n+1:2] = left[1::2]
    #A_t[3*n, 3*n+1::2] -= right[1::2]
