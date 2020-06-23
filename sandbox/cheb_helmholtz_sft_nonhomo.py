from numpy.polynomial import chebyshev as n_cheb
from sympy import chebyshevt, Symbol, sin, cos, pi, lambdify, sqrt as Sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.sparse import diags
import scipy.sparse.linalg as la
from spectralDNS.shen.shentransform import ShenDirichletBasis
from spectralDNS.shen import SFTc
from spectralDNS.shen.Matrices import ADDmat, BDDmat
from spectralDNS.shen.la import Helmholtz

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

x = Symbol("x")
u = (1-x**2)**2*cos(np.pi*x)*(x-0.25)**2 + a*(1 + x)/2. + b*(1 - x)/2.
kx = np.sqrt(5)
f = -u.diff(x, 2) + kx**2*u

# Choices
solver = "lu"
N = 64

quad = "GC"
ST = ShenDirichletFunctionSpace(quad=quad, bc=(a, b))
points, weights = ST.points_and_weights(N, quad)

# Gauss-Chebyshev quadrature to compute rhs
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points
uj = np.array([u.subs(x, j) for j in points], dtype=float)

#@profile
def solve(fk):

    k = ST.wavenumbers(N)

    if solver == "sparse":
        A = ADDmat(np.arange(N).astype(np.float)).diags()
        B = BDDmat(np.arange(N).astype(np.float), quad).diags()
        fk[0] -= kx**2*pi/2.*(a + b)
        fk[1] -= kx**2*pi/4.*(a - b)
        uk_hat = la.spsolve(A+kx**2*B, fk[:-2])
        assert np.allclose(np.dot(A.toarray()+kx**2*B.toarray(), uk_hat), fk[:-2])

    elif solver == "lu":

        uk_hat = np.zeros(N-2)
        sol = Helmholtz(N, kx, quad=quad)
        fk[0] -= kx**2*pi/2.*(a + b)
        fk[1] -= kx**2*pi/4.*(a - b)
        uk_hat = sol(uk_hat, fk[:-2])

    return uk_hat

f_hat = fj.copy()
f_hat = ST.scalar_product(fj, f_hat)

uk_hat = fj.copy()
uk_hat[:-2] = solve(f_hat)
uk_hat[-2:] = a, b

uq = uk_hat.copy()
uq = ST.ifst(uk_hat, uq)


uqf = uq.copy()
uqf = ST.fst(uq, uqf)
uq0 = uq.copy()
uq0 = ST.ifst(uqf, uq0)

assert np.allclose(uq, uj)
assert np.allclose(uq0, uq)

u_exact = np.array([u.subs(x, h) for h in points], dtype=np.float)
plt.figure(); plt.plot(points, [u.subs(x, i) for i in points]); plt.title("U")
plt.figure(); plt.plot(points, uq - u_exact); plt.title("Error")
print("Error = ", np.linalg.norm(uq - u_exact))
#plt.show()
