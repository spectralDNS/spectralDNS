from numpy.polynomial import chebyshev as n_cheb
from sympy import chebyshevt, Symbol, sin, cos, pi, lambdify, sqrt as Sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.sparse import diags
import scipy.sparse.linalg as la
from spectralDNS.shen.shentransform import ShenDirichletBasis
from spectralDNS.shen import SFTc
from spectralDNS.shen.Matrices import Amat, BDmat

"""
Solve Helmholtz equation on (-1, 1) with homogeneous bcs

    -\nabla^2 u + kx^2u = f, u(\pm 1) = 0

where kx is some integer wavenumber.

Use Shen basis \phi_k = T_k - T_{k+2}, where T_k is k'th Chebyshev 
polynomial of first kind. Solve using spectral Galerkin and the
weighted L_w norm (u, v)_w = \int_{-1}^{1} u v / \sqrt(1-x^2) dx
The equation to be solved for is

    -(\nabla^2 u, \phi_k)_w + kx^2(u, phi_k)_w = (f, \phi_k)_w 
    
    Au + kx^2*Bu = f

"""

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
u = (1-x**2)**2*cos(np.pi*x)*(x-0.25)**2
kx = np.sqrt(5)
f = -u.diff(x, 2) + kx**2*u

# Choices
solver = "lu"
N = 16

ST = ShenDirichletBasis(quad="GL")
points, weights = ST.points_and_weights(N) 

# Gauss-Chebyshev quadrature to compute rhs
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

#@profile
def solve(fk):
    
    k = ST.wavenumbers(N)
        
    if solver == "sparse":
        A = Amat(np.arange(N).astype(np.float)).diags()
        B = BDmat(np.arange(N).astype(np.float), "GL").diags()        
        uk_hat = la.spsolve(A+kx**2*B, fk[:-2])        
        assert np.allclose(np.dot(A.toarray()+kx**2*B.toarray(), uk_hat), fk[:-2])

    elif solver == "sparse-even/odd":
        M = (N-3)/2
        m = np.arange(M+1).astype(float)
        aij_e = [2*np.pi*(2*m+1)*(2*m+2)]
        for i in range(1, M+1):
            aij_e.append(np.array(4*np.pi*(2*m[:-i]+1)))    
        A_e = diags(aij_e, range(M+1))
        bij_e = np.pi*np.ones(M+1); bij_e[0] *= 1.5
        if N % 2 == 1:
            bij_e[-1] *= 1.5
        bio_e = -np.pi/2*np.ones(M)                
        B_e = diags([bio_e, bij_e, bio_e], range(-1, 2, 1)) 
        
        Mo = (N-4)/2
        m = np.arange(Mo+1).astype(float)
        aij_o = [2*np.pi*(2*m+2)*(2*m+3)]
        for i in range(1, Mo+1):
            aij_o.append(np.array(4*np.pi*(2*m[:-i]+2)))    
        A_o = diags(aij_o, range(Mo+1))
        bij_o = np.pi*np.ones(Mo+1)
        if N % 2 == 0:
            bij_o[-1] *= 1.5
        bio_o = -np.pi/2*np.ones(Mo)                
        B_o = diags([bio_o, bij_o, bio_o], range(-1, 2, 1))         
        
        uk_hat = np.zeros(N-2)
        uk_hat[::2] = la.spsolve(A_e+kx**2*B_e, fk[::2])
        uk_hat[1::2] = la.spsolve(A_o+kx**2*B_o, fk[1::2])
        
    elif solver == "lu":        
        
        uk_hat = np.zeros(N-2)
        
        M = (N-3)/2
        Mo = (N-4)/2
        d0 = np.zeros((2, M+1))
        d1 = np.zeros((2, M))
        d2 = np.zeros((2, M-1))
        L  = np.zeros((2, M))
        SFTc.LU_Helmholtz_1D(N, 0, ST.quad=="GL", kx, d0, d1, d2, L)
        SFTc.Solve_Helmholtz_1D(N, 0, fk[:-2], uk_hat, d0, d1, d2, L)

        A = Amat(np.arange(N).astype(np.float)).diags()
        B = BDmat(np.arange(N).astype(np.float), "GL").diags()        
        uk_hat2 = la.spsolve(A+kx**2*B, fk[:-2])        

        print np.linalg.norm(uk_hat-uk_hat2)
        assert np.allclose(uk_hat, uk_hat2)
        
        b = np.zeros(N-2)
        SFTc.Mult_Helmholtz_1D(N, ST.quad=="GL", 1, kx**2, uk_hat, b)
        assert np.allclose(b, fk[:-2])

        uk = np.zeros((N-2, 10, 10))
        f_hat = np.zeros((N-2, 10, 10))
        for i in range(10):
            for j in range(10):
                f_hat[:, i, j] = fk[:-2]

        kx2 = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
        alfa = np.sqrt(kx2[0]*kx2[0]+kx2[1]*kx2[1])
        u0 = np.zeros((2, M+1, 10, 10))   # Diagonal entries of U
        u1 = np.zeros((2, M, 10, 10))     # Diagonal+1 entries of U
        u2 = np.zeros((2, M-1, 10, 10))   # Diagonal+2 entries of U
        L  = np.zeros((2, M, 10, 10))     # The single nonzero row of L                 
                
        SFTc.LU_Helmholtz_3D(N, 0, ST.quad=="GL", alfa, u0, u1, u2, L)
        SFTc.Solve_Helmholtz_3D(N, 0, f_hat, uk, u0, u1, u2, L)    
        
        print np.linalg.norm(uk[:, 1, 2]-uk_hat)
        assert np.allclose(uk[:, 1, 2], uk_hat)
        
        b = np.zeros((N-2, 10, 10))
        SFTc.Mult_Helmholtz_3D(N, ST.quad=="GL", 1, alfa**2, uk, b)
        
        assert np.allclose(b[:, 1, 1], fk[:-2])
        

        uk = np.zeros((N-2, 10, 10), dtype=np.complex)
        f_hat = np.zeros((N-2, 10, 10), dtype=np.complex)
        for i in range(10):
            for j in range(10):
                f_hat[:, i, j].real = fk[:-2]
                f_hat[:, i, j].imag = fk[:-2]
                
        SFTc.Solve_Helmholtz_3D_complex(N, 0, f_hat, uk, u0, u1, u2, L)     
        
        assert np.allclose(uk[:, 1, 1].real, uk_hat)
        assert np.allclose(uk[:, 1, 1].imag, uk_hat)
        
        b = np.zeros((N-2, 10, 10), dtype=np.complex)
        SFTc.Mult_Helmholtz_3D_complex(N, ST.quad=="GL", 1, alfa**2, uk, b)
        
        assert np.allclose(b[:, 1, 1].real, fk[:-2])
        assert np.allclose(b[:, 1, 1].imag, fk[:-2])

        
        
        
    return uk_hat

f_hat = fj.copy()
f_hat = ST.fastShenScalar(fj, f_hat)
uk_hat = fj.copy()
uk_hat[:-2] = solve(f_hat)
uq = uk_hat.copy()
uq = ST.ifst(uk_hat, uq)

uqf = uq.copy()
uqf = ST.fst(uq, uqf)
uq0 = uq.copy()
assert np.allclose(ST.ifst(uqf, uq0), uq)

u_exact = np.array([u.subs(x, h) for h in points], dtype=np.float)
plt.figure(); plt.plot(points, [u.subs(x, i) for i in points]); plt.title("U")    
plt.figure(); plt.plot(points, uq - u_exact); plt.title("Error")
print "Error = ", np.linalg.norm(uq - u_exact)
#plt.show()
    