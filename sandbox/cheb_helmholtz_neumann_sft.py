from numpy.polynomial import chebyshev as n_cheb
from sympy import chebyshevt, Symbol, sin, cos, pi, lambdify, sqrt as Sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.sparse import diags
import scipy.sparse.linalg as la
from shentransform import ShenNeumannBasis, ShenDirichletBasis
import SFTc

"""
Solve Helmholtz equation on (-1, 1) with homogeneous Neumann bcs

    -\nabla^2 u + kx^2u = f, u'(\pm 1) = 0

where kx is some integer wavenumber.

Use Shen basis \phi_k = T_k - (k/(k+2))**2*T_{k+2}, where T_k is k'th Chebyshev 
polynomial of first kind. Solve using spectral Galerkin and the
weighted L_w norm (u, v)_w = \int_{-1}^{1} u v / \sqrt(1-x^2) dx
The equation to be solved for is

    -(\nabla^2 u, \phi_k)_w + kx^2(u, phi_k)_w = (f, \phi_k)_w 
    
    Au + kx^2*Bu = f

Or in short 

    Cu = f                       (1)

where C = A + kx^2*B

The off diagonal rows of A (4pi*j^2*(k+1)/(k+2)^2) contain the 
column index j and as such the values are not constant. For this reason
we solve instead for J*u, where J is the diagonal matrix with j^2 
entries on the diagonal. We rewrite (1) such that

    Cu = _C*J*u = f

and solve for J*u. Now the off diagonal entries of _C are constant 
(4pi*(k+1)/(k+2)^2) and as such we can solve very efficiently with LU 
factorization. Solve by performing LU decomposition

    _C(Ju) = LU(Ju)
    
where L has one single nonzero diagonal and U is upper triangular
with max three different values on each row. Storage is only for
the three smallest diagonals. All other values are equal to U_(k, k+2)

"""

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
u = cos(np.pi*x)
kx = np.sqrt(0.0)
f = -u.diff(x, 2) + kx**2*u

# Choices
#solver = "sparse"
solver = "lu"
N = 20

ST = ShenNeumannBasis("GC")
SD = ShenDirichletBasis("GL")
points, weights = ST.points_and_weights(N) 

# Gauss-Chebyshev quadrature to compute rhs
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points
fj -= np.dot(fj, weights)/weights.sum()


#k = ST.wavenumbers(N)
#cij = [-np.pi*(k[1:]+1),
       #-np.pi*(2-(k[1:-1]/(k[1:-1]+2))**2*(k[1:-1]+3))]
#for i in range(3, N-3, 2):
    #cij.append(-2*np.pi*(1-((k[1:-i])/(k[1:-i]+2))**2))    
#C = diags(cij, range(0, N-2, 2), shape=(N-3, N-2))

#ck = np.ones(N-3)
#if ST.quad == "GC": ck[-1] = 2
#bij = np.pi/2*np.ones(N-3)*(1+ck*(k[1:]/(k[1:]+2))**2)
#bio = -np.pi/2*np.ones(N-5)*(k[1:-2]/(k[1:-2]+2))**2
#biu = -np.pi/2*np.ones(N-4)
#B = diags([biu, bij, bio], range(-1, 4, 2), shape=(N-3, N-2)) 

#b = np.zeros((N-3), dtype=complex)
#uk = np.random.randn((N-2))+np.random.randn((N-2))*1j
#vk = np.random.randn((N-2))+np.random.randn((N-2))*1j
#wk = np.random.randn((N-2))+np.random.randn((N-2))*1j

##u = (1-points**2)*np.sin(np.pi*points)
##uk = SD.fst(u) + 0.0*1j
##vk = np.zeros((N-2)) + np.zeros((N-2))*1j
##wk = np.zeros((N-2)) + np.zeros((N-2))*1j

#SFTc.Mult_Div_1D(N, 2, 3, uk, vk, wk, b)

##bb = b.real
##ck = np.ones(N-3)
##if ST.quad == "GC": ck[-1] = 2
##a0N = np.ones(N-5)*(-np.pi/2)*(k[1:-2]/(k[1:-2]+2))**2
##b0N = np.pi/2*(1+ck*(k[1:]/(k[1:]+2))**4)
##c0N = a0N.copy()
##bcN = b0N.copy()

##bb = SFTc.TDMA_1D(a0N, b0N, c0N, bb)
##cc = ST.ifst(bb)

#uu = np.dot(C.toarray(), uk) + 1j*2*np.dot(B.toarray(), vk) + 1j*3*np.dot(B.toarray(), wk) 

#assert np.allclose(b, uu)

#uuk = np.resize(uk, (N-2,N-2, N-2)).T
#vvk = np.resize(vk, (N-2,N-2, N-2)).T
#wwk = np.resize(wk, (N-2,N-2, N-2)).T
#bb = np.zeros((N-2, N-2, N-2), dtype=complex)
#twos = 2*np.ones((N-2, N-2))
#threes = 3*np.ones((N-2, N-2))
#SFTc.Mult_Div_3D(N, twos, threes, uuk, vvk, wwk, bb[1:])

#assert np.allclose(bb[1:, 0, 0], uu)

#cij = [-np.pi*((k[2:]-1)/(k[2:]+1))**2*(k[2:]+1),
       #np.pi*(k[:-1]+1)]

#C = diags(cij, [-2, 0], shape=(N-2, N-3))

#dpdx = np.dot(C.toarray(), pk)
#ck = np.ones(N-3)
#if ST.quad == "GC": ck[-1] = 2
#bm2 = -np.pi/2*np.ones(N-5)*((k[3:]-2)/k[3:])**2
#bii = np.pi/2*np.ones(N-3)*(1+ck*(k[1:]/(k[1:]+2))**2)
#bp2 = -np.pi/2*np.ones(N-4)
#B = diags([bm2, bii, bp2], [-3, -1, 1], shape=(N-2, N-3)) 



#@profile
def solve(fk):
    
    k = ST.wavenumbers(N)
        
    if solver == "sparse":
        # Set up a naive solver based on spsolve
        aij = [2*np.pi*(k[1:]+1)/(k[1:]+2)]
        for i in range(2, N-3, 2):
            aij.append(4*np.pi*(k[1:-i]+1)/(k[1:-i]+2)**2)    
        A = diags(aij, range(0, N-3, 2))

        ck = np.ones(N-3)
        if ST.quad == "GC": ck[-1] = 2
        bij = np.pi/2*np.ones(N-3)*(1+ck*(k[1:]/(k[1:]+2))**4)/k[1:]**2
        bio = -np.pi/2*np.ones(N-5)*(k[1:-2]/(k[1:-2]+2))**2/(k[1:-2]+2)**2
        biu = -np.pi/2*np.ones(N-5)*((k[3:]-2)/(k[3:]))**2/(k[3:]-2)**2
        B = diags([biu, bij, bio], range(-2, 3, 2)) 
        
        uk_hat = la.spsolve(A+kx**2*B, fk[1:-2])
        uk_hat /= k[1:]**2
        
    elif solver == "lu":        
        # Solve by splitting the system into odds and even coefficients,
        # because these are decoupled. Then use tailored LU factorization
        # and solve.
        
        uk_hat = np.zeros(N-3)
        
        M = (N-4)/2
        u0 = np.zeros((2, M+1))   # Diagonal entries of U
        u1 = np.zeros((2, M))     # Diagonal+1 entries of U
        u2 = np.zeros((2, M-1))   # Diagonal+2 entries of U
        L  = np.zeros((2, M))     # The single nonzero row of L 
        SFTc.LU_Helmholtz_1D(N, 1, ST.quad=="GC", kx, u0, u1, u2, L) # do LU factorization
        SFTc.Solve_Helmholtz_1D(N, 1, fk[1:-2], uk_hat, u0, u1, u2, L) # Solve
        
        kx2 = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
        alfa = np.sqrt(kx2[0]*kx2[0]+kx2[1]*kx2[1])
        u0 = np.zeros((2, M+1, 10, 10))   # Diagonal entries of U
        u1 = np.zeros((2, M, 10, 10))     # Diagonal+1 entries of U
        u2 = np.zeros((2, M-1, 10, 10))   # Diagonal+2 entries of U
        L  = np.zeros((2, M, 10, 10))     # The single nonzero row of L 

        #uk = np.zeros((N-3, 10, 10))
        #fk = np.zeros((N-3, 10, 10))
        #for i in range(10):
            #for j in range(10):
                #fk[:, i, j] = f_hat
                
        #SFTc.LU_Helmholtz_3D(N, 1, ST.quad=="GC", alfa, u0, u1, u2, L)
        #SFTc.Solve_Helmholtz_3D(N, 1, fk, uk, u0, u1, u2, L)        

        #uk = np.zeros((N-3, 10, 10), dtype=np.complex)
        #fk = np.zeros((N-3, 10, 10), dtype=np.complex)
        #for i in range(10):
            #for j in range(10):
                #fk[:, i, j].real = f_hat
                #fk[:, i, j].imag = f_hat
                
        #SFTc.Solve_Helmholtz_3D_complex(N, 1, fk, uk, u0, u1, u2, L)     
        
        #assert np.allclose(uk[:, 1, 1].real, uk_hat)
        #assert np.allclose(uk[:, 1, 1].imag, uk_hat)

            
    return uk_hat

f_hat = fj.copy()
f_hat = ST.fastShenScalar(fj, f_hat)
uk_hat = fj.copy()
uk_hat[1:-2] = solve(f_hat)
uj = uk_hat.copy()
uj = ST.ifst(uk_hat, uj)

uf = uj.copy()
uf = ST.fst(uj, uf)
u0 = uj.copy()
assert np.allclose(ST.ifst(uf, u0), uj)

u_exact = np.array([u.subs(x, i) for i in points], dtype=np.float)
u_exact -= np.dot(u_exact, weights)/weights.sum()
plt.figure(); plt.plot(points, u_exact); plt.title("U")    
plt.figure(); plt.plot(points, uj - u_exact); plt.title("Error")
print "Error = ", np.linalg.norm(uj - u_exact)
#plt.show()


