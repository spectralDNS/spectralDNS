from numpy.polynomial import chebyshev as n_cheb
from sympy import chebyshevt, Symbol, sin, cos, pi, exp, lambdify, sqrt as Sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded, lu_factor, lu_solve
from scipy.sparse import diags
import scipy.sparse.linalg as la
from spectralDNS.shen.shentransform import ShenBiharmonicBasis
from spectralDNS.shen.Matrices import ABBmat, BBBmat, SBBmat
from spectralDNS.shen.la import Biharmonic
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
u = (1-x**2)*sin(8*pi*x)*cos(4*pi*x)

N = 256
k = 2*N
nu = 1./590.
dt = 5e-5
a = -(k**2+nu*dt/2*k**4)*0
#b = (1.+nu*dt*k**2)
b = 1
c = -nu*dt/2.*0
f = a*u.diff(x, 4) + b*u.diff(x, 2) + c*u

SD = ShenBiharmonicFunctionSpace("GC", True)
points, weights = SD.points_and_weights(N) 

uj = np.array([u.subs(x, j) for j in points], dtype=float)
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

#uj_hat = np.zeros(N)
#uj_hat = SD.fst(uj, uj_hat)
#uj = SD.ifst(uj_hat, uj)
#fj_hat = np.zeros(N)
#fj_hat = SD.fst(fj, fj_hat)
#fj = SD.ifst(fj_hat, fj)


solver = Biharmonic(N, a, b, c, quad=SD.quad, solver="cython")
solver2 = Biharmonic(N, a, b, c, quad=SD.quad)

f_hat = np.zeros(N)
f_hat = SD.fastShenScalar(fj, f_hat)

u_hat = np.zeros(N)
u_hat2 = np.zeros(N)

from time import time
t0 = time()
u_hat = solver(u_hat, f_hat)
t1 = time()
u_hat2 = solver2(u_hat2, f_hat)
t2 = time()
print "cython / scipy ", t1-t0, t2-t1

u1 = np.zeros(N)
u1 = SD.ifst(u_hat, u1)

fr = np.random.randn(N)
fr_hat = np.zeros(N)
fr_hat = SD.fastShenScalar(fr, fr_hat)

ur_hat = np.zeros(N)
ur_hat2 = np.zeros(N)
ur_hat2 = solver2(ur_hat2, fr_hat)
ur_hat = solver(ur_hat, fr_hat)

c0 = np.zeros(N)
c0 = solver.matvec(ur_hat, c0)
print np.sqrt(sum((c0-fr_hat)**2)/N), max(abs(c0-fr_hat))/max(abs(fr_hat))
c1 = np.zeros(N)
c1 = solver2.matvec(ur_hat2, c1)
print np.sqrt(sum((c1-fr_hat)**2)/N), max(abs(c1-fr_hat))/max(abs(fr_hat))


#fr = SD.ifst(fr_hat, fr)
#fr_hat = SD.fst(fr, fr_hat)
#fr2 = SD.ifst(fr_hat, fr2)

#assert np.allclose(fr2, fr)

print np.sqrt(sum((u1-uj)**2)/N), max(abs(u1-uj))/max(abs(uj))

cc = np.zeros(N)
cc = solver.matvec(u_hat, cc)
print np.sqrt(sum((cc-f_hat)**2)/N), max(abs(cc-f_hat))/max(abs(f_hat))


assert np.allclose(u1, uj)


##alfa = np.ones((4,4))
##beta = np.ones((4,4))
##solver = Biharmonic(N, -1, alfa, beta, quad=SD.quad)
##f_hat = f_hat.repeat(16).reshape((N, 4, 4))+f_hat.repeat(16).reshape((N, 4, 4))*1j
##u_hat = u_hat.repeat(16).reshape((N, 4, 4))+u_hat.repeat(16).reshape((N, 4, 4))*1j
##u_hat = solver(u_hat, f_hat)
##u1 = np.zeros((N, 4, 4), dtype=complex)
##u1 = SD.ifst(u_hat, u1)
##uj = uj.repeat(16).reshape((N, 4, 4)) + 1j*uj.repeat(16).reshape((N, 4, 4))
##assert np.allclose(u1, uj)

#from spectralDNS.shen.SFTc import *
#sii, siu, siuu = solver.S.dd, solver.S.ud[0], solver.S.ud[1]
#ail, aii, aiu = solver.A.ld, solver.A.dd, solver.A.ud
#bill, bil, bii, biu, biuu = solver.B.lld, solver.B.ld, solver.B.dd, solver.B.ud, solver.B.uud
#M = sii[::2].shape[0]
#u0 = np.zeros((2, M), float)   # Diagonal entries of U
#u1 = np.zeros((2, M-1), float)     # Diagonal+1 entries of U
#u2 = np.zeros((2, M-2), float)     # Diagonal+2 entries of U
#l0 = np.zeros((2, M-1), float)     # Diagonal-1 entries of L
#l1 = np.zeros((2, M-2), float)     # Diagonal-2 entries of L

##d = 1e-6*sii + a*aii + b*bii 
#d = np.ones(N-4)

#LU_Biharmonic_1D(a, b, c, sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu, u0, u1, u2, l0, l1)

#uk = np.zeros(N)
##u0[0] = solver.Le[0].diagonal(0)
##u0[1] = solver.Lo[0].diagonal(0)
##u1[0] = solver.Le[0].diagonal(1)
##u1[1] = solver.Lo[0].diagonal(1)

#AA = a*solver.S.diags().toarray() + b*solver.A.diags().toarray() + c*solver.B.diags().toarray()

##de = np.eye(N-4)/d
##AA = np.dot(de, AA)
##fr_hat[:-4] = fr_hat[:-4] / d

#U = np.zeros((2, M, M), float)
#ll0 = np.zeros((2, M-1), float)   # Diagonal-1 entries of L
#ll1 = np.zeros((2, M-2), float)     # Diagonal-2 entries of L
#ukk = np.zeros(N)
#uk2 = np.zeros(N)

#LUC_Biharmonic_1D(AA, U, ll0, ll1)
#Solve_LUC_Biharmonic_1D(fr_hat, ukk, U, ll0, ll1, 0)

#Le = diags([ll1[0], ll0[0], np.ones(M)], [-2, -1, 0]).toarray()
#assert np.allclose(np.dot(Le, U[0]), AA[::2, ::2])
#Lo = diags([ll1[1], ll0[1], np.ones(M)], [-2, -1, 0]).toarray()
#assert np.allclose(np.dot(Lo, U[1]), AA[1::2, 1::2])

#ak = np.zeros((2, M), float)
#bk = np.zeros((2, M), float)

#Biharmonic_factor_pr(ak, bk, l0, l1)
#Solve_Biharmonic_1D(fr_hat, uk, u0, u1, u2, l0, l1, ak, bk, a)

##ff = fr_hat.copy()
##ff[:-4] *= d
##u_hat = solver(u_hat, ff)

##Ae = AA[::2, ::2]
##u2 =  (Ae.diagonal()[2:] - l0[0, 1:]*u1[0, 1:] - u0[0, 2:])/l1[0, :]
##print U[0].diagonal(2) -  u2

##U[0].diagonal(3) -  1./l1[0, :-1] *(Ae.diagonal(1)[2:] - l0[0, 1:-1]*U[0].diagonal(2)[1:] - U[0].diagonal(1)[2:])

##from scipy.linalg import svd

##def my_cond(A):
    ##sigma = svd(A, full_matrices=False, compute_uv=False)
    ##sigma.sort()
    ##return sigma[-1]/sigma[0]

##print "Cond U = ", np.linalg.cond(U[0]), np.linalg.cond(U[1])
##print "Cond U = ", my_cond(U[0]), my_cond(U[1])

#Uc = U.copy()
#fc = fr_hat.copy()
#t0 = time()
#Solve_LUC_Biharmonic_1D(fc, uk2, Uc, ll0, ll1, 1)
#t1 = time()
#print t1-t0

##print "Cond U = ", np.linalg.cond(Uc[0]), np.linalg.cond(Uc[1])
##print "Cond U = ", my_cond(Uc[0]), my_cond(Uc[1])




 
