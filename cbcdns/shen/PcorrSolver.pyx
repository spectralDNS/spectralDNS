"""
Created on Thu Jan 21 13:10:05 2016

@author: Diako Darian

"""
import numpy as np
cimport cython
cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False


ctypedef fused T:
    np.float64_t
    np.complex128_t


#==================================================00==
#      RHS of pressure correction equation
#
#          I^{(2)}D^{-1}B_D U_hat[0]
#
# B_D:     Transformation matrix from Chebyshev to Dirichlet space
# D^{-1}:  Pseudo-inverse of differentiation matrix
# I^{(2)}: Pseudo-identity matrix       
#==================================================00==
def MatVecMult1(np.ndarray[np.float64_t, ndim=1] a, 
                np.ndarray[np.float64_t, ndim=1] b, 
                np.ndarray[T, ndim=3] r, 
                np.ndarray[T, ndim=3] x):

    cdef:
        unsigned int ii, jj
        
    for ii in range(r.shape[1]):
        for jj in range(r.shape[2]):
            MatVecMult1_1D(a,b, r[:, ii, jj].real, x[:, ii, jj].real)
            MatVecMult1_1D(a,b, r[:, ii, jj].imag, x[:, ii, jj].imag)
    return x

def MatVecMult1_1D(np.ndarray[np.float64_t, ndim=1] a, 
                   np.ndarray[np.float64_t, ndim=1] b,
                   np.ndarray[np.float64_t, ndim=1] r,
                   np.ndarray[np.float64_t, ndim=1] x):
    cdef:
        unsigned int N = r.shape[0]
        int i

    x[0] = x[1] = 0.
    x[2] = r[1]*(b[1]-a[1]) + r[3]*a[1]
    for i in range(3,N):
        if i < N-1:
            x[i] = -b[i-1]*r[i-3] + r[i-1]*(b[i-1] - a[i-1]) + r[i+1]*a[i-1]
        else:
            x[i] = -b[i-1]*r[i-3] + r[i-1]*(b[i-1] - a[i-1])
    return x

#==================================================00==
#      RHS of pressure correction equation
#
#        D^{-2}B_D U_hat[1] or D^{-2}B_D U_hat[2]
#
# B_D:     Transformation matrix from Chebyshev to Dirichlet space
# D^{-2}:  Pseudo-inverse of differentiation matrix (second derivative)     
#==================================================00==
def MatVecMult2(np.ndarray[np.float64_t, ndim=1] a, 
                np.ndarray[np.float64_t, ndim=1] b, 
                np.ndarray[np.float64_t, ndim=1] c, 
                np.ndarray[T, ndim=3] r, 
                np.ndarray[T, ndim=3] x):

    cdef:
        unsigned int ii, jj
        
    for ii in range(r.shape[1]):
        for jj in range(r.shape[2]):
            MatVecMult2_1D(a,b,c, r[:, ii, jj].real, x[:, ii, jj].real)
            MatVecMult2_1D(a,b,c, r[:, ii, jj].imag, x[:, ii, jj].imag)
    return x

def MatVecMult2_1D(np.ndarray[np.float64_t, ndim=1] a, 
                   np.ndarray[np.float64_t, ndim=1] b,
                   np.ndarray[np.float64_t, ndim=1] c,     
                   np.ndarray[np.float64_t, ndim=1] r,
                   np.ndarray[np.float64_t, ndim=1] x):

    cdef:
        unsigned int N = r.shape[0]
        int i

    x[0] = x[1] = 0.
    x[2] = r[0]*(c[0]-a[0]) + r[2]*(a[0]-b[0])+ r[4]*b[0]
    x[3] = r[1]*(c[1]-a[1]) + r[3]*(a[1]-b[1])+ r[5]*b[1]
    for i in xrange(4,N):
        if i < (N-2):
            x[i] = -c[i-2]*r[i-4] + r[i-2]*(c[i-2] - a[i-2]) + r[i]*(a[i-2]-b[i-2]) + r[i+2]*b[i-2]
        else:
            x[i] = -c[i-2]*r[i-4] + r[i-2]*(c[i-2] - a[i-2]) + r[i]*(a[i-2]-b[i-2])
    return x

#==================================================00==
#      Solver for pressure correction equation
#
#                  A U_hat = b,
#
# where A = I^{(2)}B_N - (l^2+m^2)D^{-2}B_N
#
# B_N:     Transformation matrix from Chebyshev to Neumann space
# D^{-2}:  Pseudo-inverse of differentiation matrix (second derivative)
# I^{(2)}: Pseudo-identity matrix       
# l: wave-number in y-direction
# m: wave-number in z-direction 
#==================================================00==
def PressureSolver(np.ndarray[np.float64_t, ndim=2] beta, 
                   np.ndarray[np.float64_t, ndim=1] bk, 
                   np.ndarray[np.float64_t, ndim=1] a, 
                   np.ndarray[np.float64_t, ndim=1] b, 
                   np.ndarray[np.float64_t, ndim=1] c, 
                   np.ndarray[np.float64_t, ndim=1] d, 
                   np.ndarray[T, ndim=3] r, 
                   np.ndarray[T, ndim=3] x):

    cdef:
        unsigned int ii, jj
        
    for ii in range(r.shape[1]):
        for jj in range(r.shape[2]):
            PressureSolver_1D(beta[ii,jj],bk,a,b,c,d, r[:, ii, jj].real, x[:, ii, jj].real)
            PressureSolver_1D(beta[ii,jj],bk,a,b,c,d, r[:, ii, jj].imag, x[:, ii, jj].imag)
    return x

def PressureSolver_1D(np.float_t xi,
                      np.ndarray[np.float64_t, ndim=1] bk, 
                      np.ndarray[np.float64_t, ndim=1] a, 
                      np.ndarray[np.float64_t, ndim=1] b,
                      np.ndarray[np.float64_t, ndim=1] c,     
                      np.ndarray[np.float64_t, ndim=1] d, 
                      np.ndarray[np.float64_t, ndim=1] r,
                      np.ndarray[np.float64_t, ndim=1] x):

    cdef:
        unsigned int N = r.shape[0]
        int i

    d = -xi*d
    a = 1.-xi*a
    b = bk-xi*b
    c = -xi*c

    d[0] /= b[0]
    a[0] /= b[0] 
    r[0] /= b[0]
    d[1] /= b[1]
    a[1] /= b[1] 
    r[1] /= b[1]  
    for i in xrange(2,N): 
        if i < (N-4):
            d[i] = d[i]/(b[i] - c[i-2]*a[i-2])
        if i<(N-2):    
            a[i] = (a[i] - c[i-2]*d[i-2])/(b[i] - c[i-2]*a[i-2])
        r[i] = (r[i] - c[i-2]*r[i-2])/(b[i] - c[i-2]*a[i-2])

    x[-1] = r[-1]
    x[-2] = r[-2]
    for i in xrange(N-3,-1,-1):
        x[i] = r[i]- a[i]*x[i+2] 
        if i<(N-4):
            x[i] -= d[i]*x[i+4]

    return x