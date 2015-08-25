import numpy as np
cimport cython
cimport numpy as np

ctypedef fused T:
    np.float64_t
    np.complex128_t

def chebDerivativeCoefficients(np.ndarray[T, ndim=1] fk, np.ndarray[T, ndim=1] fl):
    cdef:
        unsigned int N = fk.shape[0]-1
        int k
    fl[-1] = 0
    fl[-2] = 2*N*fk[-1]
    for k in range(N-2, 0, -1):
        fl[k] = 2*(k+1)*fk[k+1]+fl[k+2]
    fl[0] = fk[1] + 0.5*fl[2]

def chebDerivativeCoefficients_3D(np.ndarray[T, ndim=3] fk, np.ndarray[T, ndim=3] fl): 
    cdef unsigned int i, j
    
    for i in xrange(fk.shape[1]):
        for j in xrange(fk.shape[2]):
            chebDerivativeCoefficients(fk[:, i, j], fl[:, i, j])
    return fl    
    
