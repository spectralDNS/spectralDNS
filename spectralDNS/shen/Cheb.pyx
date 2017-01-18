cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t
ctypedef double real

ctypedef fused T:
    real_t
    complex_t

def cheb_derivative_coefficients(np.ndarray[T, ndim=1] fk, np.ndarray[T, ndim=1] ck):
    cdef:
        unsigned int N = fk.shape[0]-1
        int k
    ck[-1] = 0
    ck[-2] = 2*N*fk[-1]
    for k in range(N-2, 0, -1):
        ck[k] = 2*(k+1)*fk[k+1]+ck[k+2]
    ck[0] = fk[1] + 0.5*ck[2]
    return ck

def cheb_derivative_coefficients_3D(np.ndarray[T, ndim=3] fk, np.ndarray[T, ndim=3] ck):
    cdef unsigned int i, j

    for i in xrange(fk.shape[1]):
        for j in xrange(fk.shape[2]):
            cheb_derivative_coefficients(fk[:, i, j], ck[:, i, j])
    return ck
