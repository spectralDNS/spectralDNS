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

def Tridiagonal_matvec3(double[::1] v,
                       double[::1] b,
                       double[::1] ld,
                       double[::1] dd,
                       double[::1] ud):

    cdef:
        np.intp_t i
        np.intp_t N = dd.shape[0]

    for i in xrange(N-2):
        b[i] = ud[i]*v[i+2]
    for i in xrange(N):
        b[i] += dd[i]*v[i]
    for i in xrange(2, N):
        b[i] += ld[i-2]*v[i-2]
