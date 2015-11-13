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

def TDMA_1D(np.ndarray[real_t, ndim=1] a, 
            np.ndarray[real_t, ndim=1] b, 
            np.ndarray[real_t, ndim=1] bc,
            np.ndarray[real_t, ndim=1] c, 
            np.ndarray[T, ndim=1] d):
    cdef:
        unsigned int n = b.shape[0]
        unsigned int m = a.shape[0]
        unsigned int k = n - m
        int i
        
    for i in range(n):
        bc[i] = b[i]
    for i in range(m):
        d[i + k] -= d[i] * a[i] / bc[i]
        bc[i + k] -= c[i] * a[i] / bc[i]
    for i in range(m - 1, -1, -1):
        d[i] -= d[i + k] * c[i] / bc[i + k]
    for i in range(n):
        d[i] /= bc[i]
        

def TDMA_3D(np.ndarray[real_t, ndim=1] a, 
            np.ndarray[real_t, ndim=1] b, 
            np.ndarray[real_t, ndim=1] bc, 
            np.ndarray[real_t, ndim=1] c, 
            np.ndarray[T, ndim=3] d):
    cdef:
        unsigned int n = b.shape[0]
        unsigned int m = a.shape[0]
        unsigned int k = n - m
        int i, ii, jj
        
    for ii in range(d.shape[1]):
        for jj in range(d.shape[2]):
            for i in range(n):
                bc[i] = b[i]
            for i in range(m):
                d[i + k, ii, jj] -= d[i, ii, jj] * a[i] / bc[i]
                bc[i + k] -= c[i] * a[i] / bc[i]
            for i in range(m - 1, -1, -1):
                d[i, ii, jj] -= d[i + k, ii, jj] * c[i] / bc[i + k]
            for i in range(n):
                d[i, ii, jj] = d[i, ii, jj] / bc[i]
