cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False
import numpy as np

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
        
    bc[0] = b[0]
    bc[1] = b[1]
    for i in range(m):
        d[i + k] -= d[i] * a[i] / bc[i]
        bc[i + k] = b[i + k] - c[i] * a[i] / bc[i]
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
        int n = b.shape[0]
        int m = a.shape[0]
        int l = n - m
        int i, j, k
        
    for j in range(d.shape[1]):
        for k in range(d.shape[2]):
            bc[0] = b[0]
            bc[1] = b[1]
            for i in range(m):
                d[i + l, j, k] -= d[i, j, k] * a[i] / bc[i]
                bc[i + l] = b[i + l] - c[i] * a[i] / bc[i]
            for i in range(m - 1, -1, -1):
                d[i, j, k] -= d[i + l, j, k] * c[i] / bc[i + l]
            for i in range(n):
                d[i, j, k] = d[i, j, k] / bc[i]
