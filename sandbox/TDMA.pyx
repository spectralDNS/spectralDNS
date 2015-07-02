import numpy as np
cimport cython
cimport numpy as np

def TDMA_1D(np.ndarray[np.float64_t, ndim=1] a, 
            np.ndarray[np.float64_t, ndim=1] b, 
            np.ndarray[np.float64_t, ndim=1] c, 
            np.ndarray[np.float64_t, ndim=1] d):
    cdef:
        unsigned int n = b.shape[0]
        unsigned int m = a.shape[0]
        unsigned int k = n - m
        int i
        
    for i in range(m):
        d[i + k] -= d[i] * a[i] / b[i]
        b[i + k] -= c[i] * a[i] / b[i]
    for i in range(m - 1, -1, -1):
        d[i] -= d[i + k] * c[i] / b[i + k]
    for i in range(n):
        d[i] /= b[i]
        
    return d

def TDMA_3D(np.ndarray[np.float64_t, ndim=1] a, 
            np.ndarray[np.float64_t, ndim=1] b, 
            np.ndarray[np.float64_t, ndim=1] bc, 
            np.ndarray[np.float64_t, ndim=1] c, 
            np.ndarray[np.float64_t, ndim=3] d):
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
                d[i, ii, jj] /= bc[i]
        
    return d

def TDMA_3D_complex(np.ndarray[np.float64_t, ndim=1] a, 
            np.ndarray[np.float64_t, ndim=1] b, 
            np.ndarray[np.float64_t, ndim=1] bc, 
            np.ndarray[np.float64_t, ndim=1] c, 
            np.ndarray[np.complex128_t, ndim=3] d):
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
                d[i + k, ii, jj].real = d[i + k, ii, jj].real - (d[i, ii, jj].real * a[i] / bc[i])
                d[i + k, ii, jj].imag = d[i + k, ii, jj].imag - (d[i, ii, jj].imag * a[i] / bc[i])
                bc[i + k] -= c[i] * a[i] / bc[i]
            for i in range(m - 1, -1, -1):
                d[i, ii, jj].real = d[i, ii, jj].real - (d[i + k, ii, jj].real * c[i] / bc[i + k])
                d[i, ii, jj].imag = d[i, ii, jj].imag - (d[i + k, ii, jj].imag * c[i] / bc[i + k])
            for i in range(n):
                d[i, ii, jj].real = d[i, ii, jj].real/bc[i]
                d[i, ii, jj].imag = d[i, ii, jj].imag/bc[i]
        
    return d
    