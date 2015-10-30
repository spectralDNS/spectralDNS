import numpy as np
cimport cython
cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False

{0}

ctypedef fused T:
    real_t
    complex_t

def Chmat_matvec(np.ndarray[real_t, ndim=1] ud, np.ndarray[real_t, ndim=1] ld,
                 np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):
    cdef:
        int i, j, k
        int N = v.shape[0]-2
    for j in xrange(b.shape[1]):
        for k in xrange(b.shape[2]):
            b[0, j, k] = ud[0]*v[1, j, k]
            b[1, j, k] = ud[1]*v[2, j, k]
            b[N-1, j, k] = ld[N-3]*v[N-2, j, k]
            
    for i in xrange(2, N-1):
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[i, j, k] = ud[i]*v[i+1, j, k] + ld[i-2]*v[i-1, j, k]

def Bhmat_matvec(real_t ud, 
                 np.ndarray[real_t, ndim=1] ld,
                 np.ndarray[real_t, ndim=1] dd,
                 np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):
    cdef:
        int i, j, k
        int N = v.shape[0]-2
    for j in xrange(b.shape[1]):
        for k in xrange(b.shape[2]):
            b[0, j, k] = ud*v[2, j, k]
            b[1, j, k] = ud*v[3, j, k] + dd[0]*v[1, j, k]
            b[2, j, k] = ud*v[4, j, k] + dd[1]*v[2, j, k]
            b[N-2, j, k] = ld[N-5]*v[N-4, j, k] + dd[N-3]*v[N-2, j, k]
            b[N-1, j, k] = ld[N-4]*v[N-3, j, k] + dd[N-2]*v[N-1, j, k]
            
    for i in xrange(2, N-1):
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[i, j, k] = ud*v[i+2, j, k] + dd[i-1]*v[i, j, k] + ld[i-3]*v[i-2, j, k]

def Cmat_matvec(np.ndarray[real_t, ndim=1] ud, np.ndarray[real_t, ndim=1] ld,
                 np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):
    cdef:
        int i, j, k
        int N = v.shape[0]-2
    for j in xrange(b.shape[1]):
        for k in xrange(b.shape[2]):
            b[0, j, k] = ud[0]*v[1, j, k]
            b[N-1, j, k] = ld[N-2]*v[N-2, j, k]
            
    for i in xrange(2, N-1):
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[i, j, k] = ud[i]*v[i+1, j, k] + ld[i-1]*v[i-1, j, k]

