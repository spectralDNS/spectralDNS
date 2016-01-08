import numpy as np
cimport cython
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

def CDNmat_matvec(np.ndarray[real_t, ndim=1] ud, np.ndarray[real_t, ndim=1] ld,
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

def BDNmat_matvec(real_t ud, 
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

def CDDmat_matvec(np.ndarray[real_t, ndim=1] ud, np.ndarray[real_t, ndim=1] ld,
                 np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):
    cdef:
        int i, j, k
        int N = v.shape[0]-2
    for j in xrange(b.shape[1]):
        for k in xrange(b.shape[2]):
            b[0, j, k] = ud[0]*v[1, j, k]
            b[N-1, j, k] = ld[N-2]*v[N-2, j, k]
            
    for i in xrange(1, N-1):
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[i, j, k] = ud[i]*v[i+1, j, k] + ld[i-1]*v[i-1, j, k]

def SBBmat_matvec(np.ndarray[real_t, ndim=1] v, np.ndarray[real_t, ndim=1] b):
    cdef:
        int i, j, k
        int N = v.shape[0]-4
        long double d, p, r, s1, s2
        
    j = N-1
    s1 = 0.0
    s2 = 0.0
    b[j] = (8*(j+1)*(j+1)*(j+2)*(j+4))*np.pi*v[j]
    for k in range(N-3, -1, -2):
        j = k+2
        d = (8*(k+1)*(k+1)*(k+2)*(k+4))*np.pi
        p = k*d/(k+1)
        r = 24*(k+1)*(k+2)*np.pi
        s1 += 1./(j+3.)*v[j]
        s2 += (j+2.)*(j+2.)/(j+3.)*v[j]
        b[k] = d*v[k] + p*s1 + r*s2

    j = N-2
    s1 = 0.0
    s2 = 0.0
    b[j] = (8*(j+1)*(j+1)*(j+2)*(j+4))*np.pi*v[j]
    for k in range(N-4, -1, -2):
        j = k+2
        d = (8*(k+1)*(k+1)*(k+2)*(k+4))*np.pi
        p = k*d/(k+1)
        r = 24*(k+1)*(k+2)*np.pi
        s1 += v[j]/(j+3.)
        s2 += (j+2.)*(j+2.)/(j+3.)*v[j]
        b[k] = d*v[k] + p*s1 + r*s2

def SBBmat_matvec3D(np.ndarray[complex_t, ndim=3] v, np.ndarray[complex_t, ndim=3] b):
    cdef:
        int i, j

    for i in range(v.shape[1]):
        for j in range(v.shape[2]):
            SBBmat_matvec(v[:, i, j].real, b[:, i, j].real)
            SBBmat_matvec(v[:, i, j].imag, b[:, i, j].imag)
            