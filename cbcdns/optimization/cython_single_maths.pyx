#cython: boundscheck=False
#cython: wraparound=False
cimport numpy as np


ctypedef np.complex64_t complex_t
ctypedef np.float32_t real_t
ctypedef np.int64_t int_t


ctypedef fused T:
    np.float64_t
    np.float32_t
    np.int64_t
    np.int32_t    

def dealias_rhs_NS(np.ndarray[complex_t, ndim=4] du,
                   np.ndarray[np.uint8_t, ndim=3] dealias):
    cdef unsigned int i, j, k
    cdef np.uint8_t uu
    for i in xrange(dealias.shape[0]):
        for j in xrange(dealias.shape[1]):
            for k in xrange(dealias.shape[2]):
                uu = dealias[i, j, k]
                du[0, i, j, k].real *= uu
                du[0, i, j, k].imag *= uu
                du[1, i, j, k].real *= uu
                du[1, i, j, k].imag *= uu
                du[2, i, j, k].real *= uu
                du[2, i, j, k].imag *= uu
    return du

def dealias_rhs_VV(np.ndarray[complex_t, ndim=4] du,
                   np.ndarray[np.uint8_t, ndim=3] dealias):
    du = dealias_rhs_NS(du, dealias)
    return du
    
def dealias_rhs_MHD(np.ndarray[complex_t, ndim=4] du,
                    np.ndarray[np.uint8_t, ndim=3] dealias):
    cdef unsigned int i, j, k
    cdef np.uint8_t uu
    for i in xrange(dealias.shape[0]):
        for j in xrange(dealias.shape[1]):
            for k in xrange(dealias.shape[2]):
                uu = dealias[i, j, k]
                du[0, i, j, k].real *= uu
                du[0, i, j, k].imag *= uu
                du[1, i, j, k].real *= uu
                du[1, i, j, k].imag *= uu
                du[2, i, j, k].real *= uu
                du[2, i, j, k].imag *= uu
                du[3, i, j, k].real *= uu
                du[3, i, j, k].imag *= uu
                du[4, i, j, k].real *= uu
                du[4, i, j, k].imag *= uu
                du[5, i, j, k].real *= uu
                du[5, i, j, k].imag *= uu

    return du

def dealias_rhs_NS2D(np.ndarray[complex_t, ndim=3] du,
                     np.ndarray[np.uint8_t, ndim=2] dealias):
    cdef unsigned int i, j
    cdef np.uint8_t uu
    for i in xrange(dealias.shape[0]):
        for j in xrange(dealias.shape[1]):
            uu = dealias[i, j]
            du[0, i, j].real *= uu
            du[0, i, j].imag *= uu
            du[1, i, j].real *= uu
            du[1, i, j].imag *= uu
    return du

def dealias_rhs_Bq2D(np.ndarray[complex_t, ndim=3] du,
                     np.ndarray[np.uint8_t, ndim=2] dealias):
    cdef unsigned int i, j
    cdef np.uint8_t uu
    for i in xrange(dealias.shape[0]):
        for j in xrange(dealias.shape[1]):
            uu = dealias[i, j]
            du[0, i, j].real *= uu
            du[0, i, j].imag *= uu
            du[1, i, j].real *= uu
            du[1, i, j].imag *= uu
            du[2, i, j].real *= uu
            du[2, i, j].imag *= uu 
    return du
    
def cross1(np.ndarray[real_t, ndim=4] c,
           np.ndarray[real_t, ndim=4] a,
           np.ndarray[real_t, ndim=4] b):
    cdef unsigned int i, j, k
    cdef real_t a0, a1, a2, b0, b1, b2
    for i in xrange(a.shape[1]):
        for j in xrange(a.shape[2]):
            for k in xrange(a.shape[3]):
                a0 = a[0,i,j,k]
                a1 = a[1,i,j,k]
                a2 = a[2,i,j,k]
                b0 = b[0,i,j,k]
                b1 = b[1,i,j,k]
                b2 = b[2,i,j,k]
                c[0,i,j,k] = a1*b2 - a2*b1
                c[1,i,j,k] = a2*b0 - a0*b2
                c[2,i,j,k] = a0*b1 - a1*b0
    return c

def cross2(np.ndarray[complex_t, ndim=4] c,
           np.ndarray[T, ndim=4] a,
           np.ndarray[complex_t, ndim=4] b):
    cdef unsigned int i, j, k
    cdef T a0, a1, a2
    cdef complex_t b0, b1, b2
    for i in xrange(a.shape[1]):
        for j in xrange(a.shape[2]):
            for k in xrange(a.shape[3]):
                a0 = a[0,i,j,k]
                a1 = a[1,i,j,k]
                a2 = a[2,i,j,k]
                b0 = b[0,i,j,k]
                b1 = b[1,i,j,k]
                b2 = b[2,i,j,k]
                c[0,i,j,k].real = -(a1*b2.imag - a2*b1.imag)
                c[0,i,j,k].imag = a1*b2.real - a2*b1.real
                c[1,i,j,k].real = -(a2*b0.imag - a0*b2.imag)
                c[1,i,j,k].imag = a2*b0.real - a0*b2.real
                c[2,i,j,k].real = -(a0*b1.imag - a1*b0.imag)
                c[2,i,j,k].imag = a0*b1.real - a1*b0.real
    return c

def cross1_2D(np.ndarray[real_t, ndim=2] c,
              np.ndarray[real_t, ndim=3] a,
              np.ndarray[real_t, ndim=3] b):
    cdef unsigned int i, j
    cdef real_t a0, a1, b0, b1
    with nogil:
        for i in xrange(a.shape[1]):
            for j in xrange(a.shape[2]):
                a0 = a[0,i,j]
                a1 = a[1,i,j]
                b0 = b[0,i,j]
                b1 = b[1,i,j]
                c[i,j] = a0*b1 - a1*b0
    return c

def cross2_2D(np.ndarray[complex_t, ndim=2] c,
              np.ndarray[T, ndim=3] a,
              np.ndarray[complex_t, ndim=3] b):
    cdef unsigned int i, j
    cdef T a0, a1
    cdef complex_t b0, b1
    for i in xrange(a.shape[1]):
        for j in xrange(a.shape[2]):
            a0 = a[0,i,j]
            a1 = a[1,i,j]
            b0 = b[0,i,j]
            b1 = b[1,i,j]
            c[i,j].real = -(a0*b1.imag - a1*b0.imag)
            c[i,j].imag = a0*b1.real - a1*b0.real
    return c

    
    
    
    