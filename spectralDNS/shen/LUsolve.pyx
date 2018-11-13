#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from libcpp.algorithm cimport copy

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t
ctypedef double real

ctypedef fused T:
    real_t
    complex_t


def Mult_Div_3D(np.int_t N,
                np.ndarray[real_t, ndim=2] m,
                np.ndarray[real_t, ndim=2] n,
                np.ndarray[complex_t, ndim=3] u_hat,
                np.ndarray[complex_t, ndim=3] v_hat,
                np.ndarray[complex_t, ndim=3] w_hat,
                np.ndarray[complex_t, ndim=3] b):
    cdef unsigned int i, j

    for i in xrange(m.shape[0]):
        for j in xrange(m.shape[1]):
            Mult_Div_1D(N, m[i, j], n[i, j],
                        u_hat[:, i, j],
                        v_hat[:, i, j],
                        w_hat[:, i, j],
                        b[:, i, j])

def Mult_Div_1D(np.int_t N,
                real_t m,
                real_t n,
                np.ndarray[complex_t, ndim=1] u_hat,
                np.ndarray[complex_t, ndim=1] v_hat,
                np.ndarray[complex_t, ndim=1] w_hat,
                np.ndarray[complex_t, ndim=1] b):
    cdef:
        unsigned int M
        int i
        int c0 = 0
        real pi = np.pi
        vector[real] bii, bm2, bp2, cm1, cp1, cp3
        double complex sum_u0
        double complex sum_u1

    M = (N-3)

    sum_u0 = 0.0+0.0*1j
    sum_u1 = 0.0+0.0*1j
    for i in range(1, M+1):
        bii.push_back(pi/2.0*(1.0+((i*1.0)/(i+2.0))**2 ))

    #bii.push_back(pi/2.0*(1.0+2.0*((M*1.0)/(M+2.0))**2 ))

    bp2.resize(M)
    bm2.resize(M)
    cp1.resize(M)
    cp3.resize(M)
    cm1.resize(M)
    for i in xrange(M-2):
        bp2[i] = -pi/2.0*(((i+1)*1.0)/((i+1)+2.0))**2
    for i in xrange(1, M):
        bm2[i] = -pi/2.0
    for i in xrange(1, M+1):
        cm1[i-1] = -(i+1.0)*pi
    for i in xrange(M-1):
        cp1[i] = -pi*(2.0 - (((i+1)*1.0)/((i+1)+2.0))**2*((i+1)+3.0))
    for i in xrange(M-3):
        cp3[i] = -2.0*pi*(1.0 - (((i+1)*1.0)/((i+1)+2.0))**2)

    # k = M-1
    b[M-1] = ((m*bm2[M-1]*v_hat[M-2] + n*bm2[M-1]*w_hat[M-2]
              + m*bii[M-1]*v_hat[M] + n*bii[M-1]*w_hat[M])*1j
              + cm1[M-1]*u_hat[M-1])

    b[M-2] = ((m*bm2[M-2]*v_hat[M-3] + n*bm2[M-2]*w_hat[M-3]
              + m*bii[M-2]*v_hat[M-1] + n*bii[M-2]*w_hat[M-1])*1j
              + cm1[M-2]*u_hat[M-2]
              + cp1[M-2]*u_hat[M])

    b[M-3] = ((m*bm2[M-3]*v_hat[M-4] + n*bm2[M-3]*w_hat[M-4]
              + m*bii[M-3]*v_hat[M-2] + n*bii[M-3]*w_hat[M-2]
              + m*bp2[M-3]*v_hat[M] + n*bp2[M-3]*w_hat[M])*1j
              + cm1[M-3]*u_hat[M-3]
              + cp1[M-3]*u_hat[M-1])

    for i in xrange(M-4, 0, -1):
        b[i] = ((m*bm2[i]*v_hat[i-1] + n*bm2[i]*w_hat[i-1]
              + m*bii[i]*v_hat[i+1] + n*bii[i]*w_hat[i+1]
              + m*bp2[i]*v_hat[i+3] + n*bp2[i]*w_hat[i+3])*1j
              + cm1[i]*u_hat[i]
              + cp1[i]*u_hat[i+2])

        if i % 2 == 0:
            sum_u0 += u_hat[i+4]
            b[i] += sum_u0*cp3[i]

        else:
            sum_u1 += u_hat[i+4]
            b[i] += sum_u1*cp3[i]

    b[0] = (cm1[0]*u_hat[0]
              + cp1[0]*u_hat[2]
              + 1j*(m*bii[0]*v_hat[1] + n*bii[0]*w_hat[1]
              + m*bp2[0]*v_hat[3] + n*bp2[0]*w_hat[3]))
    sum_u0 += u_hat[4]
    b[0] += sum_u0*cp3[0]

cdef Mult_CTD_1D_ptr(int N,
                     complex_t* v_hat,
                     complex_t* w_hat,
                     complex_t* bv,
                     complex_t* bw,
                     int st):
    cdef:
        int i, ii
        double complex sum_u0, sum_u1, sum_u2, sum_u3

    sum_u0 = 0.0
    sum_u1 = 0.0
    sum_u2 = 0.0
    sum_u3 = 0.0

    bv[(N-1)*st] = 0.0
    bv[(N-2)*st] = -2.*(N-1)*v_hat[(N-3)*st]
    bv[(N-3)*st] = -2.*(N-2)*v_hat[(N-4)*st]
    bw[(N-1)*st] = 0.0
    bw[(N-2)*st] = -2.*(N-1)*w_hat[(N-3)*st]
    bw[(N-3)*st] = -2.*(N-2)*w_hat[(N-4)*st]

    for i in xrange(N-4, 0, -1):
        ii = i*st
        bv[ii] = -2.0*(i+1)*v_hat[(i-1)*st]
        bw[ii] = -2.0*(i+1)*w_hat[(i-1)*st]

        if i % 2 == 0:
            sum_u0 = sum_u0 + v_hat[(i+1)*st]
            sum_u2 = sum_u2 + w_hat[(i+1)*st]

            bv[ii] -= sum_u0*4
            bw[ii] -= sum_u2*4

        else:
            sum_u1 += v_hat[(i+1)*st]
            sum_u3 += w_hat[(i+1)*st]

            bv[ii] -= sum_u1*4
            bw[ii] -= sum_u3*4

    sum_u0 += v_hat[st]
    bv[0] = -sum_u0*2
    sum_u2 += w_hat[st]
    bw[0] = -sum_u2*2

def Mult_CTD_3D_ptr(np.int_t N,
                    complex_t[:, :, ::1] v_hat,
                    complex_t[:, :, ::1] w_hat,
                    complex_t[:, :, ::1] bv,
                    complex_t[:, :, ::1] bw,
                    int axis):
    cdef int i, j, k, strides

    strides = v_hat.strides[axis]/v_hat.itemsize
    if axis == 0:
        for j in range(v_hat.shape[1]):
            for k in range(v_hat.shape[2]):
                Mult_CTD_1D_ptr(N, &v_hat[0, j, k], &w_hat[0, j, k], &bv[0, j, k], &bw[0, j, k], strides)
    elif axis == 1:
        for i in range(v_hat.shape[0]):
            for k in range(v_hat.shape[2]):
                Mult_CTD_1D_ptr(N, &v_hat[i, 0, k], &w_hat[i, 0, k], &bv[i, 0, k], &bw[i, 0, k], strides)
    elif axis == 2:
        for i in range(v_hat.shape[0]):
            for j in range(v_hat.shape[1]):
                Mult_CTD_1D_ptr(N, &v_hat[i, j, 0], &w_hat[i, j, 0], &bv[i, j, 0], &bw[i, j, 0], strides)
