#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport cython
cimport numpy as np
from libcpp.vector cimport vector

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t

ctypedef fused T:
    real_t
    complex_t


def Biharmonic_matvec3D(np.ndarray[T, ndim=3] v,
                        np.ndarray[T, ndim=3] b,
                        np.float_t a0,
                        np.ndarray[real_t, ndim=2] alfa,
                        np.ndarray[real_t, ndim=2] beta,
                        # 3 upper diagonals of SBB
                        np.ndarray[real_t, ndim=1] sii,
                        np.ndarray[real_t, ndim=1] siu,
                        np.ndarray[real_t, ndim=1] siuu,
                        # All 3 diagonals of ABB
                        np.ndarray[real_t, ndim=1] ail,
                        np.ndarray[real_t, ndim=1] aii,
                        np.ndarray[real_t, ndim=1] aiu,
                        # All 5 diagonals of BBB
                        np.ndarray[real_t, ndim=1] bill,
                        np.ndarray[real_t, ndim=1] bil,
                        np.ndarray[real_t, ndim=1] bii,
                        np.ndarray[real_t, ndim=1] biu,
                        np.ndarray[real_t, ndim=1] biuu):
    cdef:
        int i, j, k, jj
        int N = sii.shape[0]
        double ldd, ld, dd, ud, udd
        double p, r
        T d
        np.ndarray[T, ndim=2] s1 = np.zeros((alfa.shape[0], alfa.shape[1]), dtype=v.dtype)
        np.ndarray[T, ndim=2] s2 = np.zeros((alfa.shape[0], alfa.shape[1]), dtype=v.dtype)
        np.ndarray[T, ndim=2] o1 = np.zeros((alfa.shape[0], alfa.shape[1]), dtype=v.dtype)
        np.ndarray[T, ndim=2] o2 = np.zeros((alfa.shape[0], alfa.shape[1]), dtype=v.dtype)

    #for i in range(v.shape[1]):
        #for j in range(v.shape[2]):
            #Biharmonic_matvec(v[:, i, j], b[:, i, j], a0, alfa[i, j],
                              #beta[i, j], sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu)

    for i in range(v.shape[1]):
        for j in range(v.shape[2]):
            k = N-1
            dd = a0*sii[k] + alfa[i, j]*aii[k] + beta[i, j]*bii[k]
            ld = alfa[i, j]*ail[k-2] + beta[i, j]*bil[k-2]
            ldd = beta[i, j]*bill[k-4]
            b[k, i, j] = ldd*v[k-4, i, j]+ ld*v[k-2, i, j] + dd*v[k, i, j]
            k = N-2
            dd = a0*sii[k] + alfa[i, j]*aii[k] + beta[i, j]*bii[k]
            ld = alfa[i, j]*ail[k-2] + beta[i, j]*bil[k-2]
            ldd = beta[i, j]*bill[k-4]
            b[k, i, j] = ldd*v[k-4, i, j]+ ld*v[k-2, i, j] + dd*v[k, i, j]
            k = N-3
            dd = a0*sii[k] + alfa[i, j]*aii[k] + beta[i, j]*bii[k]
            ld = alfa[i, j]*ail[k-2] + beta[i, j]*bil[k-2]
            ldd = beta[i, j]*bill[k-4]
            ud = a0*siu[k] + alfa[i, j]*aiu[k] + beta[i, j]*biu[k]
            b[k, i, j] = ldd*v[k-4, i, j]+ ld* v[k-2, i, j] + dd*v[k, i, j] + ud*v[k+2, i, j]
            k = N-4
            dd = a0*sii[k] + alfa[i, j]*aii[k] + beta[i, j]*bii[k]
            ld = alfa[i, j]*ail[k-2] + beta[i, j]*bil[k-2]
            ldd = beta[i, j]*bill[k-4]
            ud = a0*siu[k] + alfa[i, j]*aiu[k] + beta[i, j]*biu[k]
            b[k, i, j] = ldd*v[k-4, i, j]+ ld* v[k-2, i, j] + dd*v[k, i, j] + ud*v[k+2, i, j]
            k = N-5
            dd = a0*sii[k] + alfa[i, j]*aii[k] + beta[i, j]*bii[k]
            ld = alfa[i, j]*ail[k-2] + beta[i, j]*bil[k-2]
            ldd = beta[i, j]*bill[k-4]
            ud = a0*siu[k] + alfa[i, j]*aiu[k] + beta[i, j]*biu[k]
            udd = a0*siuu[k] + beta[i, j]*biuu[k]
            b[k, i, j] = ldd*v[k-4, i, j]+ ld* v[k-2, i, j] + dd*v[k, i, j] + ud*v[k+2, i, j] + udd*v[k+4, i, j]
            k = N-6
            dd = a0*sii[k] + alfa[i, j]*aii[k] + beta[i, j]*bii[k]
            ld = alfa[i, j]*ail[k-2] + beta[i, j]*bil[k-2]
            ldd = beta[i, j]*bill[k-4]
            ud = a0*siu[k] + alfa[i, j]*aiu[k] + beta[i, j]*biu[k]
            udd = a0*siuu[k] + beta[i, j]*biuu[k]
            b[k, i, j] = ldd*v[k-4, i, j]+ ld* v[k-2, i, j] + dd*v[k, i, j] + ud*v[k+2, i, j] + udd*v[k+4, i, j]

    for k in xrange(N-7, -1, -1):
        jj = k+6
        p = k*sii[k]/(k+1.)
        r = 24*(k+1)*(k+2)*np.pi
        for i in xrange(v.shape[1]):
            for j in xrange(v.shape[2]):
                dd = a0*sii[k] + alfa[i, j]*aii[k] + beta[i, j]*bii[k]
                ud = a0*siu[k] + alfa[i, j]*aiu[k] + beta[i, j]*biu[k]
                udd = a0*siuu[k] + beta[i, j]*biuu[k]
                d = v[jj, i, j]/(jj+3.)
                if k % 2 == 0:
                    s1[i, j] += d
                    s2[i, j] += (jj+2)*(jj+2)*d
                    b[k, i, j] = (p*s1[i, j] + r*s2[i, j])*a0
                else:
                    o1[i, j] += d
                    o2[i, j] += (jj+2)*(jj+2)*d
                    b[k, i, j] = (p*o1[i, j] + r*o2[i, j])*a0

                if k > 3:
                    ld = alfa[i, j]*ail[k-2] + beta[i, j]*bil[k-2]
                    ldd = beta[i, j]*bill[k-4]
                    b[k, i, j] += ldd*v[k-4, i, j]+ ld* v[k-2, i, j] + dd*v[k, i, j] + ud*v[k+2, i, j] + udd*v[k+4, i, j]
                elif k > 1:
                    ld = alfa[i, j]*ail[k-2] + beta[i, j]*bil[k-2]
                    b[k, i, j] += ld*v[k-2, i, j] + dd*v[k, i, j] + ud*v[k+2, i, j] + udd*v[k+4, i, j]
                else:
                    b[k, i, j] += dd*v[k, i, j] + ud*v[k+2, i, j] + udd*v[k+4, i, j]

def Biharmonic_matvec(np.ndarray[T, ndim=1] v,
                      np.ndarray[T, ndim=1] b,
                      np.float_t a0,
                      np.float_t alfa,
                      np.float_t beta,
                      # 3 upper diagonals of SBB
                      np.ndarray[real_t, ndim=1] sii,
                      np.ndarray[real_t, ndim=1] siu,
                      np.ndarray[real_t, ndim=1] siuu,
                      # All 3 diagonals of ABB
                      np.ndarray[real_t, ndim=1] ail,
                      np.ndarray[real_t, ndim=1] aii,
                      np.ndarray[real_t, ndim=1] aiu,
                      # All 5 diagonals of BBB
                      np.ndarray[real_t, ndim=1] bill,
                      np.ndarray[real_t, ndim=1] bil,
                      np.ndarray[real_t, ndim=1] bii,
                      np.ndarray[real_t, ndim=1] biu,
                      np.ndarray[real_t, ndim=1] biuu):

    cdef:
        int i, j, k
        int N = sii.shape[0]
        vector[double] ldd, ld, dd, ud, udd
        double p, r
        T d, s1, s2, o1, o2

    for i in xrange(N):
        dd.push_back(a0*sii[i] + alfa*aii[i] + beta*bii[i])

    for i in xrange(ail.shape[0]):
        ld.push_back(alfa*ail[i] + beta*bil[i])

    for i in xrange(bill.shape[0]):
        ldd.push_back(beta*bill[i])

    for i in xrange(siu.shape[0]):
        ud.push_back(a0*siu[i] + alfa*aiu[i] + beta*biu[i])

    for i in xrange(siuu.shape[0]):
        udd.push_back(a0*siuu[i] + beta*biuu[i])

    i = N-1
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i]
    i = N-2
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i]
    i = N-3
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-4
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-5
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2] + udd[i]*v[i+4]
    i = N-6
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2] + udd[i]*v[i+4]

    s1 = 0.0
    s2 = 0.0
    o1 = 0.0
    o2 = 0.0
    for k in xrange(N-7, -1, -1):
        j = k+6
        p = k*sii[k]/(k+1.)
        r = 24*(k+1)*(k+2)*np.pi
        d = v[j]/(j+3.)
        if k % 2 == 0:
            s1 += d
            s2 += (j+2)*(j+2)*d
            b[k] = (p*s1 + r*s2)*a0
        else:
            o1 += d
            o2 += (j+2)*(j+2)*d
            b[k] = (p*o1 + r*o2)*a0

        if k > 3:
            b[k] += ldd[k-4]*v[k-4]+ ld[k-2]* v[k-2] + dd[k]*v[k] + ud[k]*v[k+2] + udd[k]*v[k+4]
        elif k > 1:
            b[k] += ld[k-2]* v[k-2] + dd[k]*v[k] + ud[k]*v[k+2] + udd[k]*v[k+4]
        else:
            b[k] += dd[k]*v[k] + ud[k]*v[k+2] + udd[k]*v[k+4]


def Helmholtz_matvec(np.ndarray[T, ndim=1] v,
                     np.ndarray[T, ndim=1] b,
                     real_t alfa,
                     real_t beta,
                     np.ndarray[real_t, ndim=1] dd,
                     np.ndarray[real_t, ndim=1] ud,
                     np.ndarray[real_t, ndim=1] bd):
    # b = (alfa*A + beta*B)*v
    # For B matrix ld = ud = -pi/2
    cdef:
        int i, j, k
        int N = dd.shape[0]
        T s1 = 0.0
        T s2 = 0.0
        double pi_half = np.pi/2
        double p

    k = N-1
    b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - pi_half*beta*v[k-2]
    b[k-1] = (dd[k-1]*alfa + bd[k-1]*beta)*v[k-1] - pi_half*beta*v[k-3]

    for k in range(N-3, 1, -1):
        p = ud[k]*alfa
        if k % 2 == 0:
            s2 += v[k+2]
            b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - pi_half*beta*(v[k-2] + v[k+2]) + p*s2
        else:
            s1 += v[k+2]
            b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - pi_half*beta*(v[k-2] + v[k+2]) + p*s1

    k = 1
    s1 += v[k+2]
    s2 += v[k+1]
    b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - pi_half*beta*v[k+2] + ud[k]*alfa*s1
    b[k-1] = (dd[k-1]*alfa + bd[k-1]*beta)*v[k-1] - pi_half*beta*v[k+1] + ud[k-1]*alfa*s2


def Helmholtz_matvec3D(np.ndarray[T, ndim=3, mode='c'] v,
                       np.ndarray[T, ndim=3, mode='c'] b,
                       real_t alfa,
                       np.ndarray[real_t, ndim=2, mode='c'] beta,
                       np.ndarray[real_t, ndim=1, mode='c'] dd,
                       np.ndarray[real_t, ndim=1, mode='c'] ud,
                       np.ndarray[real_t, ndim=1, mode='c'] bd):
    # b = (alfa*A + beta*B)*v
    # For B matrix ld = ud = -pi/2
    cdef:
        int i, j, k
        int N = dd.shape[0]
        np.ndarray[T, ndim=2] s1 = np.zeros((v.shape[1], v.shape[2]), dtype=v.dtype)
        np.ndarray[T, ndim=2] s2 = np.zeros((v.shape[1], v.shape[2]), dtype=v.dtype)
        double pi_half = np.pi/2
        double p

    k = N-1
    for i in xrange(v.shape[1]):
        for j in xrange(v.shape[2]):
            b[k, i, j] = (dd[k]*alfa + bd[k]*beta[i, j])*v[k, i, j] - pi_half*beta[i, j]*v[k-2, i, j]
            b[k-1, i, j] = (dd[k-1]*alfa + bd[k-1]*beta[i, j])*v[k-1, i, j] - pi_half*beta[i, j]*v[k-3, i, j]

    for k in range(N-3, 1, -1):
        p = ud[k]*alfa
        for i in xrange(v.shape[1]):
            for j in xrange(v.shape[2]):
                if k % 2 == 0:
                    s2[i, j] += v[k+2, i, j]
                    b[k, i, j] = (dd[k]*alfa + bd[k]*beta[i, j])*v[k, i, j] - pi_half*beta[i, j]*(v[k-2, i, j] + v[k+2, i, j]) + p*s2[i, j]
                else:
                    s1[i, j] += v[k+2, i, j]
                    b[k, i, j] = (dd[k]*alfa + bd[k]*beta[i, j])*v[k, i, j] - pi_half*beta[i, j]*(v[k-2, i, j] + v[k+2, i, j]) + p*s1[i, j]

    k = 1
    for i in xrange(v.shape[1]):
        for j in xrange(v.shape[2]):
            s1[i, j] += v[k+2, i, j]
            s2[i, j] += v[k+1, i, j]
            b[k, i, j] = (dd[k]*alfa + bd[k]*beta[i, j])*v[k, i, j] - pi_half*beta[i, j]*v[k+2, i, j] + ud[k]*alfa*s1[i, j]
            b[k-1, i, j] = (dd[k-1]*alfa + bd[k-1]*beta[i, j])*v[k-1, i, j] - pi_half*beta[i, j]*v[k+1, i, j] + ud[k-1]*alfa*s2[i, j]
