import numpy as np
cimport cython
cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False
from libcpp.vector cimport vector

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t

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

def SBBmat_matvec(np.ndarray[T, ndim=1] v, 
                  np.ndarray[T, ndim=1] b,
                  np.ndarray[real_t, ndim=1] dd):
    cdef:
        int i, j, k
        int N = v.shape[0]-4
        double p, r
        T d, s1, s2, o1, o2

    j = N-1
    s1 = 0.0
    s2 = 0.0
    o1 = 0.0
    o2 = 0.0
    b[j] = dd[j]*v[j]
    b[j-1] = dd[j-1]*v[j-1]
    for k in range(N-3, -1, -1):
        j = k+2
        p = k*dd[k]/(k+1)
        r = 24*(k+1)*(k+2)*np.pi
        d = v[j]/(j+3.)
        if k % 2 == 0:
            s1 += d
            s2 += (j+2)*(j+2)*d
            b[k] = dd[k]*v[k] + p*s1 + r*s2
        else:
            o1 += d
            o2 += (j+2)*(j+2)*d
            b[k] = dd[k]*v[k] + p*o1 + r*o2
        

def SBBmat_matvec3D(np.ndarray[T, ndim=3] v, 
                    np.ndarray[T, ndim=3] b,
                    np.ndarray[real_t, ndim=1] dd):
    cdef:
        int i, j

    for i in range(v.shape[1]):
        for j in range(v.shape[2]):
            SBBmat_matvec(v[:, i, j], b[:, i, j], dd)

def Biharmonic_matvec3D(np.ndarray[T, ndim=3] v, 
                        np.ndarray[T, ndim=3] c,
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
        int i, j

    for i in range(v.shape[1]):
        for j in range(v.shape[2]):
            Biharmonic_matvec(v[:, i, j], c[:, i, j], a0, alfa[i, j], 
                              beta[i, j], sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu)
            
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

def Tridiagonal_matvec3D(np.ndarray[T, ndim=3] v, 
                         np.ndarray[T, ndim=3] b,
                         np.ndarray[real_t, ndim=1] ld,
                         np.ndarray[real_t, ndim=1] dd,
                         np.ndarray[real_t, ndim=1] ud):
    cdef:
        int i, j

    for i in range(v.shape[1]):
        for j in range(v.shape[2]):
            Tridiagonal_matvec(v[:, i, j], b[:, i, j], ld, dd, ud)

def Tridiagonal_matvec(np.ndarray[T, ndim=1] v, 
                       np.ndarray[T, ndim=1] b,
                       np.ndarray[real_t, ndim=1] ld,
                       np.ndarray[real_t, ndim=1] dd,
                       np.ndarray[real_t, ndim=1] ud):
    cdef:
        int i
        int N = dd.shape[0]
        int m = dd.shape[0] - ud.shape[0]
        
    b[0] = dd[0]*v[0] + ud[0]*v[2]
    b[1] = dd[1]*v[1] + ud[1]*v[3]
    for i in xrange(2, N-2):
        b[i] = ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-2
    b[i] = ld[i-2]* v[i-2] + dd[i]*v[i]
    i = N-1
    b[i] = ld[i-2]* v[i-2] + dd[i]*v[i]
    
def Pentadiagonal_matvec3D(np.ndarray[T, ndim=3] v, 
                    np.ndarray[T, ndim=3] b,
                    np.ndarray[real_t, ndim=1] ldd,
                    np.ndarray[real_t, ndim=1] ld,
                    np.ndarray[real_t, ndim=1] dd,
                    np.ndarray[real_t, ndim=1] ud,
                    np.ndarray[real_t, ndim=1] udd):
    cdef:
        int i, j

    for i in range(v.shape[1]):
        for j in range(v.shape[2]):
            Pentadiagonal_matvec(v[:, i, j], b[:, i, j], ldd, ld, dd, ud, udd)
            
def Pentadiagonal_matvec(np.ndarray[T, ndim=1] v, 
                  np.ndarray[T, ndim=1] b,
                  np.ndarray[real_t, ndim=1] ldd,
                  np.ndarray[real_t, ndim=1] ld,
                  np.ndarray[real_t, ndim=1] dd,
                  np.ndarray[real_t, ndim=1] ud,
                  np.ndarray[real_t, ndim=1] udd):
    cdef:
        int i
        int N = dd.shape[0]
    
    b[0] = dd[0]*v[0] + ud[0]*v[2] + udd[0]*v[4]
    b[1] = dd[1]*v[1] + ud[1]*v[3] + udd[1]*v[5]
    b[2] = ld[0]*v[0] + dd[2]*v[2] + ud[2]*v[4] + udd[2]*v[6]
    b[3] = ld[1]*v[1] + dd[3]*v[3] + ud[3]*v[5] + udd[3]*v[7]
    for i in xrange(4, N-4):
        b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2] + udd[i]*v[i+4]
    i = N-4
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-3
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-2
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i]
    i = N-1
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i]

def CBD_matvec3D(np.ndarray[T, ndim=3] v, 
                 np.ndarray[T, ndim=3] b,
                 np.ndarray[real_t, ndim=1] ld,
                 np.ndarray[real_t, ndim=1] ud,
                 np.ndarray[real_t, ndim=1] udd):
    cdef:
        int i, j, k
        int N = udd.shape[0]

    for i in range(v.shape[1]):
        for j in range(v.shape[2]):
            b[0, i, j] = ud[0]*v[1, i, j] + udd[0]*v[3, i, j]
            for k in xrange(1, N):
                b[k, i, j] = ld[k-1]* v[k-1, i, j] + ud[k]*v[k+1, i, j] + udd[k]*v[k+3, i, j]
            k = N
            b[k, i, j] = ld[k-1]* v[k-1, i, j] + ud[k]*v[k+1, i, j]
    
def CBD_matvec(np.ndarray[T, ndim=1] v, 
               np.ndarray[T, ndim=1] b,
               np.ndarray[real_t, ndim=1] ld,
               np.ndarray[real_t, ndim=1] ud,
               np.ndarray[real_t, ndim=1] udd):
    cdef:
        int i
        int N = udd.shape[0]
        
    b[0] = ud[0]*v[1] + udd[0]*v[3]
    for i in xrange(1, N):
        b[i] = ld[i-1]* v[i-1] + ud[i]*v[i+1] + udd[i]*v[i+3]
    i = N
    b[i] = ld[i-1]* v[i-1] + ud[i]*v[i+1]

