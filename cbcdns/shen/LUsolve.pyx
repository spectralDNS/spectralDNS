####!python
####cython: boundscheck=False
####cython: wraparound=False
import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t
ctypedef double real

ctypedef fused T:
    real_t
    complex_t

def LU_Helmholtz_3D(np.int_t N,
                    bint neumann,
                    bint GC,
                    np.ndarray[real_t, ndim=2] alfa, 
                    np.ndarray[real_t, ndim=4] d0,
                    np.ndarray[real_t, ndim=4] d1,
                    np.ndarray[real_t, ndim=4] d2,
                    np.ndarray[real_t, ndim=4] L):
    cdef:
        unsigned int ii, jj
        
    for ii in range(d0.shape[2]):
        for jj in range(d0.shape[3]):
            LU_Helmholtz_1D(N, neumann, GC,
                            alfa[ii, jj],
                            d0[:, :, ii, jj],
                            d1[:, :, ii, jj],
                            d2[:, :, ii, jj],
                            L [:, :, ii, jj])


def LU_Helmholtz_1D(np.int_t N,
                    bint neumann,
                    bint GC,
                    np.float_t alfa, 
                    np.ndarray[real_t, ndim=2] d0,
                    np.ndarray[real_t, ndim=2] d1,
                    np.ndarray[real_t, ndim=2] d2,
                    np.ndarray[real_t, ndim=2] L):
    if not neumann:
        LU_oe_Helmholtz_1D(N, 0, GC, alfa, d0[0], d1[0], d2[0], L[0])
        LU_oe_Helmholtz_1D(N, 1, GC, alfa, d0[1], d1[1], d2[1], L[1])
    
    else:
        LU_oe_HelmholtzN_1D(N, 0, GC, alfa, d0[0], d1[0], d2[0], L[0])
        LU_oe_HelmholtzN_1D(N, 1, GC, alfa, d0[1], d1[1], d2[1], L[1])
        

def LU_oe_Helmholtz_1D(np.int_t N,
                       bint odd,
                       bint GC,
                       np.float_t alfa, 
                       np.ndarray[real_t, ndim=1] d0,
                       np.ndarray[real_t, ndim=1] d1,
                       np.ndarray[real_t, ndim=1] d2,
                       np.ndarray[real_t, ndim=1] L):
    cdef:
        unsigned int M
        int i
        int c0 = 0
        real pi = np.pi
        real kx = alfa*alfa
        vector[real] d, s, g, bij_e
                
    if not odd:
        M = (N-3)/2
    else:
        M = (N-4)/2
        
    # Direct LU decomposition using the fact that there are only three unique diagonals in U and one in L
    for i in range(M+1):
        bij_e.push_back(pi)
    
    if odd == 0:
        bij_e[0] *= 1.5
        if N % 2 == 1 and GC:
            bij_e[M] *= 1.5
    
    else:
        if N % 2 == 0 and GC:
            bij_e[M] *= 1.5
        
    d.resize(M+1)
    s.resize(M)
    g.resize(M-1)

    if odd == 1:
        c0 = 1
        
    for i in xrange(M+1):
        d[i] = 2*pi*(2*i+1+c0)*(2*i+2+c0) + kx*bij_e[i]        
        if i < M:
            s[i] = 4*pi*(2*i+1+c0) - kx*pi/2
        if i < M-1:
            g[i] = 4*pi*(2*i+1+c0)
            
    d0[0] = d[0]
    d1[0] = s[0]
    d2[0] = g[0]
    L[0] = (-kx*pi/2) / d0[0]
    d0[1] = d[1] - L[0]*s[0]
    d1[1] = s[1] - L[0]*g[0]
    d2[1] = g[1] - L[0]*g[0]
    for i in range(1, M):
        L[i] = (-kx*pi/2) / d0[i]
        d0[i+1] = d[i+1] - L[i]*d1[i]
        if i < M-1:
            d1[i+1] = s[i+1] - L[i]*d2[i]
        if i < M-2:
            d2[i+1] = g[i+1] - L[i]*d2[i]
            

def LU_oe_HelmholtzN_1D(np.int_t N,
                        bint odd,
                        bint GC,
                        np.float_t alfa, 
                        np.ndarray[real_t, ndim=1] d0,
                        np.ndarray[real_t, ndim=1] d1,
                        np.ndarray[real_t, ndim=1] d2,
                        np.ndarray[real_t, ndim=1] L):
    cdef:
        unsigned int M
        int i, kk
        int c0 = 0
        real pi = np.pi
        real halfpi = np.pi/2.0
        real kx = alfa*alfa
        vector[real] d, s, g, bii, bip, bim
        int kp2
        long long int ks
        real kp24, kkp2
                
    if not odd:
        M = (N-4)/2
    else:
        M = (N-5)/2
        
    # Direct LU decomposition using the fact that there are only three unique diagonals in U and one in L
    if odd == 1:
        c0 = 1
        
    for i in xrange(1+c0, N-2, 2):
        #bii.push_back(pi*(1.0/(i*i)+pow((i*1.0)/(i+2), 2)/((i+2)*(i+2)))/2.0)
        ks = i*i
        kp2 = i+2
        kkp2 = (i*1.0)/(i+2)
        kp24 = kkp2*kkp2/(kp2*kp2)
        bii.push_back(halfpi*(1.0/ks+kp24))
        if i < N-4:
            bip.push_back(-halfpi*kp24)
        if i > 2:
            bim.push_back(-halfpi/ks)
                
    if GC:
        if odd == 0:
            if N % 2 == 0:
                bii[M] = pi/2.0*(1.0/((N-3)*(N-3))+2.0*pow(((N-3)*1.0)/(N-1), 2)/((N-1)*(N-1)))

        else:
            if N % 2 == 1:
                bii[M] = pi/2.0*(1.0/((N-3)*(N-3))+2.0*pow(((N-3)*1.0)/(N-1), 2)/((N-1)*(N-1)))

    d.resize(M+1)
    s.resize(M)
    g.resize(M-1)

    for i in xrange(M+1):
        kk = 2*i+1+c0
        kp2 = kk+2
        d[i] = 2*(pi*(kk+1))/kp2 + kx*bii[i] 
        if i < M:
            kd = 4*(pi*(kk+1))/(kp2*kp2)
            s[i] = kd + kx*bip[i]
        if i < M-1:
            g[i] = kd
            
    d0[0] = d[0]
    d1[0] = s[0]
    d2[0] = g[0]
    L[0] = (kx*bim[0]) / d0[0]
    d0[1] = d[1] - L[0]*s[0]
    d1[1] = s[1] - L[0]*g[0]
    d2[1] = g[1] - L[0]*g[0]
    for i in range(1, M):
        L[i] = (kx*bim[i]) / d0[i]
        d0[i+1] = d[i+1] - L[i]*d1[i]
        if i < M-1:
            d1[i+1] = s[i+1] - L[i]*d2[i]
        if i < M-2:
            d2[i+1] = g[i+1] - L[i]*d2[i]

# Fastest version
def Solve_Helmholtz_3D_n(np.int_t N,
                       bint neumann,
                       np.ndarray[complex_t, ndim=3] fk,
                       np.ndarray[complex_t, ndim=3] uk,
                       np.ndarray[real_t, ndim=4] d0,
                       np.ndarray[real_t, ndim=4] d1,
                       np.ndarray[real_t, ndim=4] d2,
                       np.ndarray[real_t, ndim=4] L):
    cdef:
        int i, j, k, M, ke, ko, ii
        np.ndarray[complex_t, ndim=3] y = np.zeros((uk.shape[0], uk.shape[1], uk.shape[2]), dtype=uk.dtype)
        np.ndarray[complex_t, ndim=2] s1 = np.zeros((uk.shape[1], uk.shape[2]), dtype=uk.dtype)
        np.ndarray[complex_t, ndim=2] s2 = np.zeros((uk.shape[1], uk.shape[2]), dtype=uk.dtype)
        
    M = d0.shape[1]
    for j in xrange(uk.shape[1]):
        for k in xrange(uk.shape[2]):
            y[0, j, k] = fk[0, j, k]
            y[1, j, k] = fk[1, j, k]
            
    if neumann:
        for i in xrange(1, M):
            for j in xrange(uk.shape[1]):
                for k in xrange(uk.shape[2]):
                    ke = 2*i
                    ko = ke+1
                    y[ke, j, k] = fk[ke, j, k] - L[0, i-1, j, k]*y[ke-2, j, k]
                    if i < M-1:
                        y[ko, j, k] = fk[ko, j, k] - L[1, i-1, j, k]*y[ko-2, j, k]
            
        for j in xrange(uk.shape[1]):
            for k in xrange(uk.shape[2]):        
                ke = 2*(M-1)
                uk[ke, j, k] = y[ke, j, k] / d0[0, M-1, j, k]    
        
        for i in xrange(M-2, -1, -1):
            for j in xrange(uk.shape[1]):
                for k in xrange(uk.shape[2]):
                    ke = 2*i
                    ko = ke+1
                    uk[ke, j, k] = y[ke, j, k] - d1[0, i, j, k]*uk[ke+2, j, k]
                    if i == M-2:
                        uk[ko, j, k] = y[ko, j, k]
                    else:
                        uk[ko, j, k] = y[ko, j, k] - d1[1, i, j, k]*uk[ko+2, j, k]
                    
                    if i < M-2:
                        s1[j, k] += uk[ke+4, j, k]
                        uk[ke, j, k] -= s1[j, k]*d2[0, i, j, k]
                    if i < M-3:
                        s2[j, k] += uk[ko+4, j, k]
                        uk[ko, j, k] -= s2[j, k]*d2[1, i, j, k]
                    
                    uk[ke, j, k] /= d0[0, i, j, k]
                    uk[ko, j, k] /= d0[1, i, j, k]
        
        for i in xrange(N-3):
            ii = (i+1)*(i+1)
            for j in xrange(uk.shape[1]):
                for k in xrange(uk.shape[2]):
                    uk[i, j, k] = uk[i, j, k] / ii

        
    else:
        for i in xrange(1, M):
            for j in xrange(uk.shape[1]):
                for k in xrange(uk.shape[2]):
                    ke = 2*i
                    ko = ke+1
                    y[ke, j, k] = fk[ke, j, k] - L[0, i-1, j, k]*y[ke-2, j, k]
                    y[ko, j, k] = fk[ko, j, k] - L[1, i-1, j, k]*y[ko-2, j, k]
            
        for j in xrange(uk.shape[1]):
            for k in xrange(uk.shape[2]):        
                ke = 2*(M-1)
                ko = ke+1            
                uk[ke, j, k] = y[ke, j, k] / d0[0, M-1, j, k]    
                uk[ko, j, k] = y[ko, j, k] / d0[1, M-1, j, k]    
        
        for i in xrange(M-2, -1, -1):
            for j in xrange(uk.shape[1]):
                for k in xrange(uk.shape[2]):
                    ke = 2*i
                    ko = ke+1
                    uk[ke, j, k] = y[ke, j, k] - d1[0, i, j, k]*uk[ke+2, j, k]
                    uk[ko, j, k] = y[ko, j, k] - d1[1, i, j, k]*uk[ko+2, j, k]
                    
                    if i < M-2:
                        s1[j, k] += uk[ke+4, j, k]
                        s2[j, k] += uk[ko+4, j, k]
                        uk[ke, j, k] -= s1[j, k]*d2[0, i, j, k]
                        uk[ko, j, k] -= s2[j, k]*d2[1, i, j, k]
                    uk[ke, j, k] /= d0[0, i, j, k]
                    uk[ko, j, k] /= d0[1, i, j, k]



# This version slow due to slices. Could be vastly improved using memoryviews
def Solve_Helmholtz_3D_complex(np.int_t N,
                       bint neumann,
                       np.ndarray[complex_t, ndim=3] fk,
                       np.ndarray[complex_t, ndim=3] uk,
                       np.ndarray[real_t, ndim=4] d0,
                       np.ndarray[real_t, ndim=4] d1,
                       np.ndarray[real_t, ndim=4] d2,
                       np.ndarray[real_t, ndim=4] L):
    cdef:
        unsigned int ii, jj
        
    for ii in range(fk.shape[1]):
        for jj in range(fk.shape[2]):
            Solve_Helmholtz_1D(N, neumann, 
                               fk[:, ii, jj].real, 
                               uk[:, ii, jj].real,
                               d0[:, :, ii, jj],
                               d1[:, :, ii, jj],
                               d2[:, :, ii, jj],
                               L [:, :, ii, jj])
            Solve_Helmholtz_1D(N, neumann, 
                               fk[:, ii, jj].imag, 
                               uk[:, ii, jj].imag,
                               d0[:, :, ii, jj],
                               d1[:, :, ii, jj],
                               d2[:, :, ii, jj],
                               L [:, :, ii, jj])


def Solve_Helmholtz_3D(np.int_t N,
                       bint neumann,
                       np.ndarray[real_t, ndim=3] fk,
                       np.ndarray[real_t, ndim=3] uk,
                       np.ndarray[real_t, ndim=4] d0,
                       np.ndarray[real_t, ndim=4] d1,
                       np.ndarray[real_t, ndim=4] d2,
                       np.ndarray[real_t, ndim=4] L):
    cdef:
        unsigned int ii, jj
        
    for ii in range(fk.shape[1]):
        for jj in range(fk.shape[2]):
            Solve_Helmholtz_1D(N, neumann,
                               fk[:, ii, jj],
                               uk[:, ii, jj],
                               d0[:, :, ii, jj],
                               d1[:, :, ii, jj],
                               d2[:, :, ii, jj],
                               L [:, :, ii, jj])


def Solve_Helmholtz_1D(np.int_t N,
                       bint neumann,
                       np.ndarray[real_t, ndim=1] fk,
                       np.ndarray[real_t, ndim=1] uk,
                       np.ndarray[real_t, ndim=2] d0,
                       np.ndarray[real_t, ndim=2] d1,
                       np.ndarray[real_t, ndim=2] d2,
                       np.ndarray[real_t, ndim=2] L):
    cdef int i
    if not neumann:
        Solve_oe_Helmholtz_1D(N, 0, fk, uk, d0[0], d1[0], d2[0], L[0])
        Solve_oe_Helmholtz_1D(N, 1, fk, uk, d0[1], d1[1], d2[1], L[1])
        
    else:
        Solve_oe_Helmholtz_1D(N-1, 0, fk, uk, d0[0], d1[0], d2[0], L[0])
        Solve_oe_Helmholtz_1D(N-1, 1, fk, uk, d0[1], d1[1], d2[1], L[1])
        for i in xrange(N-3):
            uk[i] = uk[i] / ((i+1)*(i+1))
            

def Solve_oe_Helmholtz_1D(np.int_t N,
                          bint odd,
                          np.ndarray[real_t, ndim=1] fk,
                          np.ndarray[real_t, ndim=1] u_hat,
                          np.ndarray[real_t, ndim=1] d0,
                          np.ndarray[real_t, ndim=1] d1,
                          np.ndarray[real_t, ndim=1] d2,
                          np.ndarray[real_t, ndim=1] L):
    """
    Solve (A+k**2*B)x = f, where A and B are stiffness and mass matrices of Shen with Dirichlet BC
    """
    cdef:
        unsigned int M
        int i
        real tmp
        vector[real] y
        vector[real] u0

    if not odd:
        M = (N-3)/2
    else:
        M = (N-4)/2
        
    y.resize(M+1)
    ForwardSolve_L(y, L, odd, fk)
        
    # Solve Backward U u = y 
    u0.resize(M+1)
    BackSolve_U(M, odd, y, u0, d0, d1, d2, u_hat)

cdef BackSolve_U(int M,
                 bint odd, 
                 vector[real]& y,
                 vector[real]& u0,
                 np.ndarray[real_t, ndim=1] d0,
                 np.ndarray[real_t, ndim=1] d1,
                 np.ndarray[real_t, ndim=1] d2,
                 np.ndarray[real_t, ndim=1] u_hat):
    cdef:
        int i, j
        real sum_u0 = 0.0
        
    u0[M] = y[M] / d0[M]    
    for i in xrange(M-1, -1, -1):
        u0[i] = y[i] - d1[i]*u0[i+1]
        if i < M-1:
            sum_u0 += u0[i+2]
            u0[i] -= sum_u0*d2[i]            
        u0[i] /= d0[i]
        u_hat[2*i+odd] = u0[i]
    u_hat[2*M+odd] = u0[M]    
    

cdef ForwardSolve_L(vector[real]& y, 
                    np.ndarray[real_t, ndim=1] L, 
                    bint odd,
                    np.ndarray[real_t, ndim=1] fk):
    # Solve Forward Ly = f
    cdef int i
    y[0] = fk[odd]
    for i in xrange(1, y.size()):
        y[i] = fk[2*i+odd] - L[i-1]*y[i-1]
    


def Mult_Helmholtz_3D_complex(np.int_t N,
                      bint GC, np.float_t factor,
                      np.ndarray[real_t, ndim=2] alfa, 
                      np.ndarray[complex_t, ndim=3] u_hat,
                      np.ndarray[complex_t, ndim=3] b):
    cdef:
        unsigned int ii, jj
        
    for ii in range(u_hat.shape[1]):
        for jj in range(u_hat.shape[2]):
            Mult_Helmholtz_1D(N, GC, factor,
                              alfa[ii, jj], 
                              u_hat[:, ii, jj].real,
                              b[:, ii, jj].real)
            Mult_Helmholtz_1D(N, GC, factor,
                              alfa[ii, jj], 
                              u_hat[:, ii, jj].imag,
                              b[:, ii, jj].imag)


def Mult_Helmholtz_3D(np.int_t N,
                      bint GC,
                      np.float_t factor,
                      np.ndarray[real_t, ndim=2] alfa, 
                      np.ndarray[real_t, ndim=3] u_hat,
                      np.ndarray[real_t, ndim=3] b):
    cdef:
        unsigned int ii, jj
        
    for ii in range(u_hat.shape[1]):
        for jj in range(u_hat.shape[2]):
            Mult_Helmholtz_1D(N, GC, factor,
                              alfa[ii, jj], 
                              u_hat[:, ii, jj],
                              b[:, ii, jj])
    

def Mult_Helmholtz_1D(np.int_t N,
                       bint GC,
                       np.float_t factor,
                       np.float_t kx, 
                       np.ndarray[real_t, ndim=1] u_hat,
                       np.ndarray[real_t, ndim=1] b):
    Mult_oe_Helmholtz_1D(N, 0, GC, factor, kx, u_hat, b)
    Mult_oe_Helmholtz_1D(N, 1, GC, factor, kx, u_hat, b)

    
def Mult_oe_Helmholtz_1D(np.int_t N,
                       bint odd,
                       bint GC,
                       np.float_t factor,
                       np.float_t kx,
                       np.ndarray[real_t, ndim=1] u_hat,
                       np.ndarray[real_t, ndim=1] b):
    cdef:
        unsigned int M
        int i
        int c0 = 0
        real pi = np.pi
        vector[real] d, s, g, bij
        real sum_u0 = 0.0
                
    if not odd:
        M = (N-3)/2
    else:
        M = (N-4)/2
        
    # Direct matvec using the fact that there are only three unique diagonals in matrix
    for i in range(M+1):
        bij.push_back(pi)
    
    if odd == 0:
        bij[0] *= 1.5
        if N % 2 == 1 and GC:
            bij[M] *= 1.5
    
    else:
        if N % 2 == 0 and GC:
            bij[M] *= 1.5
        
    d.resize(M+1)
    s.resize(M)
    g.resize(M-1)

    if odd == 1:
        c0 = 1
        
    for i in xrange(M+1):
        d[i] = 2*pi*(2*i+1+c0)*(2*i+2+c0) + kx*bij[i]        
        if i < M:
            s[i] = 4*pi*(2*i+1+c0) - kx*pi/2
        if i < M-1:
            g[i] = 4*pi*(2*i+1+c0)
            
    b[2*M+odd] += factor*(-pi/2*kx*u_hat[2*M+odd-2] + d[M]*u_hat[2*M+odd])
    for i in xrange(M-1, 0, -1):
        b[2*i+odd] += factor*(-pi/2*kx*u_hat[2*i+odd-2] + d[i]*u_hat[2*i+odd] + s[i]*u_hat[2*i+odd+2])
        if i < M-1:
            sum_u0 += u_hat[2*i+odd+4]
            b[2*i+odd] += factor*sum_u0*g[i]
    b[odd] += factor*(d[0]*u_hat[odd] + s[0]*u_hat[odd+2] + (sum_u0+u_hat[odd+4])*g[0])


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
            sum_u0.real += u_hat[i+4].real
            sum_u0.imag += u_hat[i+4].imag
            b[i].real += sum_u0.real*cp3[i]
            b[i].imag += sum_u0.imag*cp3[i]
            
        else:
            sum_u1.real += u_hat[i+4].real
            sum_u1.imag += u_hat[i+4].imag
            b[i].real += sum_u1.real*cp3[i]
            b[i].imag += sum_u1.imag*cp3[i]
            
    b[0] = (cm1[0]*u_hat[0] 
              + cp1[0]*u_hat[2] 
              + 1j*(m*bii[0]*v_hat[1] + n*bii[0]*w_hat[1] 
              + m*bp2[0]*v_hat[3] + n*bp2[0]*w_hat[3]))
    sum_u0.real += u_hat[4].real
    sum_u0.imag += u_hat[4].imag
    b[0].real += sum_u0.real*cp3[0]
    b[0].imag += sum_u0.imag*cp3[0]

def Mult_CTD_3D(np.int_t N,
                np.ndarray[complex_t, ndim=3] v_hat,
                np.ndarray[complex_t, ndim=3] w_hat,
                np.ndarray[complex_t, ndim=3] bv,
                np.ndarray[complex_t, ndim=3] bw):
    cdef unsigned int i, j
    
    for i in xrange(v_hat.shape[1]):
        for j in xrange(v_hat.shape[2]):
            Mult_CTD_1D(N, 
                        v_hat[:, i, j],
                        w_hat[:, i, j],
                        bv[:, i, j],
                        bw[:, i, j])
    
def Mult_CTD_1D(np.int_t N,
                np.ndarray[complex_t, ndim=1] v_hat,
                np.ndarray[complex_t, ndim=1] w_hat,
                np.ndarray[complex_t, ndim=1] bv,
                np.ndarray[complex_t, ndim=1] bw):
    cdef:
        int i
        real pi = np.pi
        double complex sum_u0, sum_u1, sum_u2, sum_u3 
        
    sum_u0 = 0.0
    sum_u1 = 0.0
    sum_u2 = 0.0
    sum_u3 = 0.0
    
    bv[N-1] = 0.0
    bv[N-2] = -2.*(N-1)*v_hat[N-3]
    bv[N-3] = -2.*(N-2)*v_hat[N-4]
    bw[N-1] = 0.0
    bw[N-2] = -2.*(N-1)*w_hat[N-3]
    bw[N-3] = -2.*(N-2)*w_hat[N-4]
    
    for i in xrange(N-4, 0, -1):
        bv[i] = -2.0*(i+1)*v_hat[i-1]
        bw[i] = -2.0*(i+1)*w_hat[i-1]
        
        if i % 2 == 0:
            sum_u0.real += v_hat[i+1].real
            sum_u0.imag += v_hat[i+1].imag
            sum_u2.real += w_hat[i+1].real
            sum_u2.imag += w_hat[i+1].imag
            
            bv[i].real -= sum_u0.real*4
            bv[i].imag -= sum_u0.imag*4
            bw[i].real -= sum_u2.real*4
            bw[i].imag -= sum_u2.imag*4
            
        else:
            sum_u1.real += v_hat[i+1].real
            sum_u1.imag += v_hat[i+1].imag
            sum_u3.real += w_hat[i+1].real
            sum_u3.imag += w_hat[i+1].imag
            
            bv[i].real -= sum_u1.real*4
            bv[i].imag -= sum_u1.imag*4
            bw[i].real -= sum_u3.real*4
            bw[i].imag -= sum_u3.imag*4

    sum_u0.real += v_hat[1].real
    sum_u0.imag += v_hat[1].imag
    bv[0].real = -sum_u0.real*2
    bv[0].imag = -sum_u0.imag*2
    sum_u2.real += w_hat[1].real
    sum_u2.imag += w_hat[1].imag
    bw[0].real = -sum_u2.real*2
    bw[0].imag = -sum_u2.imag*2

def LU_Biharmonic_1D(np.float_t a, 
                     np.float_t b, 
                     np.float_t c, 
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
                     np.ndarray[real_t, ndim=1] biuu,
                     # Three upper and two lower diagonals of LU decomposition
                     np.ndarray[real_t, ndim=2] u0,
                     np.ndarray[real_t, ndim=2] u1,
                     np.ndarray[real_t, ndim=2] u2,
                     np.ndarray[real_t, ndim=2] l0,
                     np.ndarray[real_t, ndim=2] l1):
    
    LU_oe_Biharmonic_1D(0, a, b, c, sii[::2], siu[::2], siuu[::2], ail[::2], aii[::2], aiu[::2], bill[::2], bil[::2], bii[::2], biu[::2], biuu[::2], u0[0], u1[0], u2[0], l0[0], l1[0])
    LU_oe_Biharmonic_1D(1, a, b, c, sii[1::2], siu[1::2], siuu[1::2], ail[1::2], aii[1::2], aiu[1::2], bill[1::2], bil[1::2], bii[1::2], biu[1::2], biuu[1::2], u0[1], u1[1], u2[1], l0[1], l1[1])

def LU_oe_Biharmonic_1D(bint odd,
                        np.float_t a, 
                        np.float_t b, 
                        np.float_t c, 
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
                        np.ndarray[real_t, ndim=1] biuu,
                        # Two upper and two lower diagonals of LU decomposition
                        np.ndarray[real_t, ndim=1] u0,
                        np.ndarray[real_t, ndim=1] u1,
                        np.ndarray[real_t, ndim=1] u2,
                        np.ndarray[real_t, ndim=1] l0,
                        np.ndarray[real_t, ndim=1] l1):

    cdef:
        int i, j, kk
        long long int m, k
        real pi = np.pi
        vector[real] c0, c1, c2
        
    M = sii.shape[0]
        
    c0.resize(M)
    c1.resize(M)
    c2.resize(M)
    
    c0[0] = a*sii[0] + b*aii[0] + c*bii[0]
    c0[1] = a*siu[0] + b*aiu[0] + c*biu[0]
    c0[2] = a*siuu[0] + c*biuu[0]
    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(6+odd+2)*(6+odd+2))
    c0[3] = m*a*pi/(6+odd+3.)
    #c0[3] = a*8./(6+odd+3.)*pi*(odd+1.)*(odd+2.)*(odd*(odd+4.)+3.*pow(6+odd+2., 2))
    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(8+odd+2)*(8+odd+2))
    c0[4] = m*a*pi/(8+odd+3.)
    #c0[4] = a*8./(8+odd+3.)*pi*(odd+1.)*(odd+2.)*(odd*(odd+4.)+3.*pow(8+odd+2., 2))
    c1[0] = b*ail[0] + c*bil[0]
    c1[1] = a*sii[1] + b*aii[1] + c*bii[1]
    c1[2] = a*siu[1] + b*aiu[1] + c*biu[1]
    c1[3] = a*siuu[1] + c*biuu[1]
    m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(8+odd+2)*(8+odd+2))
    c1[4] = m*a*pi/(8+odd+3.)
    #c1[4] = a*8./(8+odd+3.)*pi*(odd+3.)*(odd+4.)*((odd+2.)*(odd+6.)+3.*pow(8+odd+2., 2))
    c2[0] = c*bill[0]
    c2[1] = b*ail[1] + c*bil[1]
    c2[2] = a*sii[2] + b*aii[2] + c*bii[2]
    c2[3] = a*siu[2] + b*aiu[2] + c*biu[2]
    c2[4] = a*siuu[2] + c*biuu[2]
    for i in xrange(5, M):
        j = 2*i+odd
        m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(j+2)*(j+2))
        c0[i] = m*a*pi/(j+3.)
        m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(j+2)*(j+2))
        c1[i] = m*a*pi/(j+3.)
        m = 8*(odd+5)*(odd+6)*((odd+4)*(odd+8)+3*(j+2)*(j+2))
        c2[i] = m*a*pi/(j+3.)
        #c0[i] = a*8./(j+3.)*pi*(odd+1.)*(odd+2.)*(odd*(odd+4.)+3.*pow(j+2., 2))
        #c1[i] = a*8./(j+3.)*pi*(odd+3.)*(odd+4.)*((odd+2)*(odd+6.)+3.*pow(j+2., 2))
        #c2[i] = a*8./(j+3.)*pi*(odd+5.)*(odd+6.)*((odd+4)*(odd+8.)+3.*pow(j+2., 2))
        
    u0[0] = c0[0]
    u1[0] = c0[1]    
    u2[0] = c0[2]    
    for kk in xrange(1, M):
        l0[kk-1] = c1[kk-1]/u0[kk-1]
        if kk < M-1:
            l1[kk-1] = c2[kk-1]/u0[kk-1]
            
        for i in xrange(kk, M):
            c1[i] = c1[i] - l0[kk-1]*c0[i]
        
        if kk < M-1:
            for i in xrange(kk, M):
                c2[i] = c2[i] - l1[kk-1]*c0[i]
        
        for i in xrange(kk, M):
            c0[i] = c1[i]
            c1[i] = c2[i]
        
        if kk < M-2:
            c2[kk] = c*bill[kk]
            c2[kk+1] = b*ail[kk+1] + c*bil[kk+1]
            c2[kk+2] = a*sii[kk+2] + b*aii[kk+2] + c*bii[kk+2]
            if kk < M-3:
                c2[kk+3] = a*siu[kk+2] + b*aiu[kk+2] + c*biu[kk+2]
            if kk < M-4:
                c2[kk+4] = a*siuu[kk+2] + c*biuu[kk+2]
            if kk < M-5:
                k = 2*(kk+2)+odd
                for i in xrange(kk+5, M):
                    j = 2*i+odd
                    m = 8*(k+1)*(k+2)*(k*(k+4)+3*(j+2)*(j+2))
                    c2[i] = m*a*pi/(j+3.)
                    #c2[i] = a*8./(j+3.)*pi*(k+1.)*(k+2.)*(k*(k+4.)+3.*pow(j+2., 2))

        u0[kk] = c0[kk]
        if kk < M-1:
            u1[kk] = c0[kk+1]
        if kk < M-2:
            u2[kk] = c0[kk+2]    

cdef ForwardBsolve_L(np.ndarray[T, ndim=1] y, 
                     np.ndarray[real_t, ndim=1] l0,
                     np.ndarray[real_t, ndim=1] l1,
                     np.ndarray[T, ndim=1] fk):
    # Solve Forward Ly = f
    cdef np.intp_t i, N
    y[0] = fk[0]
    y[1] = fk[1] - l0[0]*y[0]
    N = l0.shape[0]
    for i in xrange(2, N):
        y[i] = fk[i] - l0[i-1]*y[i-1] - l1[i-2]*y[i-2]

cdef ForwardBsolve_L3_c(vector[double complex]& y, 
                     np.ndarray[real_t, ndim=1] l0,
                     np.ndarray[real_t, ndim=1] l1,
                     np.ndarray[complex_t, ndim=1] fk):
    # Solve Forward Ly = f
    cdef np.intp_t i, N
    y[0] = fk[0]
    y[1] = fk[1] - l0[0]*y[0]
    N = l0.shape[0]
    for i in xrange(2, N):
        y[i] = fk[i] - l0[i-1]*y[i-1] - l1[i-2]*y[i-2]

def LUC_Biharmonic_1D(np.ndarray[real_t, ndim=2] A,
                      np.ndarray[real_t, ndim=3] U,
                      np.ndarray[real_t, ndim=2] l0,
                      np.ndarray[real_t, ndim=2] l1):
    
    LUC_oe_Biharmonic_1D(A[::2, ::2], U[0], l0[0], l1[0])
    LUC_oe_Biharmonic_1D(A[1::2, 1::2], U[1], l0[1], l1[1])

def LUC_oe_Biharmonic_1D(np.ndarray[real_t, ndim=2] A,
                         np.ndarray[real_t, ndim=2] U,
                         np.ndarray[real_t, ndim=1] l0,
                         np.ndarray[real_t, ndim=1] l1):

    cdef:
        int i, j, k, kk
        
    M = A.shape[0]    
    U[:] = A[:]
    for kk in xrange(1, M):
        l0[kk-1] = U[kk, kk-1]/U[kk-1, kk-1]
        if kk < M-1:
            l1[kk-1] = A[kk+1, kk-1]/U[kk-1, kk-1]
            
        U[kk, kk-1] = 0
        for i in xrange(kk, M):
            U[kk, i] = U[kk, i] - l0[kk-1]*U[kk-1, i]
        
        if kk < M-1:
            U[kk+1, kk-1] = 0
            for i in xrange(kk, M):
                U[kk+1, i] = A[kk+1, i] - l1[kk-1]*U[kk-1, i]

def Solve_LUC_Biharmonic_1D(np.ndarray[real_t, ndim=1] fk,
                            np.ndarray[real_t, ndim=1] uk,
                            np.ndarray[real_t, ndim=3] U,
                            np.ndarray[real_t, ndim=2] l0,
                            np.ndarray[real_t, ndim=2] l1,
                            bint ldu=0):
    cdef:
        int i
    
    Solve_LUC_oe_Biharmonic_1D(fk[::2], uk[::2], U[0], l0[0], l1[0], ldu)
    Solve_LUC_oe_Biharmonic_1D(fk[1::2], uk[1::2], U[1], l0[1], l1[1], ldu)

def Solve_LUC_oe_Biharmonic_1D(np.ndarray[real_t, ndim=1] fk,
                               np.ndarray[real_t, ndim=1] uk,
                               np.ndarray[real_t, ndim=2] U,
                               np.ndarray[real_t, ndim=1] l0,
                               np.ndarray[real_t, ndim=1] l1,
                               bint ldu=0):
    cdef:
        unsigned int M
        int i
        real tmp
        np.ndarray y = fk.copy()
            
    M = U.shape[0]        
    #y.resize(M)
    y[:] = 0
    ForwardBsolve_L(y, l0, l1, fk)
    
    # Solve Backward U u = y 
    if ldu == 1:
        Back_LDUC_solve_U(M, y, uk, U)
    else:
        Back_LUC_solve_U(M, y, uk, U)

cdef Back_LUC_solve_U(int M, 
                      np.ndarray[real_t, ndim=1] f,  # Uc = f
                      np.ndarray[real_t, ndim=1] uk,
                      np.ndarray[real_t, ndim=2] U):
    cdef:
        int i, j, k
        real s
        
    uk[M-1] = f[M-1] / U[M-1, M-1]
    for i in xrange(M-2, -1, -1):
        s = 0.0
        for j in xrange(i+1, M):
            s += U[i, j] * uk[j]
        uk[i] = (f[i] - s) / U[i, i]

cdef Back_LDUC_solve_U(int M, 
                       np.ndarray[real_t, ndim=1] f,  # Uc = f
                       np.ndarray[real_t, ndim=1] uk,
                       np.ndarray[real_t, ndim=2] U):
    cdef:
        int i, j, k
        real s, d
        
    for i in xrange(M):
        d = U[i, i]
        f[i] /= d 
        for j in xrange(i, M):
            U[i, j] = U[i, j] / d
        
    uk[M-1] = f[M-1] / U[M-1, M-1]
    for i in xrange(M-2, -1, -1):
        s = 0.0
        for j in xrange(i+1, M):
            s += U[i, j] * uk[j]
        uk[i] = (f[i] - s) / U[i, i]

def Biharmonic_factor_pr_3D(np.ndarray[real_t, ndim=4] a,
                            np.ndarray[real_t, ndim=4] b,
                            np.ndarray[real_t, ndim=4] l0,
                            np.ndarray[real_t, ndim=4] l1):
    
    cdef:
        unsigned int ii, jj
        
    for ii in range(a.shape[2]):
        for jj in range(a.shape[3]):
            Biharmonic_factor_pr(a[:, :, ii, jj], 
                                 b[:, :, ii, jj], 
                                 l0[:, :, ii, jj], 
                                 l1[:, :, ii, jj])

def Biharmonic_factor_pr(np.ndarray[real_t, ndim=2] a,
                         np.ndarray[real_t, ndim=2] b,
                         np.ndarray[real_t, ndim=2] l0,
                         np.ndarray[real_t, ndim=2] l1):

    Biharmonic_factor_oe_pr(0, a[0], b[0], l0[0], l1[0])
    Biharmonic_factor_oe_pr(1, a[1], b[1], l0[1], l1[1])

def Biharmonic_factor_oe_pr(bint odd,
                            np.ndarray[real_t, ndim=1] a,
                            np.ndarray[real_t, ndim=1] b,
                            np.ndarray[real_t, ndim=1] l0,
                            np.ndarray[real_t, ndim=1] l1):
    cdef:
        int i, j, M
        real pi = np.pi
        long long int pp, rr, k, kk
        
    M = l0.shape[0]+1
    k = odd
    a[0] = 8*k*(k+1)*(k+2)*(k+4)*pi
    b[0] = 24*(k+1)*(k+2)*pi
    k = 2+odd
    a[1] = 8*k*(k+1)*(k+2)*(k+4)*pi - l0[0]*a[0]
    b[1] = 24*(k+1)*(k+2)*pi - l0[0]*b[0]
    for k in xrange(2, M-3):
        kk = 2*k+odd
        pp = 8*kk*(kk+1)*(kk+2)*(kk+4)
        rr = 24*(kk+1)*(kk+2)
        a[k] = pp*pi - l0[k-1]*a[k-1] - l1[k-2]*a[k-2]
        b[k] = rr*pi - l0[k-1]*b[k-1] - l1[k-2]*b[k-2]

    
def Solve_Biharmonic_1D(np.ndarray[T, ndim=1] fk,
                        np.ndarray[T, ndim=1] uk,
                        np.ndarray[real_t, ndim=2] u0,
                        np.ndarray[real_t, ndim=2] u1,
                        np.ndarray[real_t, ndim=2] u2,
                        np.ndarray[real_t, ndim=2] l0,
                        np.ndarray[real_t, ndim=2] l1,
                        np.ndarray[real_t, ndim=2] a, 
                        np.ndarray[real_t, ndim=2] b, 
                        np.float_t ac):
    
    Solve_oe_Biharmonic_1D(0, fk[::2], uk[::2], u0[0], u1[0], u2[0], l0[0], l1[0], a[0], b[0], ac)
    Solve_oe_Biharmonic_1D(1, fk[1::2], uk[1::2], u0[1], u1[1], u2[1], l0[1], l1[1], a[1], b[1], ac)
    
def Solve_Biharmonic_1D_c(np.ndarray[complex_t, ndim=1] fk,
                        np.ndarray[complex_t, ndim=1] uk,
                        np.ndarray[real_t, ndim=2] u0,
                        np.ndarray[real_t, ndim=2] u1,
                        np.ndarray[real_t, ndim=2] u2,
                        np.ndarray[real_t, ndim=2] l0,
                        np.ndarray[real_t, ndim=2] l1,
                        np.ndarray[real_t, ndim=2] a, 
                        np.ndarray[real_t, ndim=2] b, 
                        np.float_t ac):
    
    Solve_oe_Biharmonic_1D_c(0, fk[::2], uk[::2], u0[0], u1[0], u2[0], l0[0], l1[0], a[0], b[0], ac)
    Solve_oe_Biharmonic_1D_c(1, fk[1::2], uk[1::2], u0[1], u1[1], u2[1], l0[1], l1[1], a[1], b[1], ac)


def Solve_oe_Biharmonic_1D_c(bint odd,
                           np.ndarray[complex_t, ndim=1] fk,
                           np.ndarray[complex_t, ndim=1] uk,
                           np.ndarray[real_t, ndim=1] u0,
                           np.ndarray[real_t, ndim=1] u1,
                           np.ndarray[real_t, ndim=1] u2,
                           np.ndarray[real_t, ndim=1] l0,
                           np.ndarray[real_t, ndim=1] l1,
                           np.ndarray[real_t, ndim=1] a,
                           np.ndarray[real_t, ndim=1] b,
                           np.float_t ac):
    """
    Solve (aS+b*A+cB)x = f, where S, A and B are 4th order Laplace, stiffness and mass matrices of Shen with Dirichlet BC
    """
    cdef:
        unsigned int M
        vector[double complex] y
            
    M = u0.shape[0]        
    y.resize(M)
    ForwardBsolve_L3_c(y, l0, l1, fk)
    
    # Solve Backward U u = y 
    BackBsolve_U_c(M, odd, y, uk, u0, u1, u2, l0, l1, a, b, ac)

def Solve_oe_Biharmonic_1D(bint odd,
                           np.ndarray[T, ndim=1] fk,
                           np.ndarray[T, ndim=1] uk,
                           np.ndarray[real_t, ndim=1] u0,
                           np.ndarray[real_t, ndim=1] u1,
                           np.ndarray[real_t, ndim=1] u2,
                           np.ndarray[real_t, ndim=1] l0,
                           np.ndarray[real_t, ndim=1] l1,
                           np.ndarray[real_t, ndim=1] a,
                           np.ndarray[real_t, ndim=1] b,
                           np.float_t ac):
    """
    Solve (aS+b*A+cB)x = f, where S, A and B are 4th order Laplace, stiffness and mass matrices of Shen with Dirichlet BC
    """
    cdef:
        unsigned int M
        np.ndarray[T, ndim=1] y = np.zeros(u0.shape[0], dtype=fk.dtype)
            
    M = u0.shape[0]
    ForwardBsolve_L(y, l0, l1, fk)
    
    # Solve Backward U u = y 
    BackBsolve_U(M, odd, y, uk, u0, u1, u2, l0, l1, a, b, ac)
    
cdef BackBsolve_U(int M,
                  bint odd, 
                  np.ndarray[T, ndim=1] f,  # Uc = f
                  np.ndarray[T, ndim=1] uk,
                  np.ndarray[real_t, ndim=1] u0,
                  np.ndarray[real_t, ndim=1] u1,
                  np.ndarray[real_t, ndim=1] u2,
                  np.ndarray[real_t, ndim=1] l0,
                  np.ndarray[real_t, ndim=1] l1,
                  np.ndarray[real_t, ndim=1] a,
                  np.ndarray[real_t, ndim=1] b,
                  np.float_t ac):
    cdef:
        int i, j, k, kk
        T s1 = 0.0
        T s2 = 0.0
    
    uk[M-1] = f[M-1] / u0[M-1]
    uk[M-2] = (f[M-2] - u1[M-2]*uk[M-1]) / u0[M-2]
    uk[M-3] = (f[M-3] - u1[M-3]*uk[M-2] - u2[M-3]*uk[M-1]) / u0[M-3]
    
    s1 = 0.0
    s2 = 0.0
    for kk in xrange(M-4, -1, -1):
        k = 2*kk+odd
        j = k+6
        s1 += uk[kk+3]/(j+3.)
        s2 += (uk[kk+3]/(j+3.))*((j+2)*(j+2))
        uk[kk] = (f[kk] - u1[kk]*uk[kk+1] - u2[kk]*uk[kk+2] - a[kk]*ac*s1 - b[kk]*ac*s2) / u0[kk]

cdef BackBsolve_U_c(int M,
                  bint odd, 
                  vector[double complex]& f,  # Uc = f
                  np.ndarray[complex_t, ndim=1] uk,
                  np.ndarray[real_t, ndim=1] u0,
                  np.ndarray[real_t, ndim=1] u1,
                  np.ndarray[real_t, ndim=1] u2,
                  np.ndarray[real_t, ndim=1] l0,
                  np.ndarray[real_t, ndim=1] l1,
                  np.ndarray[real_t, ndim=1] a,
                  np.ndarray[real_t, ndim=1] b,
                  np.float_t ac):
    cdef:
        int i, j, k, kk
        double complex s1 = 0.0
        double complex s2 = 0.0
    
    uk[M-1] = f[M-1] / u0[M-1]
    uk[M-2] = (f[M-2] - u1[M-2]*uk[M-1]) / u0[M-2]
    uk[M-3] = (f[M-3] - u1[M-3]*uk[M-2] - u2[M-3]*uk[M-1]) / u0[M-3]
    
    s1 = 0.0
    s2 = 0.0
    for kk in xrange(M-4, -1, -1):
        k = 2*kk+odd
        j = k+6
        s1 += uk[kk+3]/(j+3.)
        s2 += (uk[kk+3]/(j+3.))*((j+2)*(j+2))
        uk[kk] = (f[kk] - u1[kk]*uk[kk+1] - u2[kk]*uk[kk+2] - a[kk]*ac*s1 - b[kk]*ac*s2) / u0[kk]

def Solve_Biharmonic_3D(np.ndarray[T, ndim=3] fk,
                        np.ndarray[T, ndim=3] uk,
                        np.ndarray[real_t, ndim=4] u0,
                        np.ndarray[real_t, ndim=4] u1,
                        np.ndarray[real_t, ndim=4] u2,
                        np.ndarray[real_t, ndim=4] l0,
                        np.ndarray[real_t, ndim=4] l1,
                        np.ndarray[real_t, ndim=4] a,
                        np.ndarray[real_t, ndim=4] b,
                        np.float_t ac):
    cdef:
        unsigned int ii, jj
        
    for ii in range(fk.shape[1]):
        for jj in range(fk.shape[2]):
            Solve_Biharmonic_1D(fk[:, ii, jj], 
                                uk[:, ii, jj],
                                u0[:, :, ii, jj],
                                u1[:, :, ii, jj],
                                u2[:, :, ii, jj],
                                l0[:, :, ii, jj],
                                l1[:, :, ii, jj],
                                a[:, :, ii, jj],
                                b[:, :, ii, jj],
                                ac)

def Solve_Biharmonic_3D_c(np.ndarray[complex_t, ndim=3] fk,
                        np.ndarray[complex_t, ndim=3] uk,
                        np.ndarray[real_t, ndim=4] u0,
                        np.ndarray[real_t, ndim=4] u1,
                        np.ndarray[real_t, ndim=4] u2,
                        np.ndarray[real_t, ndim=4] l0,
                        np.ndarray[real_t, ndim=4] l1,
                        np.ndarray[real_t, ndim=4] a,
                        np.ndarray[real_t, ndim=4] b,
                        np.float_t ac):
    cdef:
        unsigned int ii, jj
        
    for ii in range(fk.shape[1]):
        for jj in range(fk.shape[2]):
            Solve_Biharmonic_1D_c(fk[:, ii, jj], 
                                uk[:, ii, jj],
                                u0[:, :, ii, jj],
                                u1[:, :, ii, jj],
                                u2[:, :, ii, jj],
                                l0[:, :, ii, jj],
                                l1[:, :, ii, jj],
                                a[:, :, ii, jj],
                                b[:, :, ii, jj],
                                ac)
            
def LU_Biharmonic_3D(np.float_t a0,  
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
                     np.ndarray[real_t, ndim=1] biuu,
                     np.ndarray[real_t, ndim=4] u0,
                     np.ndarray[real_t, ndim=4] u1,
                     np.ndarray[real_t, ndim=4] u2,
                     np.ndarray[real_t, ndim=4] l0,
                     np.ndarray[real_t, ndim=4] l1):
    cdef:
        unsigned int ii, jj
        
    for ii in range(u0.shape[2]):
        for jj in range(u0.shape[3]):
            LU_Biharmonic_1D(a0,
                            alfa[ii, jj],
                            beta[ii, jj],
                            sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu,
                            u0[:, :, ii, jj],
                            u1[:, :, ii, jj],
                            u2[:, :, ii, jj],
                            l0[:, :, ii, jj],
                            l1[:, :, ii, jj])
            

def Solve_oe_Biharmonic_1D(bint odd,
                           np.ndarray[T, ndim=1] fk,
                           np.ndarray[T, ndim=1] uk,
                           np.ndarray[real_t, ndim=1] u0,
                           np.ndarray[real_t, ndim=1] u1,
                           np.ndarray[real_t, ndim=1] u2,
                           np.ndarray[real_t, ndim=1] l0,
                           np.ndarray[real_t, ndim=1] l1,
                           np.ndarray[real_t, ndim=1] a,
                           np.ndarray[real_t, ndim=1] b,
                           np.float_t ac):
    """
    Solve (aS+b*A+cB)x = f, where S, A and B are 4th order Laplace, stiffness and mass matrices of Shen with Dirichlet BC
    """
    cdef:
        unsigned int M
        np.ndarray[T, ndim=1] y = np.zeros(u0.shape[0], dtype=fk.dtype)
            
    M = u0.shape[0]
    ForwardBsolve_L(y, l0, l1, fk)
    
    # Solve Backward U u = y 
    BackBsolve_U(M, odd, y, uk, u0, u1, u2, l0, l1, a, b, ac)

# This one is fastest by far
@cython.cdivision(True)
def Solve_Biharmonic_3D_n(np.ndarray[T, ndim=3] fk,
                        np.ndarray[T, ndim=3] uk,
                        np.ndarray[real_t, ndim=4] u0,
                        np.ndarray[real_t, ndim=4] u1,
                        np.ndarray[real_t, ndim=4] u2,
                        np.ndarray[real_t, ndim=4] l0,
                        np.ndarray[real_t, ndim=4] l1,
                        np.ndarray[real_t, ndim=4] a,
                        np.ndarray[real_t, ndim=4] b,
                        np.float_t ac):
    
    cdef:
        int i, j, k, kk, m, M, ke, ko, jj
        np.ndarray[T, ndim=2] s1 = np.zeros((fk.shape[1], fk.shape[2]), dtype=fk.dtype)
        np.ndarray[T, ndim=2] s2 = np.zeros((fk.shape[1], fk.shape[2]), dtype=fk.dtype)
        np.ndarray[T, ndim=2] o1 = np.zeros((fk.shape[1], fk.shape[2]), dtype=fk.dtype)
        np.ndarray[T, ndim=2] o2 = np.zeros((fk.shape[1], fk.shape[2]), dtype=fk.dtype)
        np.ndarray[T, ndim=3] y = np.zeros((fk.shape[0], fk.shape[1], fk.shape[2]), dtype=fk.dtype)


    M = u0.shape[1]
    for j in range(fk.shape[1]):
        for k in range(fk.shape[2]):
            y[0, j, k] = fk[0, j, k]
            y[1, j, k] = fk[1, j, k]
            y[2, j, k] = fk[2, j, k] - l0[0, 0, j, k]*y[0, j, k]
            y[3, j, k] = fk[3, j, k] - l0[1, 0, j, k]*y[1, j, k]
            
    for i in xrange(2, M):
        for j in range(fk.shape[1]):
            for k in range(fk.shape[2]): 
                ke = 2*i
                ko = ke+1
                y[ko, j, k] = fk[ko, j, k] - l0[1, i-1, j, k]*y[ko-2, j, k] - l1[1, i-2, j, k]*y[ko-4, j, k]
                y[ke, j, k] = fk[ke, j, k] - l0[0, i-1, j, k]*y[ke-2, j, k] - l1[0, i-2, j, k]*y[ke-4, j, k]
    
    for j in range(fk.shape[1]):
        for k in range(fk.shape[2]):
            ke = 2*(M-1)
            ko = ke+1
            uk[ke, j, k] = y[ke, j, k] / u0[0, M-1, j, k]
            uk[ko, j, k] = y[ko, j, k] / u0[1, M-1, j, k]
            ke = 2*(M-2)
            ko = ke+1
            uk[ke, j, k] = (y[ke, j, k] - u1[0, M-2, j, k]*uk[ke+2, j, k]) / u0[0, M-2, j, k]
            uk[ko, j, k] = (y[ko, j, k] - u1[1, M-2, j, k]*uk[ko+2, j, k]) / u0[1, M-2, j, k]
            ke = 2*(M-3)
            ko = ke+1
            uk[ke, j, k] = (y[ke, j, k] - u1[0, M-3, j, k]*uk[ke+2, j, k] - u2[0, M-3, j, k]*uk[ke+4, j, k]) / u0[0, M-3, j, k]
            uk[ko, j, k] = (y[ko, j, k] - u1[1, M-3, j, k]*uk[ko+2, j, k] - u2[1, M-3, j, k]*uk[ko+4, j, k]) / u0[1, M-3, j, k]
            
    
    for kk in xrange(M-4, -1, -1):
        for j in range(fk.shape[1]):
            for k in range(fk.shape[2]):
                ke = 2*kk
                ko = ke+1
                jj = ke+6
                s1[j, k] += uk[jj, j, k]/(jj+3.)
                s2[j, k] += (uk[jj, j, k]/(jj+3.))*((jj+2.)*(jj+2.))
                uk[ke, j, k] = (y[ke, j, k] - u1[0, kk, j, k]*uk[ke+2, j, k] - u2[0, kk, j, k]*uk[ke+4, j, k] - a[0, kk, j, k]*ac*s1[j, k] - b[0, kk, j, k]*ac*s2[j, k]) / u0[0, kk, j, k]
                jj = ko+6
                o1[j, k] += uk[jj, j, k]/(jj+3.)
                o2[j, k] += (uk[jj, j, k]/(jj+3.))*((jj+2.)*(jj+2.))
                uk[ko, j, k] = (y[ko, j, k] - u1[1, kk, j, k]*uk[ko+2, j, k] - u2[1, kk, j, k]*uk[ko+4, j, k] - a[1, kk, j, k]*ac*o1[j, k] - b[1, kk, j, k]*ac*o2[j, k]) / u0[1, kk, j, k]
