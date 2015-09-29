import numpy as np
cimport cython
cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False
from libcpp.vector cimport vector

ctypedef fused T:
    np.float64_t
    np.complex128_t

def TDMA_1D(np.ndarray[np.float64_t, ndim=1] a, 
            np.ndarray[np.float64_t, ndim=1] b, 
            np.ndarray[np.float64_t, ndim=1] c, 
            np.ndarray[T, ndim=1] d):
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
        
    return d

    
def BackSubstitution_1D(np.ndarray[np.float64_t, ndim=1] u, 
                        np.ndarray[np.float64_t, ndim=1] f):
    """
    Solve Ux = f, where U is an upper diagonal Shen Poisson matrix aij 
    """
    cdef:
        unsigned int n = u.shape[0]        
        int i, l
        
    for i in range(n-1, -1, -1):
        for l in range(i+2, n, 2):
            f[i] += 4*np.pi*(i+1)*u[l]
        u[i] = -f[i] / (2*np.pi*(i+1)*(i+2))
        
    return u

def BackSubstitution_1D_complex(np.ndarray[np.complex128_t, ndim=1] u, 
                                np.ndarray[np.complex128_t, ndim=1] f):
    """
    Solve Ux = f, where U is an upper diagonal Shen Poisson matrix aij 
    """
    cdef:
        unsigned int n = u.shape[0]        
        int i, l
        
    for i in range(n-1, -1, -1):
        for l in range(i+2, n, 2):
            f[i].real = f[i].real + 4*np.pi*(i+1)*u[l].real
            f[i].imag = f[i].imag + 4*np.pi*(i+1)*u[l].imag
        u[i] = -f[i] / (2*np.pi*(i+1)*(i+2))
        
    return u

def BackSubstitution_3D(np.ndarray[np.float64_t, ndim=3] u, 
                        np.ndarray[np.float64_t, ndim=3] f):
    cdef:
        unsigned int n = u.shape[0]
        int i, l
        
    fc = np.zeros(n)
    uc = np.zeros(n)
    for ii in range(u.shape[1]):
        for jj in range(u.shape[2]):
            for i in xrange(n):
                fc[i] = f[i, ii, jj]
                uc[i] = u[i, ii, jj]
            u[:, ii, jj] = BackSubstitution_1D(uc, fc)
        
    return u
 
def BackSubstitution_3D_complex(np.ndarray[np.complex128_t, ndim=3] u, 
                                np.ndarray[np.complex128_t, ndim=3] f):
    cdef:
        unsigned int n = u.shape[0]
        int i, l
        vector[double complex] fc
        vector[double complex] uc
        
    fc.resize(n)
    uc.resize(n)
    for ii in range(u.shape[1]):
        for jj in range(u.shape[2]):
            for i in xrange(n):
                fc[i] = f[i, ii, jj]
                uc[i] = u[i, ii, jj]
            u[:, ii, jj] = BackSubstitution_1D_complex(uc, fc)
        
    return u


def LU_Helmholtz_3D(np.int_t N,
                    bint neumann,
                    bint GC,
                    np.ndarray[np.float64_t, ndim=2] alfa, 
                    np.ndarray[np.float64_t, ndim=4] d0,
                    np.ndarray[np.float64_t, ndim=4] d1,
                    np.ndarray[np.float64_t, ndim=4] d2,
                    np.ndarray[np.float64_t, ndim=4] L):
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


def Solve_Helmholtz_3D_complex(np.int_t N,
                       bint neumann,
                       np.ndarray[np.complex128_t, ndim=3] fk,
                       np.ndarray[np.complex128_t, ndim=3] uk,
                       np.ndarray[np.float64_t, ndim=4] d0,
                       np.ndarray[np.float64_t, ndim=4] d1,
                       np.ndarray[np.float64_t, ndim=4] d2,
                       np.ndarray[np.float64_t, ndim=4] L):
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
                       np.ndarray[np.float64_t, ndim=3] fk,
                       np.ndarray[np.float64_t, ndim=3] uk,
                       np.ndarray[np.float64_t, ndim=4] d0,
                       np.ndarray[np.float64_t, ndim=4] d1,
                       np.ndarray[np.float64_t, ndim=4] d2,
                       np.ndarray[np.float64_t, ndim=4] L):
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
                       np.ndarray[np.float64_t, ndim=1] fk,
                       np.ndarray[np.float64_t, ndim=1] uk,
                       np.ndarray[np.float64_t, ndim=2] d0,
                       np.ndarray[np.float64_t, ndim=2] d1,
                       np.ndarray[np.float64_t, ndim=2] d2,
                       np.ndarray[np.float64_t, ndim=2] L):
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
                          np.ndarray[np.float64_t, ndim=1] fk,
                          np.ndarray[np.float64_t, ndim=1] u_hat,
                          np.ndarray[np.float64_t, ndim=1] d0,
                          np.ndarray[np.float64_t, ndim=1] d1,
                          np.ndarray[np.float64_t, ndim=1] d2,
                          np.ndarray[np.float64_t, ndim=1] L):
    """
    Solve (A+k**2*B)x = f, where A and B are stiffness and mass matrices of Shen with Dirichlet BC
    """
    cdef:
        unsigned int M
        int i
        double tmp
        vector[double] y
        vector[double] u0

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
                 vector[double]& y,
                 vector[double]& u0,
                 np.ndarray[np.float64_t, ndim=1] d0,
                 np.ndarray[np.float64_t, ndim=1] d1,
                 np.ndarray[np.float64_t, ndim=1] d2,
                 np.ndarray[np.float64_t, ndim=1] u_hat):
    cdef:
        int i, j
        double sum_u0 = 0.0
        
    u0[M] = y[M] / d0[M]    
    for i in xrange(M-1, -1, -1):
        u0[i] = y[i] - d1[i]*u0[i+1]
        if i < M-1:
            sum_u0 += u0[i+2]
            u0[i] -= sum_u0*d2[i]            
        u0[i] /= d0[i]
        u_hat[2*i+odd] = u0[i]
    u_hat[2*M+odd] = u0[M]    
    

cdef ForwardSolve_L(vector[double]& y, 
                    np.ndarray[np.float64_t, ndim=1] L, 
                    bint odd,
                    np.ndarray[np.float64_t, ndim=1] fk):
    # Solve Forward Ly = f
    cdef int i
    y[0] = fk[odd]
    for i in xrange(1, y.size()):
        y[i] = fk[2*i+odd] - L[i-1]*y[i-1]
    

def LU_Helmholtz_1D(np.int_t N,
                    bint neumann,
                    bint GC,
                    np.float_t alfa, 
                    np.ndarray[np.float64_t, ndim=2] d0,
                    np.ndarray[np.float64_t, ndim=2] d1,
                    np.ndarray[np.float64_t, ndim=2] d2,
                    np.ndarray[np.float64_t, ndim=2] L):
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
                       np.ndarray[np.float64_t, ndim=1] d0,
                       np.ndarray[np.float64_t, ndim=1] d1,
                       np.ndarray[np.float64_t, ndim=1] d2,
                       np.ndarray[np.float64_t, ndim=1] L):
    cdef:
        unsigned int M
        int i
        int c0 = 0
        double pi = np.pi
        double kx = alfa*alfa
        vector[double] d, s, g, bij_e
                
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
                        np.ndarray[np.float64_t, ndim=1] d0,
                        np.ndarray[np.float64_t, ndim=1] d1,
                        np.ndarray[np.float64_t, ndim=1] d2,
                        np.ndarray[np.float64_t, ndim=1] L):
    cdef:
        unsigned int M
        int i, kk
        int c0 = 0
        double pi = np.pi
        double halfpi = np.pi/2.0
        double kx = alfa*alfa
        vector[double] d, s, g, bii, bip, bim
        int kp2
        long long int ks
        double kp24, kkp2
                
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


def Mult_Helmholtz_3D_complex(np.int_t N,
                      bint GC, np.float_t factor,
                      np.ndarray[np.float64_t, ndim=2] alfa, 
                      np.ndarray[np.complex128_t, ndim=3] u_hat,
                      np.ndarray[np.complex128_t, ndim=3] b):
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
                      np.ndarray[np.float64_t, ndim=2] alfa, 
                      np.ndarray[np.float64_t, ndim=3] u_hat,
                      np.ndarray[np.float64_t, ndim=3] b):
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
                       np.ndarray[np.float64_t, ndim=1] u_hat,
                       np.ndarray[np.float64_t, ndim=1] b):
    Mult_oe_Helmholtz_1D(N, 0, GC, factor, kx, u_hat, b)
    Mult_oe_Helmholtz_1D(N, 1, GC, factor, kx, u_hat, b)

    
def Mult_oe_Helmholtz_1D(np.int_t N,
                       bint odd,
                       bint GC,
                       np.float_t factor,
                       np.float_t kx,
                       np.ndarray[np.float64_t, ndim=1] u_hat,
                       np.ndarray[np.float64_t, ndim=1] b):
    cdef:
        unsigned int M
        int i
        int c0 = 0
        double pi = np.pi
        vector[double] d, s, g, bij
        double sum_u0 = 0.0
                
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


def Mult_Mass_3D_complex(np.int_t N,
                         bint GC, np.float_t factor,
                         np.ndarray[np.complex128_t, ndim=3] u_hat,
                         np.ndarray[np.complex128_t, ndim=3] b):
    cdef:
        unsigned int ii, jj
        
    for ii in range(u_hat.shape[1]):
        for jj in range(u_hat.shape[2]):
            Mult_Mass_1D(N, GC, factor,
                              u_hat[:, ii, jj].real,
                              b[:, ii, jj].real)
            Mult_Mass_1D(N, GC, factor,
                              u_hat[:, ii, jj].imag,
                              b[:, ii, jj].imag)


def Mult_Mass_3D(np.int_t N,
                 bint GC,
                 np.float_t factor,
                 np.ndarray[np.float64_t, ndim=3] u_hat,
                 np.ndarray[np.float64_t, ndim=3] b):
    cdef:
        unsigned int ii, jj
        
    for ii in range(u_hat.shape[1]):
        for jj in range(u_hat.shape[2]):
            Mult_Helmholtz_1D(N, GC, factor,
                              u_hat[:, ii, jj],
                              b[:, ii, jj])
    

def Mult_Mass_1D(np.int_t N,
                 bint GC,
                 np.float_t factor,
                 np.ndarray[np.float64_t, ndim=1] u_hat,
                 np.ndarray[np.float64_t, ndim=1] b):
    Mult_oe_Mass_1D(N, 0, GC, factor, u_hat, b)
    Mult_oe_Mass_1D(N, 1, GC, factor, u_hat, b)


def Mult_oe_Mass_1D(np.int_t N,
                    bint odd,
                    bint GC,
                    np.float_t factor,
                    np.ndarray[np.float64_t, ndim=1] u_hat,
                    np.ndarray[np.float64_t, ndim=1] b):
    cdef:
        unsigned int M
        int i
        int c0 = 0
        double pi = np.pi
        vector[double] d, s, g, bij
        double sum_u0 = 0.0
                
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

    if odd == 1:
        c0 = 1
        
    for i in xrange(M+1):
        d[i] = bij[i]
        if i < M:
            s[i] = - pi/2
            
    b[2*M+odd] += factor*(-pi/2*u_hat[2*M+odd-2] + d[M]*u_hat[2*M+odd])
    for i in xrange(M-1, 0, -1):
        b[2*i+odd] += factor*(-pi/2*u_hat[2*i+odd-2] + d[i]*u_hat[2*i+odd] + s[i]*u_hat[2*i+odd+2])
    b[odd] += factor*(d[0]*u_hat[odd] + s[0]*u_hat[odd+2])
    

def Mult_Div_3D(np.int_t N,
                np.ndarray[np.float64_t, ndim=2] m,
                np.ndarray[np.float64_t, ndim=2] n,
                np.ndarray[np.complex128_t, ndim=3] u_hat,
                np.ndarray[np.complex128_t, ndim=3] v_hat,
                np.ndarray[np.complex128_t, ndim=3] w_hat,
                np.ndarray[np.complex128_t, ndim=3] b):
    cdef unsigned int i, j
    
    for i in xrange(m.shape[0]):
        for j in xrange(m.shape[1]):
            Mult_Div_1D(N, m[i, j], n[i, j], 
                        u_hat[:, i, j],
                        v_hat[:, i, j],
                        w_hat[:, i, j],
                        b[:, i, j])

def Mult_Div_1D(np.int_t N,
                np.float64_t m,
                np.float64_t n,
                np.ndarray[np.complex128_t, ndim=1] u_hat,
                np.ndarray[np.complex128_t, ndim=1] v_hat,
                np.ndarray[np.complex128_t, ndim=1] w_hat,
                np.ndarray[np.complex128_t, ndim=1] b):
    cdef:
        unsigned int M
        int i
        int c0 = 0
        double pi = np.pi
        vector[double] bii, bm2, bp2, cm1, cp1, cp3
        double complex sum_u0              
        double complex sum_u1
        
    M = (N-3)
        
    sum_u0 = 0.0+0.0*1j
    sum_u1 = 0.0+0.0*1j
    # Direct matvec using the fact that there are only three unique diagonals in matrix
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

def Mult_DPhidT_3D(np.int_t N,
                np.ndarray[np.complex128_t, ndim=3] v_hat,
                np.ndarray[np.complex128_t, ndim=3] w_hat,
                np.ndarray[np.complex128_t, ndim=3] bv,
                np.ndarray[np.complex128_t, ndim=3] bw):
    cdef unsigned int i, j
    
    for i in xrange(v_hat.shape[1]):
        for j in xrange(v_hat.shape[2]):
            Mult_DPhidT_1D(N, 
                           v_hat[:, i, j],
                           w_hat[:, i, j],
                           bv[:, i, j],
                           bw[:, i, j])
    
def Mult_DPhidT_1D(np.int_t N,
                np.ndarray[np.complex128_t, ndim=1] v_hat,
                np.ndarray[np.complex128_t, ndim=1] w_hat,
                np.ndarray[np.complex128_t, ndim=1] bv,
                np.ndarray[np.complex128_t, ndim=1] bw):
    cdef:
        int i
        double pi = np.pi
        double complex sum_u0, sum_u1, sum_u2, sum_u3 
        
    sum_u0 = 0.0+0.0*1j
    sum_u1 = 0.0+0.0*1j
    sum_u2 = 0.0+0.0*1j
    sum_u3 = 0.0+0.0*1j
    
    # Direct matvec using the fact that there are only three unique diagonals in matrix            
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
            
            bv[i].real -= sum_u0.real*2
            bv[i].imag -= sum_u0.imag*2
            bw[i].real -= sum_u2.real*2
            bw[i].imag -= sum_u2.imag*2
            
        else:
            sum_u1.real += v_hat[i+1].real
            sum_u1.imag += v_hat[i+1].imag
            sum_u3.real += w_hat[i+1].real
            sum_u3.imag += w_hat[i+1].imag
            
            bv[i].real -= sum_u1.real*2
            bv[i].imag -= sum_u1.imag*2
            bw[i].real -= sum_u3.real*2
            bw[i].imag -= sum_u3.imag*2

    sum_u0.real += v_hat[1].real
    sum_u0.imag += v_hat[1].imag
    bv[0].real = -sum_u0.real*1
    bv[0].imag = -sum_u0.imag*1
    sum_u2.real += w_hat[1].real
    sum_u2.imag += w_hat[1].imag
    bw[0].real = -sum_u2.real*1
    bw[0].imag = -sum_u2.imag*1
    