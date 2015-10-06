# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:48:17 2015

@author: Diako Darian
"""
import numpy as np
cimport cython
cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False
#from libcpp.vector cimport vector

ctypedef fused T:
    np.float64_t
    np.complex128_t


def B_mat(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=1] ck,
          np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
          np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
          np.ndarray[np.float64_t, ndim=2] c):

    cdef:
        int j, k
        int N = K.shape[0]-2
        double pi = np.pi
  
    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                c[k,j] = (pi/2.)*(ck[j] + a_j[j]*a_k[k] + b_j[j]*b_k[k])
            if k == j-1:    
                c[k,j] = (pi/2.)*(a_k[k] + a_j[j]*b_k[k])
            if k == j+1:    
                c[k,j] = (pi/2.)*(a_j[j] + b_j[j]*a_k[k])
            if k==(j-2): 
                c[k,j] = (pi/2.)*b_k[k]
            if k==(j+2): 
                c[k,j] = (pi/2.)*b_j[j]
    return c

                 
def B_matvec(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] B,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            B_matvec_1D(K,B, v[:, ii, jj].real, b[:, ii, jj].real)
            B_matvec_1D(K,B, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def B_matvec_1D(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] c,
                np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = K.shape[0]-2

    b[0] = v[0]*c[0,0] + v[1]*c[0,1] + v[2]*c[0,2]
    b[1] = v[0]*c[1,0] + v[1]*c[1,1] + v[2]*c[1,2] + v[3]*c[1,3]

    b[(N-2)] = v[(N-4)]*c[N-2,N-4] + v[(N-3)]*c[N-2,N-3] + v[(N-2)]*c[N-2,N-2] + v[N-1]*c[N-2,N-1] 
    b[(N-1)] = v[(N-3)]*c[N-1,N-3] + v[(N-2)]*c[N-1,N-2] + v[N-1]*c[N-1,N-1]
            
    for i in xrange(2, N-2):
        b[i] = v[(i-2)]*c[i,(i-2)] + v[(i-1)]*c[i,i-1] + v[i]*c[i,i] + v[(i+1)]*c[i,i+1] + v[(i+2)]*c[i,i+2] 

    return b

def C_mat(np.ndarray[np.float64_t, ndim=1] K,
          np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
          np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
          np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = K.shape[0]-2
        
    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                c[k,j] = np.pi*(a_j[j]*(K[j] + 1) + b_j[j]*a_k[j]*(K[j] + 2))
            if k == j-1:    
                c[k,j] = np.pi*(K[j] + a_j[j]*a_k[k]*(K[j]+1) + b_j[j]*(K[j]+2) + b_j[j]*b_k[k]*(K[j]+2))
            if k == j+1:    
                c[k,j] = np.pi*b_j[j]*(K[j]+2) 
            if j>(k+1):
                if (j-k)%2==0:
                    c[k,j] = np.pi*(a_k[k]*K[j] + a_j[j]*(K[j]+1) + a_j[j]*b_k[k]*(K[j]+1) + b_j[j]*a_k[k]*(K[j]+2))
                else:
                    c[k,j] = np.pi*(K[j] + b_k[k]*K[j] + a_j[j]*a_k[k]*(K[j]+1) + b_j[j]*(K[j]+2) + b_j[j]*b_k[k]*(K[j]+2))
                             
    return c


def C_matvec(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] C,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            C_matvec_1D(K,C, v[:, ii, jj].real, b[:, ii, jj].real)
            C_matvec_1D(K,C, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def C_matvec_1D(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] C,
                np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = K.shape[0]-2
    
    for k in xrange(N):
        for j in xrange(N):
            if j>(k-2):
                b[k] += v[j]*C[k,j]

    return b


def A_mat(np.ndarray[np.float64_t, ndim=1] K, 
          np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
          np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
          np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = K.shape[0]-2  
        double pi = np.pi
    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                #c[k,j] = 2*pi*(K[j] + 1)*(K[j] + 2)*b_j[j]
                c[k,j] = 2*pi*(j + 1)*(j + 2)*b_j[j]
            if j>k:
                if (j-k)%2==0:
                    #c[k,j] = (np.pi/2)*( K[j]*(K[j]**2 - K[k]**2) + b_j[j]*(K[j]+2)*((K[j]+2)**2 - K[k]**2) + a_k[k]* (K[j]*(K[j]**2 - (K[k]+1)**2) + b_j[j]*(K[j]+2)*((K[j]+2)**2 - (K[k]+1)**2))+b_k[k]* (K[j]*(K[j]**2 - (K[k]+2)**2) + b_j[j]*(K[j]+2)*((K[j]+2)**2 - (K[k]+2)**2)))
                    c[k,j] = (pi/2.)*( j*(j**2 - k**2) + b_j[j]*(j+2)*((j+2)**2 - k**2) + a_k[k]* (j*(j**2 - (k+1)**2) + b_j[j]*(j+2)*((j+2)**2 - (k+1)**2))+b_k[k]*(j*(j**2 - (k+2)**2) + b_j[j]*(j+2)*((j+2)**2 - (k+2)**2)))

                else:
                    #c[k,j] = (pi/2.0)*(a_j[j]*(K[j]+1)*((K[j]+1)**2 - K[k]**2) +a_k[k]*a_j[j]*(K[j]+1)*((K[j]+1)**2 - (K[k]+1)**2) + b_k[k]*a_j[j]*(K[j]+1)*((K[j]+1)**2 - (K[k]+2)**2))
                    c[k,j] = (pi/2.0)*(a_j[j]*(j+1)*((j+1)**2 - k**2) +a_k[k]*a_j[j]*(j+1)*((j+1)**2 - (k+1)**2) + b_k[k]*a_j[j]*(j+1)*((j+1)**2 - (k+2)**2))
    return c
                             

def A_matvec(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] A,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            A_matvec_1D(K,A, v[:, ii, jj].real, b[:, ii, jj].real)
            A_matvec_1D(K,A, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def A_matvec_1D(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] A,
                np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = K.shape[0]-2

    for k in xrange(N):
        for j in xrange(N):
            if j>(k-1):
                b[k] += v[j]*A[k,j]       

    return b


def D_matvec(np.ndarray[np.float64_t, ndim=1] cj, 
             np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            D_matvec_1D(cj,a_j,b_j, v[:, ii, jj].real, b[:, ii, jj].real)
            D_matvec_1D(cj,a_j,b_j, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def D_matvec_1D(np.ndarray[np.float64_t, ndim=1] cj,
                np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
                np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = b.shape[0]-2

    for i in xrange(N-2):
        b[i] = (np.pi/2.0)*(v[i]*cj[i] + v[i+1]*a_j[i] + v[i+2]*b_j[i])       
    b[N-2] = (np.pi/2.0)*(v[N-2]*cj[N-2] + v[N-1]*a_j[N-2])
    b[N-1] = (np.pi/2.0)*v[N-1]*cj[N-1]
    return b
  
  
def UTDMA(np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
          np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            UTDMA_1D(a_k,b_k, v[:, ii, jj].real, b[:, ii, jj].real)
            UTDMA_1D(a_k,b_k, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def UTDMA_1D(np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
             np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):
    
    cdef:
        int i
        int N = v.shape[0]-2
        double pi = np.pi

    b[N-1] = v[N-1]*(2.0/pi)
    b[N-2] = v[N-2]*(2.0/pi) - a_k[N-2]*b[N-1]
    for i in xrange(N-3,-1,-1):
        b[i] = v[i]*(2.0/pi) - b[i+1]*a_k[i] - b[i+2]*b_k[i] 
    b[0] /= 2.0

    return b


def PDMA(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b,
         np.ndarray[np.float64_t, ndim=1] c, np.ndarray[np.float64_t, ndim=1] d,
         np.ndarray[np.float64_t, ndim=1] e, np.ndarray[T, ndim=3] r, np.ndarray[T, ndim=3] x):

    cdef:
        unsigned int ii, jj
        
    for ii in range(r.shape[1]):
        for jj in range(r.shape[2]):
            PDMA_1D(a,b,c,d,e, r[:, ii, jj].real, x[:, ii, jj].real)
            PDMA_1D(a,b,c,d,e, r[:, ii, jj].imag, x[:, ii, jj].imag)
    return x

def PDMA_1D(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b,
            np.ndarray[np.float64_t, ndim=1] c, np.ndarray[np.float64_t, ndim=1] d,
            np.ndarray[np.float64_t, ndim=1] e, np.ndarray[np.float64_t, ndim=1] r, np.ndarray[np.float64_t, ndim=1] x):
    
    cdef:
        int i
        int N = r.shape[0]-2 
        np.ndarray[np.float64_t, ndim=1] rho
        np.ndarray[np.float64_t, ndim=1] alpha
        np.ndarray[np.float64_t, ndim=1] beta


    rho = np.empty(N)
    alpha = np.zeros(N-1)
    beta = np.zeros(N-2)

    alpha[0] = d[0]/c[0]
    beta[0] = e[0]/c[0]
    
    alpha[1] = (d[1]-b[1]*beta[0])/(c[1]-b[1]*alpha[0])
    beta[1] = e[1]/(c[1]-b[1]*alpha[0])
    
    
    rho[0] = r[0]/c[0]
    rho[1] = (r[1] - b[1]*rho[0])/(c[1]-b[1]*alpha[0])

    for i in xrange(2,N):
        rho[i] = (r[i] - a[i]*rho[i-2] - rho[i-1]*(b[i] - a[i]*alpha[i-2])) / (c[i] - a[i]*beta[i-2] - alpha[i-1]*(b[i] - a[i]*alpha[i-2]))
        if i <= (N-2):
            alpha[i] = (d[i] - beta[i-1]*(b[i] - a[i]*alpha[i-2])) / (c[i] - a[i]*beta[i-2] - alpha[i-1]*(b[i] - a[i]*alpha[i-2]))
        if i <= (N-3):
            beta[i] = e[i]/(c[i] - a[i]*beta[i-2] - alpha[i-1]*(b[i] - a[i]*alpha[i-2]))

    for i in xrange(N-1,-1,-1):
        x[i] = rho[i]
        if i<=(N-2):
            x[i] -= alpha[i]*x[i+1]
        if i<=(N-3):
            x[i] -= beta[i]*x[i+2]    
    return x



def Helmholtz_AB_vector(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] A, 
                        np.ndarray[np.float64_t, ndim=2] B,
                        np.ndarray[np.float64_t, ndim=2] alpha,
                        np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b): 

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Helmholtz_AB_1D(K, A, B, alpha[ii, jj], v[:, ii, jj].real, b[:, ii, jj].real)
            Helmholtz_AB_1D(K, A, B, alpha[ii, jj], v[:, ii, jj].imag, b[:, ii, jj].imag)

    return b

def Helmholtz_AB_1D(np.ndarray[np.float64_t, ndim=1] K, 
                    np.ndarray[np.float64_t, ndim=2] Amat, np.ndarray[np.float64_t, ndim=2] Bmat,
                    np.float64_t alpha, np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i, j
        int N = K.shape[0]-2 

    for i in xrange(N):
        for j in xrange(N):
            if j>=i:
                b[i] += -Amat[i,j]*v[j]
                if (j-2)<=i:
                    b[i] += alpha*Bmat[i,j]*v[j]
            if (j+2)==i or (j+1)==i:
                b[i] += alpha*Bmat[i,j]*v[j]

    return b

def Helmholtz_CB_vector(np.ndarray[np.float64_t, ndim=1] K,np.ndarray[np.float64_t, ndim=2] C, 
                        np.ndarray[np.float64_t, ndim=2] B,
                        np.ndarray[np.float64_t, ndim=2] m, np.ndarray[np.float64_t, ndim=2] n,
                        np.ndarray[np.complex128_t, ndim=3] u_hat,
                        np.ndarray[np.complex128_t, ndim=3] v_hat,
                        np.ndarray[np.complex128_t, ndim=3] w_hat,
                        np.ndarray[np.complex128_t, ndim=3] b):                          

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Helmholtz_CB_1D_vector(K, C, B, m[ii, jj],n[ii, jj],u_hat[:, ii, jj].real, -v_hat[:, ii, jj].imag, -w_hat[:, ii, jj].imag, b[:, ii, jj].real)
            Helmholtz_CB_1D_vector(K, C, B, m[ii, jj],n[ii, jj],u_hat[:, ii, jj].imag,v_hat[:, ii, jj].real, w_hat[:, ii, jj].real, b[:, ii, jj].imag)

    return b

def Helmholtz_CB_1D_vector(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] C, 
                        np.ndarray[np.float64_t, ndim=2] B,
                        np.float64_t m, np.float64_t n,
                        np.ndarray[np.float64_t, ndim=1] u_hat,
                        np.ndarray[np.float64_t, ndim=1] v_hat,
                        np.ndarray[np.float64_t, ndim=1] w_hat,
                        np.ndarray[np.float64_t, ndim=1] b):    
                            
    cdef:
        int k, j
        int N = K.shape[0]-2 
    
    for k in xrange(N):
        for j in xrange(N):
            if (j+1)>=k:
                b[k] += C[k,j]*u_hat[j]
                if (j-2)<=k:
                    b[k] += m*B[k,j]*v_hat[j] + n*B[k,j]*w_hat[j]
            elif (j+2)==k:
                b[k] += m*B[k,j]*v_hat[j] + n*B[k,j]*w_hat[j]

    return b

def Helmholtz_AB_Solver(np.ndarray[np.float64_t, ndim=1] K,
                        np.ndarray[np.float64_t, ndim=2] alpha, np.int64_t Neu, np.ndarray[T, ndim=3] v,
                        np.ndarray[np.float64_t, ndim=2] A, np.ndarray[np.float64_t, ndim=2] B,
                        np.ndarray[T, ndim=3] b): 


    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Helmholtz_AB_1D_Solver(K, alpha[ii, jj], Neu, v[:, ii, jj].real, A, B, b[:, ii, jj].real)
            Helmholtz_AB_1D_Solver(K, alpha[ii, jj], Neu, v[:, ii, jj].imag, A, B, b[:, ii, jj].imag)

    return b

def Helmholtz_AB_1D_Solver(np.ndarray[np.float64_t, ndim=1] K, np.float64_t alpha,
                           np.int64_t Neu, np.ndarray[np.float64_t, ndim=1] v, 
                           np.ndarray[np.float64_t, ndim=2] A, np.ndarray[np.float64_t, ndim=2] B, 
                           np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i, j, k, index1, index2
        np.float64_t s, maxU
        int N = K.shape[0]-2
        np.ndarray[np.int64_t, ndim=1] pivot
        np.ndarray[np.float64_t, ndim=1] r
        np.ndarray[np.float64_t, ndim=1] y
        np.ndarray[np.float64_t, ndim=2] U
        np.ndarray[np.float64_t, ndim=2] L
    
    U = np.zeros((N,N))
    L = np.zeros((N,N))
    r = np.zeros(N)
    y = np.zeros(N)
    pivot = np.empty(N, dtype = int)
    
    # Make L and the pivot matrix  
    for i in xrange(N):
        pivot[i] = i
        for j in xrange(N): 
            U[i,j] = -A[i,j] + alpha*B[i,j]
            if i == j:
                L[i,j] = 1.0

    # LU-decomposition
    for j in xrange(Neu, N-1):
        if j == N-2:
            maxU = np.abs(U[j,j])
            if np.abs(U[j+1,j])>= maxU:
                maxU = np.abs(U[j+1,j])
                pivot[[j,j+1]] = pivot[[j+1,j]]
                U[[pivot[j], pivot[j+1]],j:N] = U[[pivot[j+1], pivot[j]],j:N]
                L[[pivot[j], pivot[j+1]],:] = L[[pivot[j+1], pivot[j]],:]            

            L[j+1,j] = U[j+1,j]/U[j,j]
            U[j+1,j] = U[j+1,j] - L[j+1,j]*U[j,j]
            U[j+1,j+1] = U[j+1,j+1] - L[j+1,j]*U[j,j+1]           
        else:  
            index1 = j
            index2 = j        
            maxU = np.abs(U[j,j])
            for k in xrange(j+1,j+3):
                if np.abs(U[k,j])>= maxU:
                    maxU = np.abs(U[k,j])
                    index2 = k

            if index1 != index2:        
                U[[index1, index2],j:N] = U[[index2, index1],j:N]
                L[[index1, index2],:] = L[[index2, index1],:]
                pivot[[index1,index2]] = pivot[[index2,index1]]   

            for k in xrange(j+1, j+3):
                L[k,j] = U[k,j]/U[j,j]
                for i in xrange(j, N): 
                    U[k,i] = U[k,i] - L[k,j]*U[j,i] 
    # Compute Pv
    for i in range(N):
        r[i] = v[pivot[i]]

    # L - Forward substitution
    y[0] = r[0]
    y[1] = r[1] - L[1,0]*y[0]

    for i in xrange(2,N):
        s = 0.0
        for k in xrange((i-2), i):
            s += L[i,k]*y[k]
        y[i] = r[i] - s
    # U - Backward substitution
    b[N-1] = y[N-1]/U[N-1,N-1]
    for i in xrange(N-2,Neu-1,-1):
        s = 0.0
        for j in xrange(i+1,N):
            s += U[i,j]*b[j]
        b[i] = (y[i]-s) / U[i,i]    


    return b
