# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:41:49 2015

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

# -----------------------------------------------------------------------------------------------
#         Minus matrices: Bmm = (phi_j^{minus},    phi_k^{minus})_w
#                         Cmm = (phi_j^{minus}',  phi_k^{minus})_w
#                         Amm = (phi_j^{minus}'', phi_k^{minus})_w
#         -----------------------------o---------------------------------------
#         Bmm_matvec, Cmm_matvec and Amm_matvec give the vector multiplication with matrices
# -----------------------------------------------------------------------------------------------
def Bmm_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[np.float64_t, ndim=2] B):

    cdef:
        int k
        int N = b_k.shape[0]
        double pi = np.pi
  
    for k in xrange(N):
        B[k,k] = (pi/2.)*(b_k[k]**2)
    return B

def Bmm_matvec(np.ndarray[np.float64_t, ndim=2] B,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Bmm_matvec_1D(B, v[:, ii, jj].real, b[:, ii, jj].real)
            Bmm_matvec_1D(B, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Bmm_matvec_1D(np.ndarray[np.float64_t, ndim=2] B,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = v.shape[0]-2

    for i in xrange(N):
        b[i] = v[i]*B[i,i]

    return b

def Bmm_inv(np.ndarray[np.float64_t, ndim=2] Bmat,
            np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Bmm_inv_1D(Bmat, v[:, ii, jj].real, b[:, ii, jj].real)
            Bmm_inv_1D(Bmat, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Bmm_inv_1D(np.ndarray[np.float64_t, ndim=2] Bmat,
               np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = v.shape[0]-2

    for i in xrange(N):
        b[i] = v[i]/Bmat[i,i]

    return b


def Cmm_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = b_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        for j in xrange(N):
            if j>k:
                if (j-k)%2 != 0:
                    c[k,j] = pi*(j+2)*b_k[k]*b_k[j]                 
    return c


def Cmm_matvec(np.ndarray[np.float64_t, ndim=2] C,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Cmm_matvec_1D(C, v[:, ii, jj].real, b[:, ii, jj].real)
            Cmm_matvec_1D(C, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Cmm_matvec_1D(np.ndarray[np.float64_t, ndim=2] C,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        for j in xrange((k+1), N, 2):
            b[k] += v[j]*C[k,j]

    return b


def Amm_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[T, ndim=2] A):

    cdef:
        int j, k
        int N = b_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        for j in xrange((k+2), N, 2):
            A[k,j] = 0.5*pi*(j+2)*((j+2)**2 - (k+2)**2)*b_k[k]*b_k[j]                 
    return A


def Amm_matvec(np.ndarray[np.float64_t, ndim=2] A,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Amm_matvec_1D(A, v[:, ii, jj].real, b[:, ii, jj].real)
            Amm_matvec_1D(A, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Amm_matvec_1D(np.ndarray[np.float64_t, ndim=2] A,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        for j in xrange((k+2), N, 2):
            b[k] += v[j]*A[k,j]

    return b

# -----------------------------------------------------------------------------------------------
#         Matrices: Bpm = (phi_j^{plus},   phi_k^{minus})_w
#                   Cpm = (phi_j^{plus}',  phi_k^{minus})_w
#                   Apm = (phi_j^{plus}'', phi_k^{minus})_w
#         -----------------------------o---------------------------------------
#         Bpm_matvec, Cpm_matvec and Apm_matvec give the vector multiplication with matrices
# -----------------------------------------------------------------------------------------------
def Bpm_mat(np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[np.float64_t, ndim=2] B):

    cdef:
        int k, j
        int N = bm_k.shape[0]
        double pi = np.pi
  
    for k in xrange(N):
        for j in xrange(N):
            if j==k:
                B[k,j] = (pi/2.)*(bm_k[k]*bp_k[j])
            elif k == (j-2):
                B[k,j] = pi*bm_k[k]
    return B

def Bpm_matvec(np.ndarray[np.float64_t, ndim=2] B,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Bpm_matvec_1D(B, v[:, ii, jj].real, b[:, ii, jj].real)
            Bpm_matvec_1D(B, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Bpm_matvec_1D(np.ndarray[np.float64_t, ndim=2] B,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = v.shape[0]-2

    for i in xrange(N):
        b[i] = v[i]*B[i,i] + v[i+2]*B[i,(i+2)]

    return b


def Cpm_mat(np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = bm_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        for j in xrange((k+1), N, 2):
            if j==(k+1):
                c[k,j] = pi*(j+2)*bp_k[j]*bm_k[k]
            else:
                c[k,j] = pi*bm_k[k]*(2.*j + (j+2.)*bp_k[j])                
    return c


def Cpm_matvec(np.ndarray[np.float64_t, ndim=2] C,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Cpm_matvec_1D(C, v[:, ii, jj].real, b[:, ii, jj].real)
            Cpm_matvec_1D(C, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Cpm_matvec_1D(np.ndarray[np.float64_t, ndim=2] C,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        for j in xrange((k+1), N, 2):
            b[k] += v[j]*C[k,j]

    return b


def Apm_mat(np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[T, ndim=2] A):

    cdef:
        int j, k
        int N = bm_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        for j in xrange((k+2), N, 2):
            if j==(k+2):
                A[k,j] = 0.5*pi*(j+2)*((j+2)**2 - (k+2)**2)*bp_k[j]*bm_k[k]          
            else:
                A[k,j] = 0.5*pi*bm_k[k]*( 2.*j*(j**2 - (k+2)**2) + bp_k[j]*(j+2)*((j+2)**2 - (k+2)**2) )
    return A


def Apm_matvec(np.ndarray[np.float64_t, ndim=2] A,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Apm_matvec_1D(A, v[:, ii, jj].real, b[:, ii, jj].real)
            Apm_matvec_1D(A, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Apm_matvec_1D(np.ndarray[np.float64_t, ndim=2] A,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        for j in xrange((k+2), N, 2):
            b[k] += v[j]*A[k,j]

    return b

# -----------------------------------------------------------------------------------------------
#         Matrices: Bbm = (phi_j^{breve},   phi_k^{minus})_w
#                   Cbm = (phi_j^{breve}',  phi_k^{minus})_w
#                   Abm = (phi_j^{breve}'', phi_k^{minus})_w
#         -----------------------------o---------------------------------------
#         Bbm_matvec, Cbm_matvec and Abm_matvec give the vector multiplication with matrices
# -----------------------------------------------------------------------------------------------
def Bbm_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[np.float64_t, ndim=2] B):

    cdef:
        int k, j
        int N = bm_k.shape[0]
        double pi = np.pi
  
    for k in xrange(N):
        for j in xrange(N):
            if j==k:
                B[k,j] = (pi/2.)*(bm_k[k]*b_k[j])
            elif k == (j-2):
                B[k,j] = (pi/2.)*bm_k[k]
    return B

def Bbm_matvec(np.ndarray[np.float64_t, ndim=2] B,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Bbm_matvec_1D(B, v[:, ii, jj].real, b[:, ii, jj].real)
            Bbm_matvec_1D(B, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Bbm_matvec_1D(np.ndarray[np.float64_t, ndim=2] B,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = v.shape[0]-2

    for i in xrange(N):
        b[i] = v[i]*B[i,i] + v[i+2]*B[i,(i+2)]

    return b


def Cbm_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = b_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        for j in xrange((k+1), N, 2):
            if j==(k+1):
                c[k,j] = pi*(j+2)*b_k[j]*bm_k[k]
            else:
                c[k,j] = pi*bm_k[k]*(j + (j+2.)*b_k[j])                
    return c


def Cbm_matvec(np.ndarray[np.float64_t, ndim=2] C,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Cbm_matvec_1D(C, v[:, ii, jj].real, b[:, ii, jj].real)
            Cbm_matvec_1D(C, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Cbm_matvec_1D(np.ndarray[np.float64_t, ndim=2] C,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        for j in xrange((k+1), N, 2):
            b[k] += v[j]*C[k,j]

    return b


def Abm_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[T, ndim=2] A):

    cdef:
        int j, k
        int N = b_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        for j in xrange((k+2), N, 2):
            if j==(k+2):
                A[k,j] = 0.5*pi*(j+2)*((j+2)**2 - (k+2)**2)*b_k[j]*bm_k[k]          
            else:
                A[k,j] = 0.5*pi*bm_k[k]*( j*(j**2 - (k+2)**2) + b_k[j]*(j+2)*((j+2)**2 - (k+2)**2) )
    return A


def Abm_matvec(np.ndarray[np.float64_t, ndim=2] A,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Abm_matvec_1D(A, v[:, ii, jj].real, b[:, ii, jj].real)
            Abm_matvec_1D(A, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Abm_matvec_1D(np.ndarray[np.float64_t, ndim=2] A,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        for j in xrange((k+2), N, 2):
            b[k] += v[j]*A[k,j]

    return b


# -----------------------------------------------------------------------------------------------
#         Matrices: Bmb = (phi_k^{minus},   phi_j^{breve})_w
#                   Cmb = (phi_k^{minus}',  phi_j^{breve})_w
#                   Amb = (phi_k^{minus}'', phi_j^{breve})_w
#         -----------------------------o---------------------------------------
#         Bmb_matvec, Cmb_matvec and Amb_matvec give the vector multiplication with matrices
# -----------------------------------------------------------------------------------------------
def Bmb_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[np.float64_t, ndim=2] B):

    cdef:
        int k, j
        int N = bm_k.shape[0]
        double pi = np.pi
  
    for k in xrange(N):
        for j in xrange(N):
            if j==k:
                B[k,j] = (pi/2.)*(bm_k[j]*b_k[k])
            elif k == (j-2):
                B[k,j] = (pi/2.)*bm_k[j]
    return B

def Bmb_matvec(np.ndarray[np.float64_t, ndim=2] B,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Bmb_matvec_1D(B, v[:, ii, jj].real, b[:, ii, jj].real)
            Bmb_matvec_1D(B, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Bmb_matvec_1D(np.ndarray[np.float64_t, ndim=2] B,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = v.shape[0]-2

    for i in xrange(N):
        b[i] = v[i]*B[i,i] + v[i+2]*B[i,(i+2)]

    return b


def Cmb_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = b_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        if k>=1:
            c[k,(k-1)] = pi*(k+1)*bm_k[k-1]
        for j in xrange((k+1), N, 2):
            c[k,j] = pi*(j+2)*bm_k[j]*(1. + b_k[k])                
    return c


def Cmb_matvec(np.ndarray[np.float64_t, ndim=2] C,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Cmb_matvec_1D(C, v[:, ii, jj].real, b[:, ii, jj].real)
            Cmb_matvec_1D(C, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Cmb_matvec_1D(np.ndarray[np.float64_t, ndim=2] C,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        if k>=1:
            b[k] += C[k,(k-1)]*v[k-1]
        for j in xrange((k+1), N, 2):
            b[k] += v[j]*C[k,j]

    return b


def Amb_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[T, ndim=2] A):

    cdef:
        int j, k
        int N = b_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        for j in xrange(k, N, 2):
            if j==k:
                A[k,j] = 0.5*pi*(j+2)*bm_k[j]*((j+2)**2 - k**2)          
            else:
                A[k,j] = 0.5*pi*(j+2)*bm_k[j]*( (j+2)**2 - k**2 + b_k[k]*((j+2)**2 - (k+2)**2) )
    return A


def Amb_matvec(np.ndarray[np.float64_t, ndim=2] A,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Amb_matvec_1D(A, v[:, ii, jj].real, b[:, ii, jj].real)
            Amb_matvec_1D(A, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Amb_matvec_1D(np.ndarray[np.float64_t, ndim=2] A,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        for j in xrange(k, N, 2):
            b[k] += v[j]*A[k,j]

    return b


# -----------------------------------------------------------------------------------------------
#         Matrices: Bbb = (phi_k^{breve},   phi_j^{breve})_w
#                   Cbb = (phi_k^{breve}',  phi_j^{breve})_w
#                   Abb = (phi_k^{breve}'', phi_j^{breve})_w
#         -----------------------------o---------------------------------------
#         Bbb_matvec, Cbb_matvec and Abb_matvec give the vector multiplication with matrices
# -----------------------------------------------------------------------------------------------
def Bbb_mat(np.ndarray[np.float64_t, ndim=1] c_k, np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[np.float64_t, ndim=2] B):

    cdef:
        int k, j
        int N = b_k.shape[0]
        double pi = np.pi
  
    for k in xrange(N):
        for j in xrange(N):
            if j==k:
                B[k,j] = (pi/2.)*(c_k[j] + b_k[j]*b_k[k])
            elif k == (j-2):
                B[k,j] = (pi/2.)*b_k[k]
            elif k == (j+2):
                B[k,j] = (pi/2.)*b_k[j]
    return B

def Bbb_matvec(np.ndarray[np.float64_t, ndim=2] B,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Bbb_matvec_1D(B, v[:, ii, jj].real, b[:, ii, jj].real)
            Bbb_matvec_1D(B, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Bbb_matvec_1D(np.ndarray[np.float64_t, ndim=2] B,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = v.shape[0]-2

    for i in xrange(N):
        b[i] = v[i]*B[i,i] + v[i+2]*B[i,(i+2)]
        if i >=2:
            b[i] += v[(i-2)]*B[i,(i-2)]
    return b


def Cbb_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = b_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        if k>=1:
            c[k,(k-1)] = pi*(k+1)*b_k[k-1]
        for j in xrange((k+1), N, 2):
            if j ==(k+1):
                c[k,j] = pi*(j + b_k[j]*(j+2) + b_k[j]*b_k[k]*(j+2))
            else:
                c[k,j] = pi*(j + j*b_k[k] + b_k[j]*(j+2) + b_k[j]*b_k[k]*(j+2))               
    return c


def Cbb_matvec(np.ndarray[np.float64_t, ndim=2] C,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Cbb_matvec_1D(C, v[:, ii, jj].real, b[:, ii, jj].real)
            Cbb_matvec_1D(C, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Cbb_matvec_1D(np.ndarray[np.float64_t, ndim=2] C,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        if k>=1:
            b[k] += C[k,(k-1)]*v[k-1]
        for j in xrange((k+1), N, 2):
            b[k] += v[j]*C[k,j]

    return b


def Abb_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[T, ndim=2] A):

    cdef:
        int j, k
        int N = b_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        for j in xrange(k, N, 2):
            if j==k:
                A[k,j] = 2*pi*(j+1)*(j+2)*b_k[j]          
            else:
                A[k,j] = 0.5*pi*( j*(j**2-k**2) + b_k[k]*j*(j**2 - (k+2)**2) + (j+2)*b_k[j]*( (j+2)**2 - k**2 + b_k[k]*((j+2)**2 - (k+2)**2) ))
    return A


def Abb_matvec(np.ndarray[np.float64_t, ndim=2] A,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Abb_matvec_1D(A, v[:, ii, jj].real, b[:, ii, jj].real)
            Abb_matvec_1D(A, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Abb_matvec_1D(np.ndarray[np.float64_t, ndim=2] A,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        for j in xrange(k, N, 2):
            b[k] += v[j]*A[k,j]

    return b


# -----------------------------------------------------------------------------------------------
#         Matrices: Bmp = (phi_k^{minus},   phi_j^{plus})_w
#                   Cmp = (phi_k^{minus}',  phi_j^{plus})_w
#                   Amp = (phi_k^{minus}'', phi_j^{plus})_w
#         -----------------------------o---------------------------------------
#         Bmp_matvec, Cmp_matvec and Amp_matvec give the vector multiplication with matrices
# -----------------------------------------------------------------------------------------------
def Bmp_mat(np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[np.float64_t, ndim=2] B):

    cdef:
        int k, j
        int N = bm_k.shape[0]
        double pi = np.pi
  
    for k in xrange(N):
        for j in xrange(N):
            if j==k:
                B[k,j] = (pi/2.)*(bm_k[j]*bp_k[k])
            elif j == (k-2):
                B[k,j] = pi*bm_k[j]
    return B

def Bmp_matvec(np.ndarray[np.float64_t, ndim=2] B,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Bmp_matvec_1D(B, v[:, ii, jj].real, b[:, ii, jj].real)
            Bmp_matvec_1D(B, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Bmp_matvec_1D(np.ndarray[np.float64_t, ndim=2] B,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = v.shape[0]-2

    for i in xrange(N):
        b[i] = v[i]*B[i,i]
        if i >=2:
            b[i] += v[(i-2)]*B[i,(i-2)]
    return b


def Cmp_mat(np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = bp_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        if k>=1:
            c[k,(k-1)] = 2*pi*(k+1)*bm_k[k-1]
        for j in xrange((k+1), N, 2):
            c[k,j] = pi*(2.*bm_k[j]*(j+2) + bm_k[j]*bp_k[k]*(j+2))               
    return c


def Cmp_matvec(np.ndarray[np.float64_t, ndim=2] C,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Cmp_matvec_1D(C, v[:, ii, jj].real, b[:, ii, jj].real)
            Cmp_matvec_1D(C, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Cmp_matvec_1D(np.ndarray[np.float64_t, ndim=2] C,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        if k>=1:
            b[k] += C[k,(k-1)]*v[k-1]
        for j in xrange((k+1), N, 2):
            b[k] += v[j]*C[k,j]

    return b


def Amp_mat(np.ndarray[np.float64_t, ndim=1] bm_k, np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[T, ndim=2] A):

    cdef:
        int j, k
        int N = bp_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        for j in xrange(k, N, 2):
            if j==k:
                A[k,j] = 4*pi*(j+1)*(j+2)*bm_k[j]          
            else:
                A[k,j] = pi*bm_k[j]*(j+2)*( (j+2)**2 - k**2 + 0.5*bp_k[k]*( (j+2)**2 - (k+2)**2) )
    return A


def Amp_matvec(np.ndarray[np.float64_t, ndim=2] A,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Amp_matvec_1D(A, v[:, ii, jj].real, b[:, ii, jj].real)
            Amp_matvec_1D(A, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Amp_matvec_1D(np.ndarray[np.float64_t, ndim=2] A,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        for j in xrange(k, N, 2):
            b[k] += v[j]*A[k,j]

    return b

# -----------------------------------------------------------------------------------------------
#         Matrices: Bpp = (phi_k^{plus},   phi_j^{plus})_w
#                   Cpp = (phi_k^{plus}',  phi_j^{plus})_w
#                   App = (phi_k^{plus}'', phi_j^{plus})_w
#         -----------------------------o---------------------------------------
#         Bpp_matvec, Cpp_matvec and App_matvec give the vector multiplication with matrices
# -----------------------------------------------------------------------------------------------
def Bpp_mat(np.ndarray[np.float64_t, ndim=1] ck, np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[np.float64_t, ndim=2] B):

    cdef:
        int k, j
        int N = bp_k.shape[0]
        double pi = np.pi
  
    for k in xrange(N):
        for j in xrange(N):
            if j==k:
                B[k,j] = (pi/2.)*(4.*ck[j] + bp_k[j]*bp_k[k])
            elif j == (k-2):
                B[k,j] = pi*bp_k[j]
            elif j == (k+2):
                B[k,j] = pi*bp_k[k]
    return B

def Bpp_matvec(np.ndarray[np.float64_t, ndim=2] B,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Bpp_matvec_1D(B, v[:, ii, jj].real, b[:, ii, jj].real)
            Bpp_matvec_1D(B, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Bpp_matvec_1D(np.ndarray[np.float64_t, ndim=2] B,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = v.shape[0]-2

    for i in xrange(N):
        b[i] = v[i]*B[i,i] + v[i+2]*B[i,(i+2)]
        if i >=2:
            b[i] += v[(i-2)]*B[i,(i-2)]
    return b


def Cpp_mat(np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = bp_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        if k>=1:
            c[k,(k-1)] = 2*pi*(k+1)*bp_k[k-1]
        for j in xrange((k+1), N, 2):
            if j == (k+1):
                c[k,j] = pi*(4.*j + 2.*bp_k[j]*(j+2) + bp_k[k]*bp_k[j]*(j+2))
            else:  
                c[k,j] = pi*(4.*j + 2.*bp_k[k]*j + 2.*bp_k[j]*(j+2) + bp_k[k]*bp_k[j]*(j+2))               
    return c


def Cpp_matvec(np.ndarray[np.float64_t, ndim=2] C,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Cpp_matvec_1D(C, v[:, ii, jj].real, b[:, ii, jj].real)
            Cpp_matvec_1D(C, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Cpp_matvec_1D(np.ndarray[np.float64_t, ndim=2] C,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        if k>=1:
            b[k] += C[k,(k-1)]*v[k-1]
        for j in xrange((k+1), N, 2):
            b[k] += v[j]*C[k,j]

    return b


def App_mat(np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[T, ndim=2] A):

    cdef:
        int j, k
        int N = bp_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        for j in xrange(k, N, 2):
            if j==k:
                A[k,j] = 4*pi*(j+1)*(j+2)*bp_k[j]          
            else:
                A[k,j] = (pi/2.)*( 4.*j*(j**2 - k**2) + 2.*bp_k[k]*j*(j**2 - (k+2)**2) + bp_k[j]*(j+2)*( 2.*(j+2)**2 - 2.*k**2 + bp_k[k]*( (j+2)**2 - (k+2)**2) ))
    return A


def App_matvec(np.ndarray[np.float64_t, ndim=2] A,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            App_matvec_1D(A, v[:, ii, jj].real, b[:, ii, jj].real)
            App_matvec_1D(A, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def App_matvec_1D(np.ndarray[np.float64_t, ndim=2] A,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        for j in xrange(k, N, 2):
            b[k] += v[j]*A[k,j]

    return b

# -----------------------------------------------------------------------------------------------
#         Matrices: Bbp = (phi_k^{breve},   phi_j^{plus})_w
#                   Cbp = (phi_k^{breve}',  phi_j^{plus})_w
#                   Abp = (phi_k^{breve}'', phi_j^{plus})_w
#         -----------------------------o---------------------------------------
#         Bbp_matvec, Cbp_matvec and Abp_matvec give the vector multiplication with matrices
# -----------------------------------------------------------------------------------------------
def Bbp_mat(np.ndarray[np.float64_t, ndim=1] ck, np.ndarray[np.float64_t, ndim=1] b_k , np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[np.float64_t, ndim=2] B):

    cdef:
        int k, j
        int N = b_k.shape[0]
        double pi = np.pi
  
    for k in xrange(N):
        for j in xrange(N):
            if j==k:
                B[k,j] = (pi/2.)*(2.*ck[j] + b_k[j]*bp_k[k])
            elif j == (k-2):
                B[k,j] = pi*b_k[j]
            elif j == (k+2):
                B[k,j] = pi*bp_k[k]
    return B

def Bbp_matvec(np.ndarray[np.float64_t, ndim=2] B,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Bbp_matvec_1D(B, v[:, ii, jj].real, b[:, ii, jj].real)
            Bbp_matvec_1D(B, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Bbp_matvec_1D(np.ndarray[np.float64_t, ndim=2] B,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = v.shape[0]-2

    for i in xrange(N):
        b[i] = v[i]*B[i,i] + v[i+2]*B[i,(i+2)]
        if i >=2:
            b[i] += v[(i-2)]*B[i,(i-2)]
    return b


def Cbp_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = b_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        if k>=1:
            c[k,(k-1)] = pi*(k+1)*b_k[k-1]
        for j in xrange((k+1), N, 2):
            if j == (k+1):
                c[k,j] = pi*(2.*j + b_k[j]*(j+2) + bp_k[k]*b_k[j]*(j+2))
            else:  
                c[k,j] = (pi/2.)*(2*j + 2.*bp_k[k]*j + b_k[j]*(j+2) + bp_k[k]*b_k[j]*(j+2))               
    return c


def Cbp_matvec(np.ndarray[np.float64_t, ndim=2] C,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Cbp_matvec_1D(C, v[:, ii, jj].real, b[:, ii, jj].real)
            Cbp_matvec_1D(C, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Cbp_matvec_1D(np.ndarray[np.float64_t, ndim=2] C,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        if k>=1:
            b[k] += C[k,(k-1)]*v[k-1]
        for j in xrange((k+1), N, 2):
            b[k] += v[j]*C[k,j]

    return b


def Abp_mat(np.ndarray[np.float64_t, ndim=1] b_k, np.ndarray[np.float64_t, ndim=1] bp_k, np.ndarray[T, ndim=2] A):

    cdef:
        int j, k
        int N = b_k.shape[0]
        double pi = np.pi
    
    for k in xrange(N):
        for j in xrange(k, N, 2):
            if j==k:
                A[k,j] = 2.*pi*(j+1)*(j+2)*b_k[j]          
            else:
                A[k,j] = (pi/2.)*( 2.*j*(j**2 - k**2) + 2.*bp_k[k]*j*(j**2 - (k+2)**2) + b_k[j]*(j+2)*( (j+2)**2 - k**2 + bp_k[k]*( (j+2)**2 - (k+2)**2) ))
    return A


def Abp_matvec(np.ndarray[np.float64_t, ndim=2] A,
               np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Abp_matvec_1D(A, v[:, ii, jj].real, b[:, ii, jj].real)
            Abp_matvec_1D(A, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def Abp_matvec_1D(np.ndarray[np.float64_t, ndim=2] A,
                  np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = v.shape[0]-2
    
    for k in xrange(N):
        for j in xrange(k, N, 2):
            b[k] += v[j]*A[k,j]

    return b