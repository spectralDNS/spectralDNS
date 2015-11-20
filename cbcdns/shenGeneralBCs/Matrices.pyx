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
                            
