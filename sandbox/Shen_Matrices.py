# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:48:17 2015

@author: Diako Darian
"""
from numpy import *
from SFTc import C_matvec, B_matvec, A_matvec, C_mat, B_mat, A_mat, UTDMA_1D, UTDMA
from scipy.sparse import diags

class B_matrix(object):
    """Shen mass matrix for inner product \sum_j (phi_j, phi_k)_w  u_j = B * u_hat
    where u_hat is a vector of coefficients for a general Shen transform
    and phi_k is a general Shen basis function.
    """
    def __init__(self, K, quad, a_j, b_j, a_k, b_k, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        ck = ones(K.shape)
        N = self.shape[0]
        if quad == "GL":
            ck[0] = 2
        elif quad == "GC":
            ck[0] = 2; ck[-3] = 2  
        self.dd = pi/2*(ck[:-2] + a_j*a_k + b_j*b_k)
        self.ud = pi/2*(a_k[:N-1] + a_j[1:]*b_k[:N-1])
        self.uud = pi*b_k[:(N-2)]/2.0
        self.ld = pi/2*(a_j[:N-1] + b_j[:N-1]*a_k[1:])
        self.lld = pi*b_j[:(N-2)]/2.0
        

    def mat(self, c):
        return B_mat(self.K, self.ck, self.a_j, self.b_j, self.a_k, self.b_k, c)
        
    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=complex)
        if len(v.shape) > 1:
            B_matvec1(self.uud, self.ud, self.lld, self.ld, self.dd, v, c)
        else:
	    c[:(N-1)]  = self.uud*v[2:]
            c[:N]     += self.ud*v[1:]
            c[:]      += self.dd*v[:]
            c[1:]     += self.ld*v[:N]
            c[2:]     += self.lld*v[:(N-1)]
        return c
    
    def diags(self):
        #return diags([self.lld, self.ld, self.dd, self.ud, self.uud], [-2, -1, 0, 1, 2], shape=self.shape)
        return self.lld, self.ld, self.dd, self.ud, self.uud

    
class C_matrix(object):
    """Shen gradient matrix for inner product (u', phi_k) = (phi_j', phi_k) u_hat = C * u_hat
    where u_hat is a vector of coefficients for a general Shen transform
    and phi_j and phi_k are general Shen basis functions.
    """

    def __init__(self, K, quad, a_j, b_j, a_k, b_k, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        ck = ones(K.shape)
        if quad == "GL":
            ck[0] = 2
        elif quad == "GC":
            ck[0] = 2; ck[-1] = 2   

    def mat(self, c):
        C_mat(self.K, self.ck, self.a_j, self.b_j, self.a_k, self.b_k, c)
        return c 
    
    def matvec(self, v):
        c = zeros(v.shape, dtype=complex)
        if len(v.shape) > 1:
            C_matvec(self.K, self.ck, self.a_j, self.b_j, self.a_k, self.b_k, v, c)
        else:
	    b = zeros(self.shape)
	    self.mat(b)
	    c = dot(b,v)
        return c



class A_matrix(object):    
    """Shen stiffness matrix for inner product (u'', phi_k) = (phi_j'', phi_k) u_hat = A * u_hat
    where u_hat is a vector of coefficients for a general Shen transform
    and phi_j and phi_k are general Shen basis functions.
    """

    def __init__(self, K, quad, a_j, b_j, a_k, b_k, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        ck = ones(K.shape)
        if quad == "GL":
            ck[0] = 2
        elif quad == "GC":
            ck[0] = 2; ck[-1] = 2   

    def mat(self, c):
        A_mat(self.K, self.ck, self.a_j, self.b_j, self.a_k, self.b_k, c)
        return c 
    
    def matvec(self, v):
        c = zeros(v.shape, dtype=complex)
        if len(v.shape) > 1:
	    A_matvec(self.K, self.ck, self.a_j, self.b_j, self.a_k, self.b_k, v, c)
        else:
            b = zeros(self.shape)
	    self.mat(b)
	    c = dot(b,v)
        return c

        
class D_matrix(object):    
    """Chebyshev-Shen matrix for the linear system (T_j, phi_k) u_hat = D * u_hat = v_hat 
    where u_hat is a vector of coefficients for a general Shen transform
    and T_j and phi_k are Chebyshev and general Shen basis functions, respectively.
    """

    def __init__(self, K, quad, a_k, b_k, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        ck = ones(K.shape)
        N = self.shape[0]
        if quad == "GL":
            ck[0] = 2
        elif quad == "GC":
            ck[0] = 2; ck[-1] = 2   
        self.dd = (pi/2)*ck[:N]
        self.ud = (pi/2)*a_k[:(N-1)]
        self.uud = (pi/2)*b_k[:(N-2)]
    
    # Not finished!!!
    def matvec(self, v):
        c = zeros(v.shape, dtype=complex)
        c = D_matvec(self.dd, self.ud, self.uud, v, c)
    def UTDMA(self, v):
        c = zeros(v.shape, dtype=complex)
        v *= (2./pi) 
        if len(v.shape) > 1:
	    UTDMA(self.a_k, self.b_k, v, c)
        else:
            UTDMA_1D(self.a_k, self.b_k, v, c)
        return c        
        
    def diags(self):
        #return diags([self.dd, self.ud, self.uud], [0, 1, 2], shape=self.shape)
        return self.dd, self.ud, self.uud