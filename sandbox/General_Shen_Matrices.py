# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:25:07 2015

@author: Diako Darian
"""

from numpy import *
from SFTc import C_matvec, B_matvec, A_matvec
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
        if self.quad == "GL":
            ck[0] = 2
        elif self.quad == "GC":
            ck[0] = 2; ck[-1] = 2     
        self.dd = pi/2*(ck + a_j*a_k + b_j*b_k)
        self.ud = pi/2*(a_k[:N] + a_j[1:]*b_k[:N])
        self.uud = pi*b_k[:(N-1)]/2
        self.ld = pi/2*(a_j[:N] + b_j[:N]*a_k[1:])
        self.lld = pi*b_j[:(N-1)]/2

    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=complex)
        if len(v.shape) > 1:
            B_matvec(self.uud, self.ud, self.lld, self.ld, self.dd, v, c)
        else:
	      c[:(N-1)]  = self.uud*v[2:]
            c[:N]     += self.ud*v[1:]
            c[:]      += self.dd*v[:]
            c[1:]     += self.ld*v[:N]
            c[2:]     += self.lld*v[:(N-1)]
        return c
    
    def diags(self):
        return diags([self.lld, self.ld, self.dd, self.ud, self.uud], [-2, -1, 0, 1, 2], shape=self.shape)


    
class C_matrix(object):
    """Shen gradient matrix for inner product (u', phi_k) = (phi_j', phi_k) u_hat = C * u_hat
    where u_hat is a vector of coefficients for a general Shen transform
    and phi_j and phi_k are general Shen basis functions.
    """

    def __init__(self, K, quad, a_j, b_j, a_k, b_k, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        ck = ones(K.shape)
        N = self.shape[0]
        if self.quad == "GL":
            ck[0] = 2
        elif self.quad == "GC":
            ck[0] = 2; ck[-1] = 2   
        self.dd = pi*(a_j*(K + 1) + b_j*a_k*(K + 2))
        self.ud = pi*(K[1:] + a_j[1:]*a_k[:N]*(K[1:]+1) + b_j[1:]*(K[1:]+2) + b_j[1:]*b_k[:N]*(K[1:]+2))
        self.ld = pi*b_j[:N]*(K[:N]+2) 

        self.uud = []
        for i in range(2, N, 2):
            self.uud.append(array( pi*( a_k[:(-i)]*K[i:] + a_j[i:]*(K[i:]+1) + a_j[i:]*b_k[:(-i)]*(K[i:]+1) + b_j[i:]*a_k[:(-i)]*(K[i:]+2))))  
            if (i+1)<N:
		    self.uud.append(array( pi*(K[(i+1):] + b_k[:-(i+1)]*K[(i+1):] + a_j[(i+1):]*a_k[:-(i+1)]*(K[(i+1):]+1) + b_j[(i+1):]*(K[(i+1):]+2) + b_j[(i+1):]*b_k[:-(i+1)]*(K[(i+1):]+2)))) 

    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=complex)
        uud = asarray(self.uud)
        if len(v.shape) > 1:
            C_matvec(self.ld, self.ud, uud, self.dd, v, c)
        else:
            c[1:]  = self.ld*v[:N]
            c[:]  += self.dd*v[:]
            c[:N] += self.ud*v[1:]    
            for i in range(1,N):
                c[:(N-i)] += self.uud[i-1]*v[(i+1):]
        return c

    def diags(self):
        N = self.shape[0]
        return diags([self.ld, self.dd, self.ud] + self.uud, range(-1, N+1))



class A_matrix(object):    
    """Shen stiffness matrix for inner product (u'', phi_k) = (phi_j'', phi_k) u_hat = A * u_hat
    where u_hat is a vector of coefficients for a general Shen transform
    and phi_j and phi_k are general Shen basis functions.
    """

    def __init__(self, K, quad, a_j, b_j, a_k, b_k, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        ck = ones(K.shape)
        N = self.shape[0]
        if self.quad == "GL":
            ck[0] = 2
        elif self.quad == "GC":
            ck[0] = 2; ck[-1] = 2   
        self.dd = 2*pi*(K + 1)*(K + 2)*b_j

        self.ud = []
        for i in range(2, N, 2):
            T1 = a_j[i:]*(K[i:]+1)*((K[i:] + 1)**2 - K[:-i]**2)
            T2 = a_k[:-i]*a_j[i:]*(K[i:]+1)*((K[i:] + 1)**2 - (K[:-i]+1)**2)
            T3 = b_k[:-i]*a_j[i:]*(K[i:]+1)*((K[i:] + 1)**2 - (K[:-i]+2)**2)
            self.ud.append(array((pi/2)*(T1 + T2 + T3) )) 
            if (i+1)<N:
                L1 = K[i:]*(K[i:]**2-K[:-i]**2) + b_j[i:]*(K[i:]+2)*((K[i:]+2)**2 - K[:-i]**2)
                L2 = a_k[:-i]*( K[i:]*(K[i:]**2-(K[:-i]+1)**2) + b_j[i:]*(K[i:]+2)*((K[i:]+2)**2 - (K[:-i]+1)**2) )
                L2 = b_k[:-i]*( K[i:]*(K[i:]**2-(K[:-i]+2)**2) + b_j[i:]*(K[i:]+2)*((K[i:]+2)**2 - (K[:-i]+2)**2) )
                self.ud.append(array((pi/2)*(L1 + L2 + L3)))  

    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=complex)
        ud = asarray(self.ud)
        if len(v.shape) > 1:
            raise NotImplementedError
            #A_matvec(ud, self.dd, v, c)
        else:
            c[:]  = self.dd*v[:]   
            for i in range(1,N):
                c[:-i] += self.ud[i-1]*v[i:]
        return c
        
    def diags(self):
        N = self.shape[0]
        return diags([self.dd] + self.ud, range(0, N+1))
    
