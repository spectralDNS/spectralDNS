cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False
import numpy as np

ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t
ctypedef double real

ctypedef fused T:
    real_t
    complex_t

#def TDC_decomp(np.ndarray[real_t, ndim=1, mode='c'] d, 
               #np.ndarray[real_t, ndim=1, mode='c'] a,
               #np.ndarray[real_t, ndim=1, mode='c'] l1,
               #np.ndarray[real_t, ndim=1, mode='c'] l2):
    #cdef:
        #unsigned int n = d.shape[0]
        #unsigned int i
        
     #for i in range(n-1):
         #l1[i] = sqrt(d[i])
         #l2[i] = a[i] / l1[i]
         
           #L(k,k)=sqrt(A(k,k));
           #L(k+1:n,k)=(A(k+1:n,k))/L(k,k);
           #A(k+1:n,k+1:n)=A(k+1:n,k+1:n)-L(k+1:n,k)*L(k+1:n,k).T
     #end
     #L(n,n)=sqrt(A(n,n));

def TDMA_SymLU(np.ndarray[real_t, ndim=1, mode='c'] d, 
               np.ndarray[real_t, ndim=1, mode='c'] a,
               np.ndarray[real_t, ndim=1, mode='c'] l):
    cdef:
        unsigned int n = d.shape[0]
        int i
        
    for i in range(2, n):
        l[i-2] = a[i-2]/d[i-2]
        d[i] = d[i] - l[i-2]*a[i-2]
        
def TDMA_SymSolve(np.ndarray[real_t, ndim=1, mode='c'] d, 
                  np.ndarray[real_t, ndim=1, mode='c'] a,
                  np.ndarray[real_t, ndim=1, mode='c'] l,
                  np.ndarray[T, ndim=1, mode='c'] x):
    cdef:
        unsigned int n = d.shape[0]
        int i
        np.ndarray[T, ndim=1, mode='c'] y = np.zeros_like(x)
        
    y[0] = x[0]
    y[1] = x[1]
    for i in range(2, n):
        y[i] = x[i] - l[i-2]*y[i-2]
        
    x[n-1] = y[n-1]/d[n-1]
    x[n-2] = y[n-2]/d[n-2]
    for i in range(n - 3, -1, -1):
        x[i] = (y[i] - a[i]*x[i+2])/d[i]    

def TDMA_SymSolve3D(np.ndarray[real_t, ndim=1, mode='c'] d, 
                    np.ndarray[real_t, ndim=1, mode='c'] a,
                    np.ndarray[real_t, ndim=1, mode='c'] l,
                    np.ndarray[T, ndim=3, mode='c'] x):
    cdef:
        unsigned int n = d.shape[0]
        int i, j, k
        np.ndarray[T, ndim=3, mode='c'] y = np.zeros_like(x)
        
    for j in range(x.shape[1]):
        for k in range(x.shape[2]):
            y[0, j, k] = x[0, j, k]
            y[1, j, k] = x[1, j, k]
    
    for i in range(2, n):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                y[i, j, k] = x[i, j, k] - l[i-2]*y[i-2, j, k]

    for j in range(x.shape[1]):
        for k in range(x.shape[2]):        
            x[n-1, j, k] = y[n-1, j, k]/d[n-1]
            x[n-2, j, k] = y[n-2, j, k]/d[n-2]
    
    for i in range(n - 3, -1, -1):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):            
                x[i, j, k] = (y[i, j, k] - a[i]*x[i+2, j, k])/d[i]

def TDMA_1D(np.ndarray[real_t, ndim=1, mode='c'] a, 
            np.ndarray[real_t, ndim=1, mode='c'] b, 
            np.ndarray[real_t, ndim=1, mode='c'] bc,
            np.ndarray[real_t, ndim=1, mode='c'] c, 
            np.ndarray[T, ndim=1, mode='c'] d):
    cdef:
        unsigned int n = b.shape[0]
        unsigned int m = a.shape[0]
        unsigned int k = n - m
        int i
        
    bc[0] = b[0]
    bc[1] = b[1]
    for i in range(m):
        d[i + k] -= d[i] * a[i] / bc[i]
        bc[i + k] = b[i + k] - c[i] * a[i] / bc[i]
    for i in range(m - 1, -1, -1):
        d[i] -= d[i + k] * c[i] / bc[i + k]
    for i in range(n):
        d[i] /= bc[i]
        

def TDMA_3D(np.ndarray[real_t, ndim=1, mode='c'] a, 
            np.ndarray[real_t, ndim=1, mode='c'] b, 
            np.ndarray[real_t, ndim=1, mode='c'] bc, 
            np.ndarray[real_t, ndim=1, mode='c'] c, 
            np.ndarray[T, ndim=3, mode='c'] d):
    cdef:
        int n = b.shape[0]
        int m = a.shape[0]
        int l = n - m
        int i, j, k
        
    bc[0] = b[0]
    bc[1] = b[1]
    for i in range(m):
        bc[i + l] = b[i + l] - c[i] * a[i] / bc[i]
    
    for i in range(m):
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                d[i + l, j, k] -= d[i, j, k] * a[i] / bc[i]
                
    for i in range(m - 1, -1, -1):
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                d[i, j, k] -= d[i + l, j, k] * c[i] / bc[i + l]
                
    for i in range(n):
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                d[i, j, k] = d[i, j, k] / bc[i]
