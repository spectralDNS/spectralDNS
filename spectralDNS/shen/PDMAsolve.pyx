import numpy as np
cimport cython
cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False

ctypedef fused T:
    np.float64_t
    np.complex128_t


def PDMA_SymLU(np.ndarray[np.float64_t, ndim=1, mode='c'] d, 
               np.ndarray[np.float64_t, ndim=1, mode='c'] e,
               np.ndarray[np.float64_t, ndim=1, mode='c'] f):
    cdef:
        unsigned int n = d.shape[0]
        unsigned int m = e.shape[0]
        unsigned int k = n - m
        unsigned int i
        double lam
        
    for i in xrange(n-2*k):
        lam = e[i]/d[i]
        d[i+k] -= lam*e[i]
        e[i+k] -= lam*f[i]
        e[i] = lam
        lam = f[i]/d[i]
        d[i+2*k] -= lam*f[i]
        f[i] = lam

    lam = e[n-4]/d[n-4]
    d[n-2] -= lam*e[n-4]
    e[n-4] = lam
    lam = e[n-3]/d[n-3]
    d[n-1] -= lam*e[n-3]
    e[n-3] = lam
 
def PDMA_Symsolve(np.ndarray[np.float64_t, ndim=1, mode='c'] d, 
                  np.ndarray[np.float64_t, ndim=1, mode='c'] e, 
                  np.ndarray[np.float64_t, ndim=1, mode='c'] f, 
                  np.ndarray[T, ndim=1, mode='c'] b):
    cdef:
        unsigned int n = d.shape[0]
        int k
        
    b[2] -= e[0]*b[0]
    b[3] -= e[1]*b[1]    
    for k in range(4, n):
        b[k] -= (e[k-2]*b[k-2] + f[k-4]*b[k-4])
 
    b[n-1] /= d[n-1]
    b[n-2] /= d[n-2]    
    b[n-3] /= d[n-3] 
    b[n-3] -= e[n-3]*b[n-1]
    b[n-4] /= d[n-4]
    b[n-4] -= e[n-4]*b[n-2]    
    for k in range(n-5,-1,-1):
        b[k] /= d[k] 
        b[k] -= (e[k]*b[k+2] + f[k]*b[k+4])

def PDMA_Symsolve3D(np.float64_t [::1] d, 
                    np.float64_t [::1] e, 
                    np.float64_t [::1] f, 
                    T [:, :, ::1] b):
    cdef:
        int i, j, k
        int n = d.shape[0]
        
    for i in xrange(b.shape[1]):
        for j in xrange(b.shape[2]):
            #PDMA_Symsolve(d, e, f, b[:, i, j])        
            b[2, i, j] -= e[0]*b[0, i, j]
            b[3, i, j] -= e[1]*b[1, i, j]    
        
    for k in xrange(4, n):
        for i in xrange(b.shape[1]):
            for j in xrange(b.shape[2]):
                b[k, i, j] -= (e[k-2]*b[k-2, i, j] + f[k-4]*b[k-4, i, j])
        
    for i in xrange(b.shape[1]):
        for j in xrange(b.shape[2]):
            b[n-1, i, j] /= d[n-1]
            b[n-2, i, j] /= d[n-2]    
            b[n-3, i, j] /= d[n-3] 
            b[n-3, i, j] -= e[n-3]*b[n-1, i, j]
            b[n-4, i, j] /= d[n-4]
            b[n-4, i, j] -= e[n-4]*b[n-2, i, j]    

    for k in xrange(n-5,-1,-1):
        for i in xrange(b.shape[1]):
            for j in xrange(b.shape[2]):            
                b[k, i, j] /= d[k] 
                b[k, i, j] -= (e[k]*b[k+2, i, j] + f[k]*b[k+4, i, j])
