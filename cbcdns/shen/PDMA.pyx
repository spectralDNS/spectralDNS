import numpy as np
cimport cython
cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False

ctypedef fused T:
    np.float64_t
    np.complex128_t


def PDMA_SymLU(np.ndarray[np.float64_t, ndim=1] d, np.ndarray[np.float64_t, ndim=1] e,
               np.ndarray[np.float64_t, ndim=1] f):
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
 
def PDMA_Symsolve(np.ndarray[np.float64_t, ndim=1] d, np.ndarray[np.float64_t, ndim=1] e, 
                  np.ndarray[np.float64_t, ndim=1] f, np.ndarray[T, ndim=1] b):
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

def PDMA_Symsolve3D(np.ndarray[np.float64_t, ndim=1] d, np.ndarray[np.float64_t, ndim=1] e, 
                    np.ndarray[np.float64_t, ndim=1] f, np.ndarray[T, ndim=3] u):
    cdef:
        unsigned int ii, jj
        
    for ii in range(u.shape[1]):
        for jj in range(u.shape[2]):
            PDMA_Symsolve(d, e, f, u[:, ii, jj])
