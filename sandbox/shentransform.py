import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from scipy.fftpack import dct, idct
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
import SFTc

"""
Fast transforms for pure Chebyshev basis or 
Shen's Chebyshev basis: 

  For homogeneous Dirichlet boundary conditions:
 
    phi_k = T_k - T_{k+2}
    
  For homogeneous Neumann boundary conditions:
    
    phi_k = T_k - (k/k+2)**2 * T_{k+2}

Use either Chebyshev-Gauss (GC) or Gauss-Lobatto (GL)
points in real space.

The ChebyshevTransform may be used to compute derivatives
through fast Chebyshev transforms.

"""
pi, zeros, ones = np.pi, np.zeros, np.ones

dct1 = dct
def dct(x, i, axis=0):
    if np.iscomplexobj(x):
        xreal = dct1(x.real, i, axis=axis)
        ximag = dct1(x.imag, i, axis=axis)
        return xreal + ximag*1j
    else:
        return dct1(x, i, axis=axis)

class ChebyshevTransform(object):
    
    def __init__(self, quad="GC"): 
        self.quad = quad
        
    def points_and_weights(self, N):
        self.N = N
        if self.quad == "GC":
            points = n_cheb.chebpts2(N)[::-1]
            weights = np.zeros((N))+np.pi/(N-1)
            weights[0] /= 2
            weights[-1] /= 2

        elif self.quad == "GL":
            points, weights = n_cheb.chebgauss(N)
            
        return points, weights
        
    def chebDerivativeCoefficients(self, fk, fj):
        SFTc.chebDerivativeCoefficients(fk, fj)  
        return fj

    def chebDerivative_3D(self, fj, fd, fk, fkd):
        fk = self.fct(fj, fk)
        fkd = SFTc.chebDerivativeCoefficients_3D(fk, fkd)
        fd = self.ifct(fl, fd)
        return fd
    
    def fastChebDerivative(self, fj, fd, fk, fkd):
        """Compute derivative of fj at the same points."""
        fk = self.fct(fj, fk)
        fkd = self.chebDerivativeCoefficients(fk, fkd)
        fd  = self.ifct(fkd, fd)
        return fd
        
    def fct(self, fj, cj):
        """Fast Chebyshev transform."""
        N = fj.shape[0]
        if self.quad == "GL":
            cj = dct(fj, 2, axis=0)
            cj /= N
            cj[0] /= 2
                
        elif self.quad == "GC":
            cj = dct(fj, 1, axis=0)/(N-1)
            cj[0] /= 2
            cj[-1] /= 2
            
        return cj

    def ifct(self, fk, cj):
        """Inverse fast Chebyshev transform."""
        if self.quad == "GL":
            cj = 0.5*dct(fk, 3, axis=0)
            cj += 0.5*fk[0]
        
        elif self.quad == "GC":
            cj = 0.5*dct(fk, 1, axis=0)
            cj += 0.5*fk[0]
            cj[::2] += 0.5*fk[-1]
            cj[1::2] -= 0.5*fk[-1]

        return cj
    
    def fastChebScalar(self, fj, fk):
        """Fast Chebyshev scalar product."""
        N = fj.shape[0]
        if self.quad == "GL":
            fk = dct(fj, 2, axis=0)*np.pi/(2*N)
        
        elif self.quad == "GC":
            fk = dct(fj, 1, axis=0)*np.pi/(2*(N-1))
        return fk

class ShenDirichletBasis(ChebyshevTransform):
    
    def __init__(self, quad="GC", fast_transform=True):
        self.quad = quad
        self.fast_transform = fast_transform
        self.points = None
        self.weights = None
        self.N = -1
        
    def init(self, N):
        """Vandermonde matrix is used just for verification"""
        self.points, self.weights = self.points_and_weights(N)
        # Build Vandermonde matrix. Note! N points in real space gives N-2 bases in spectral space
        self.V = n_cheb.chebvander(self.points, N-3).T - n_cheb.chebvander(self.points, N-1)[:, 2:].T

    def wavenumbers(self, N):
        if isinstance(N, tuple):
            if len(N) == 1:
                N = N[0]
        if isinstance(N, int): 
            return np.arange(N-2).astype(np.float)
        
        else:
            kk = np.mgrid[:N[0]-2, :N[1], :N[2]].astype(float)
            return kk[0]

    def fastShenScalar(self, fj, fk):
                
        if self.fast_transform:
            fk = self.fastChebScalar(fj, fk)
            fk[:-2] -= fk[2:]
        else:
            if self.points is None: self.init(fj.shape[0])
            fk[:-2] = np.dot(self.V, fj*self.weights)
        return fk
        
    def ifst(self, fk, fj):
        """Fast inverse Shen transform
        Transform needs to take into account that phi_k = T_k - T_{k+2}
        fk contains Shen coefficients in the first fk.shape[0]-2 positions
        """
        if self.fast_transform:
            N = len(fj)
            if len(fk.shape) == 3:
                w_hat = np.zeros(fj.shape, dtype=fk.dtype)
                w_hat[:-2] = fk[:-2]
                w_hat[2:] -= fk[:-2]
            if len(fk.shape) == 1:
                w_hat = np.zeros(N)                
                w_hat[:-2] = fk[:-2] 
                w_hat[2:] -= fk[:-2] 
                #- np.hstack([0, 0, fk[:-2]])    
                #w_hat[-2] = -fk[-2]
                #w_hat[-1] = -fk[-1]
            fj = self.ifct(w_hat, fj)
            return fj    
        
        else:
            if not self.points: self.init(fj.shape[0])
            return np.dot(self.V.T, fk[:-2])

    def fst(self, fj, fk):
        """Fast Shen transform
        """
        fk = self.fastShenScalar(fj, fk)
        
        N = fj.shape[0]
        if self.quad == "GL":
            ck = np.ones(N-2); ck[0] = 2
            
        elif self.quad == "GC":
            ck = np.ones(N-2); ck[0] = 2; ck[-1] = 2  # Note!! Shen paper has only ck[0] = 2, not ck[-1] = 2. For Gauss points ck[-1] = 1, but not here! 
            
        a = np.ones(N-4)*(-np.pi/2)
        b = np.pi/2*(ck+1)
        c = a.copy()
        if len(fk.shape) == 3:
            bc = b.copy()
            fk[:-2] = SFTc.TDMA_3D_complex(a, b, bc, c, fk[:-2])

        elif len(fk.shape) == 1:
            fk[:-2] = SFTc.TDMA_1D(a, b, c, fk[:-2])
            
        return fk
    

class ShenNeumannBasis(ShenDirichletBasis):
    
    def __init__(self, quad="GL"): 
        ShenDirichletBasis.__init__(self, quad)
            
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        k = self.wavenumbers(N)
        # Build Vandermonde matrix. Note! N points in real space gives N-3 bases in spectral space
        self.V = n_cheb.chebvander(self.points, N-3).T - ((k/(k+2))**2)[:, np.newaxis]*n_cheb.chebvander(self.points, N-1)[:, 2:].T
        self.V = self.V[1:, :]

    def fastShenScalar(self, fj, fk):
        """Fast Shen scalar product.
        Chebyshev transform taking into account that phi_k = T_k - (k/(k+2))**2*T_{k+2}
        Note, this is the non-normalized scalar product

        """
        
        if self.fast_transform:
            k  = self.wavenumbers(fj.shape)
            fk = self.fastChebScalar(fj, fk)
            fk[:-2] -= ((k/(k+2))**2) * fk[2:]

        else:
            if self.points is None: self.init(fj.shape[0])
            fk = np.dot(self.V, fj*self.weights)
        return fk

    def ifst(self, fk, fj):
        """Fast inverse Shen scalar transform
        """
        if len(fk.shape)==3:
            k = self.wavenumbers(fk.shape)
            w_hat = np.zeros(fk.shape, dtype=fk.dtype)
            w_hat[1:-2] = fk[1:-2]
            w_hat[3:] -= (k[1:]/(k[1:]+2))**2*fk[1:-2]
            
        elif len(fk.shape)==1:
            k = self.wavenumbers(fk.shape[0])
            w_hat = np.zeros(fk.shape[0])    
            w_hat[1:-2] = fk[1:-2]
            w_hat[3:] -= (k[1:]/(k[1:]+2))**2*fk[1:-2]

        fj = self.ifct(w_hat, fj)
        return fj
        
    def fst(self, fj, fk):
        """Fast Shen transform.
        """
        fk = self.fastShenScalar(fj, fk)
        N = fj.shape[0]
        k = self.wavenumbers(N)
        ck = np.ones(N-3)
        if self.quad == "GC": ck[-1] = 2 # Note not the first since basis phi_0 is not included        
        a = np.ones(N-5)*(-np.pi/2)*(k[1:-2]/(k[1:-2]+2))**2
        b = np.pi/2*(1+ck*(k[1:]/(k[1:]+2))**4)
        c = a.copy()
        if len(fk.shape) == 3:
            bc = b.copy()
            fk[1:-2] = SFTc.TDMA_3D_complex(a, b, bc, c, fk[1:-2])

        elif len(fk.shape) == 1:
            fk[1:-2] = SFTc.TDMA_1D(a, b, c, fk[1:-2])

        return fk

class Chmat(LinearOperator):
    """Matrix for inner product (p', phi)_w = Chmat * p_hat
    
    where p_hat is a vector of coefficients for a Shen Neumann basis
    and phi is a Shen Dirichlet basis.
    """
    
    def __init__(self, K, **kwargs):
        shape = (K.shape[0]-2, K.shape[0]-3)
        LinearOperator.__init__(self, shape, None, **kwargs)
        N = shape[0]
        self.ud = (K[:(N-1)]+1)*pi
        self.ld = -((K[2:N]-1)/(K[2:N]+1))**2*(K[2:N]+1)*pi
        self.c = zeros(K.shape, dtype=complex)
        
    def matvec(self, v):
        N = self.shape[0]
        self.c[:] = 0
        self.c[:(N-1)] = self.ud*v[1:N]
        self.c[2:N] += self.ld*v[1:(N-1)]
        return self.c
    
class Bhmat(LinearOperator):
    """Matrix for inner product (p, phi)_w = Bhmat * p_hat
    
    where p_hat is a vector of coefficients for a Shen Neumann basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, quad, **kwargs):
        shape = (K.shape[0]-2, K.shape[0]-3)
        LinearOperator.__init__(self, shape, None, **kwargs)
        ck = ones(K.shape)
        N = shape[0]
        if quad == "GC": ck[N-1] = 2        
        self.dd = pi/2.*(1+ck[1:N]*(K[1:N]/(K[1:N]+2))**2)
        self.ud = -pi/2
        self.ld = -pi/2*((K[3:N]-2)/(K[3:N]))**2
        self.c = zeros(K.shape, dtype=complex)
    
    def matvec(self, v):
        N = self.shape[0]
        self.c[:] = 0
        self.c[:(N-2)] = self.ud*v[2:N]
        self.c[1:N]   += self.dd*v[1:N]
        self.c[3:N]   += self.ld*v[1:(N-2)]
        return self.c
    
    def diags(self):
        if len(self.dd.shape) == 3:
            return diags([self.ld[:, 0, 0], self.dd[:, 0, 0], self.ud*ones(self.shape[1]-1)], [-3, -1, 1], shape=self.shape)
        elif len(self.dd.shape) == 1:
            return diags([self.ld, self.dd, self.ud*ones(self.shape[1]-1)], [-3, -1, 1], shape=self.shape)
        else:
            raise NotImplementedError

class Cmat(LinearOperator):
    """Matrix for inner product (u', phi) = (phi', phi) u_hat =  Cmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, **kwargs):
        shape = (K.shape[0]-2, K.shape[0]-2)
        N = shape[0]
        LinearOperator.__init__(self, shape, None, **kwargs)
        self.ud = (K[:(N-1)]+1)*pi
        self.ld = -(K[1:N]+1)*pi
        self.c = zeros(K.shape, dtype=complex)
        
    def matvec(self, v):
        N = self.shape[0]
        self.c[:] = 0
        self.c[:N-1] = self.ud*v[1:N]
        self.c[1:N] += self.ld*v[:N-1]
        return self.c

    def diags(self):
        if len(self.dd.shape) == 3:
            return diags([self.ld[:, 0, 0], self.dd[:, 0, 0], self.ud[:, 0, 0]], [-1, 0, 1], shape=self.shape)
        elif len(self.dd.shape) == 1:
            return diags([self.ld, self.dd, self.ud], [-1, 0, 1], shape=self.shape)
        else:
            raise NotImplementedError


class Bmat(LinearOperator):
    """Matrix for inner product (p, phi_N)_w = Bmat * p_hat
    
    where p_hat is a vector of coefficients for a Shen Neumann basis
    and phi_N is a Shen Neumann basis.
    """

    def __init__(self, K, quad, **kwargs):
        shape = (K.shape[0]-3, K.shape[0]-3)
        LinearOperator.__init__(self, shape, None, **kwargs)
        ck = ones(K.shape)
        N = shape[0]+1        
        if quad == "GC": ck[N-1] = 2        
        self.dd = pi/2*(1+ck[1:N]*(K[1:N]/(K[1:N]+2))**4)/K[1:N]**2
        self.ud = -pi/2*(K[1:(N-2)]/(K[1:(N-2)]+2))**2/(K[1:(N-2)]+2)**2
        self.ld = -pi/2*((K[3:N]-2)/(K[3:N]))**2/(K[3:N]-2)**2
        self.c = zeros(K.shape, dtype=complex)
    
    def matvec(self, v):
        N = self.shape[0]
        self.c[:] = 0
        self.c[1:(N-2)] = self.ud*v[3:N]
        self.c[1:N]    += self.dd*v[1:N]
        self.c[3:N]    += self.ld*v[1:(N-2)]
        return self.c
    
    def diags(self):
        if len(self.dd.shape) == 3:
            return diags([self.ld[:, 0, 0], self.dd[:, 0, 0], self.ud[:, 0, 0]], [-2, 0, 2], shape=self.shape)
        elif len(self.dd.shape) == 1:
            return diags([self.ld, self.dd, self.ud], [-2, 0, 2], shape=self.shape)
        else:
            raise NotImplementedError

class Amat(LinearOperator):
    """Matrix for inner product -(u'', phi) = -(phi'', phi) u_hat = Amat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, **kwargs):
        shape = (K.shape[0]-2, K.shape[0]-2)
        N = shape[0]
        LinearOperator.__init__(self, shape, None, **kwargs)
        self.dd = 2*np.pi*(K[:N]+1)*(K[:N]+2)   
        self.ud = []
        for i in range(2, N-2, 2):
            self.ud.append(np.array(4*np.pi*(k[:-i]+1)))    

        self.ud = (K[:(N-1)]+1)*pi
        self.ld = None
        
    def matvec(self, v):
        raise NotImplementedError

    def diags(self):
        N = shape[0]
        if len(self.dd.shape) == 3:
            return diags([self.dd[:, 0, 0]] + [ud[:, 0, 0] for ud in self.ud], range(0, N-2, 2))
        elif len(self.dd.shape) == 1:
            return diags([self.dd] + self.ud, range(0, N-2, 2))

        else:
            raise NotImplementedError


class dP2Tmat(LinearOperator):
    """Matrix for projecting -(u', T) = -(phi', T) u_hat = dP2Tmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and T is the Chebyshev basis.
    """

    def __init__(self, K, **kwargs):
        shape = (K.shape[0], K.shape[0]-2)
        N = shape[0]
        LinearOperator.__init__(self, shape, None, **kwargs)
        self.dd = 0
        self.ud = -2*pi
        self.ld = -(K[1:(N-1)]+1)*pi
        self.c = zeros(K.shape, dtype=complex)
        
    def matvec(self, v):
        raise NotImplementedError

    def diags(self):
        N = shape[0]
        if len(self.dd.shape) == 3:
            return diags([self.dd[:, 0, 0]] + [ud[:, 0, 0] for ud in self.ud], range(0, N-2, 2))
        elif len(self.dd.shape) == 1:
            return diags([self.dd] + self.ud, range(0, N-2, 2))

        else:
            raise NotImplementedError


    
if __name__ == "__main__":
    N = 8
    a = np.random.random((N, N, N/2+1))+1j*np.random.random((N, N, N/2+1))
    af = np.zeros((N, N, N/2+1), dtype=a.dtype)
    a[0,:,:] = 0
    a[-1,:,:] = 0
    #a = np.random.random(N)
    #af = np.zeros(N, dtype=np.complex)
    #a[0] = 0
    #a[-1] = 0
    
    ST = ShenDirichletBasis(quad="GC")
    af = ST.fst(a, af) 
    a0 = a.copy()
    a0 = ST.ifst(af, a0)
    assert np.allclose(a0, a)
     