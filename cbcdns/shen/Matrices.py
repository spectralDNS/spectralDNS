import numpy as np
from cbcdns import config
if config.precision == "double":
    from SFTc_double import Chmat_matvec, Bhmat_matvec, Cmat_matvec
else:
    from SFTc_single import Chmat_matvec, Bhmat_matvec, Cmat_matvec
    
from scipy.sparse import diags

pi, zeros, ones, array = np.pi, np.zeros, np.ones, np.array

class Chmat(object):
    """Matrix for inner product (p', phi)_w = Chmat * p_hat
    
    where p_hat is a vector of coefficients for a Shen Neumann basis
    and phi is a Shen Dirichlet basis.
    """
    
    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-3)
        N = shape[0]
        self.ud = (K[:(N-1)]+1)*pi
        self.ld = -((K[2:N]-1)/(K[2:N]+1))**2*(K[2:N]+1)*pi
        
    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=complex)
        if len(v.shape) > 1:
            #c[:(N-1)] = self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[1:N].shape)*v[1:N]
            #c[2:N]   += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[1:(N-1)].shape)*v[1:(N-1)]
            Chmat_matvec(self.ud, self.ld, v, c)
        else:
            c[:(N-1)] = self.ud*v[1:N]
            c[2:N]   += self.ld*v[1:(N-1)]
        return c
    
class Bhmat(object):
    """Matrix for inner product (p, phi)_w = Bhmat * p_hat
    
    where p_hat is a vector of coefficients for a Shen Neumann basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-3)
        ck = ones(K.shape)
        N = shape[0]
        if quad == "GL": ck[N-1] = 2        
        self.dd = pi/2.*(1+ck[1:N]*(K[1:N]/(K[1:N]+2))**2)
        self.ud = -pi/2
        self.ld = -pi/2*((K[3:N]-2)/(K[3:N]))**2
    
    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=complex)
        if len(v.shape) > 1:
            #c[:(N-2)] = self.ud*v[2:N]
            #c[1:N] += self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[1:N].shape)*v[1:N]
            #c[3:N] += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[1:(N-2)].shape)*v[1:(N-2)]
            Bhmat_matvec(self.ud, self.ld, self.dd, v, c)

        else:
            c[:(N-2)] = self.ud*v[2:N]
            c[1:N]   += self.dd*v[1:N]
            c[3:N]   += self.ld*v[1:(N-2)]
        return c
    
    def diags(self):
        return diags([self.ld, self.dd, self.ud*ones(self.shape[1]-1)], [-3, -1, 1], shape=self.shape)

class Cmat(object):
    """Matrix for inner product (u', phi) = (phi', phi) u_hat =  Cmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        N = shape[0]
        self.ud = (K[:(N-1)]+1)*pi
        self.ld = -(K[1:N]+1)*pi
        
    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=complex)
        if len(v.shape) > 1:
            #c[:N-1] = self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[1:N].shape)*v[1:N]
            #c[1:N] += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[:(N-1)].shape)*v[:(N-1)]
            Cmat_matvec(self.ud, self.ld, v, c)
        else:
            c[:N-1] = self.ud*v[1:N]
            c[1:N] += self.ld*v[:(N-1)]
        return c

    def diags(self):
        return diags([self.ld, self.dd, self.ud], [-1, 0, 1], shape=self.shape)

class Bmat(object):
    """Matrix for inner product (p, phi_N)_w = Bmat * p_hat
    
    where p_hat is a vector of coefficients for a Shen Neumann basis
    and phi_N is a Shen Neumann basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-3, K.shape[0]-3)
        ck = ones(K.shape)
        N = shape[0]+1        
        if quad == "GL": ck[N-1] = 2        
        self.dd = pi/2*(1+ck[1:N]*(K[1:N]/(K[1:N]+2))**4)/K[1:N]**2
        self.ud = -pi/2*(K[1:(N-2)]/(K[1:(N-2)]+2))**2/(K[1:(N-2)]+2)**2
        self.ld = -pi/2*((K[3:N]-2)/(K[3:N]))**2/(K[3:N]-2)**2
    
    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=complex)
        if len(v.shape) > 1:
            c[1:(N-2)] = self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[3:N].shape)*v[3:N]
            c[1:N]    += self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[1:N].shape)*v[1:N]
            c[3:N]    += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[1:(N-2)].shape)*v[1:(N-2)] 
        else:
            c[1:(N-2)] = self.ud*v[3:N]
            c[1:N]    += self.dd*v[1:N]
            c[3:N]    += self.ld*v[1:(N-2)]
        return c
    
    def diags(self):
        return diags([self.ld, self.dd, self.ud], [-2, 0, 2], shape=self.shape)

class BDmat(object):
    """Matrix for inner product (u, phi)_w = BDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        N = shape[0] 
        ck = ones(K.shape)
        ck[0] = 2
        if quad == "GL": ck[N-1] = 2
        self.dd = pi/2*(ck[:-2]+ck[2:])
        self.ud = -pi/2
        self.ld = -pi/2
    
    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=complex)
        if len(v.shape) > 1:
            c[:(N-2)] = self.ud.repeat(array(v.shape[:]).prod()).reshape(v[2:N].shape)*v[2:N]
            c[:N]    += self.dd.repeat(array(v.shape[:]).prod()).reshape(v[:N].shape)*v[:N]
            c[2:N]    += self.ld.repeat(array(v.shape[:]).prod()).reshape(v[:(N-2)].shape)*v[:(N-2)] 
        else:
            c[:(N-2)] = self.ud*v[2:N]
            c[:N]    += self.dd*v[:N]
            c[2:N]    += self.ld*v[:(N-2)]
        return c
    
    def diags(self):
        return diags([self.ld, self.dd, self.ud], [-2, 0, 2], shape=self.shape)
    
class Amat(object):
    """Matrix for inner product -(u'', phi) = -(phi'', phi) u_hat = Amat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        N = shape[0]
        self.dd = 2*np.pi*(K[:N]+1)*(K[:N]+2)   
        self.ud = []
        for i in range(2, N-2, 2):
            self.ud.append(np.array(4*np.pi*(K[:-(i+2)]+1)))    

        self.ld = None
        
    def matvec(self, v):
        raise NotImplementedError

    def diags(self):
        N = self.shape[0]
        return diags([self.dd] + self.ud, range(0, N-2, 2))


class ANmat(object):
    """Matrix for inner product -(u'', phi_N) = -(phi_N'', phi_N) p_hat = ANmat * p_hat
    
    where u_hat is a vector of coefficients for a Shen Neumann basis
    and phi is a Shen Neumann basis.
    """

    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-3, K.shape[0]-3)
        N = shape[0]+1
        self.dd = 2*np.pi*(K[1:N]+1)/(K[1:N]+2)        
        self.ud = []
        for i in range(2, N-1, 2):
            self.ud.append(np.array(4*np.pi*(K[1:-(i+2)]+1)/(K[1:-(i+2)]+2)**2))    

        self.ld = None
        
    def matvec(self, v):
        raise NotImplementedError

    def diags(self):
        N = self.shape[0]
        return diags([self.dd] + self.ud, range(0, N, 2))


class dP2Tmat(object):
    """Matrix for projecting -(u', T) = -(phi', T) u_hat = dP2Tmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and T is the Chebyshev basis.
    """

    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0], K.shape[0]-2)
        N = shape[0]
        self.dd = 0
        self.ud = []
        self.ld = -(K[1:N]+1)*pi
        for i in range(1, N-2, 2):
            self.ud.append(-2*pi)
            
        self.c = zeros(K.shape, dtype=complex)
        
    def matvec(self, v):
        raise NotImplementedError

    def diags(self):
        N = self.shape[0]
        return diags([self.ld] + self.ud, range(-1, N-2, 2), shape=self.shape)


class dTSmat(object):
    """Matrix for inner product (p', phi) = (T', phi) p_hat = dTSmat * p_hat
    
    where p_hat is a vector of coefficients for a Chebyshev basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0])
        N = shape[0]
        self.ld = []
        self.ud = np.pi*(K[1:N]+1)

    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=v.dtype)
        c[:N] = self.ud.repeat(array(v.shape[:]).prod()).reshape(v[1:(N+1)].shape)*v[1:(N+1)]
        return c        

    def diags(self):
        return diags(self.ud, [1], shape=self.shape)
