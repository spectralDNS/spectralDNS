import numpy as np
from cbcdns import config
from SFTc import CDNmat_matvec, BDNmat_matvec, CDDmat_matvec
from scipy.sparse import diags

pi, zeros, ones, array = np.pi, np.zeros, np.ones, np.array
float, complex = np.float64, np.complex128

# Mass matrices    
class BNDmat(object):
    """Matrix for inner product (p, phi)_w = BNDmat * p_hat
    
    where p_hat is a vector of coefficients for a Shen Neumann basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-3, K.shape[0]-2)
        ck = ones(K.shape[0], int)
        N = shape[1]
        if quad == "GL": ck[-1] = 2
        self.dd = (pi/2.*(1+ck[3:]*(K[1:N]/(K[1:N]+2))**2)).astype(float)
        self.ud = (-pi/2*(K[1:N-2]/(K[1:N-2]+2))**2).astype(float)
        self.ld = -pi/2
    
    def matvec(self, v):
        N = self.shape[1]
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            c[1:(N-2)] = self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[3:N].shape)*v[3:N]
            c[1:N] += self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[1:N].shape)*v[1:N]
            c[2:N] += self.ld*v[:(N-2)]

        else:
            c[1:(N-2)]= self.ud*v[3:N]
            c[1:N]   += self.dd*v[1:N]
            c[2:N]   += self.ld*v[:(N-2)]

        return c

    def diags(self):
        return diags([self.ld*ones(self.shape[0]-1), self.dd, self.ud], [-1, 1, 3], shape=self.shape)
    

class BDNmat(object):
    """Matrix for inner product (p, phi_N)_w = BDNmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi_N is a Shen Neumann basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-3)
        ck = ones(K.shape[0], int)
        N = shape[0]
        if quad == "GL": ck[-1] = 2        
        self.dd = (pi/2.*(1+ck[3:]*(K[1:N]/(K[1:N]+2))**2)).astype(float)
        self.ud = -pi/2
        self.ld = -pi/2*((K[3:N]-2)/(K[3:N]))**2
    
    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            #c[:(N-2)] = self.ud*v[2:N]
            #c[1:N] += self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[1:N].shape)*v[1:N]
            #c[3:N] += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[1:(N-2)].shape)*v[1:(N-2)]
            BDNmat_matvec(self.ud, self.ld, self.dd, v, c)

        else:
            c[:(N-2)] = self.ud*v[2:N]
            c[1:N]   += self.dd*v[1:N]
            c[3:N]   += self.ld*v[1:(N-2)]
        return c

    def diags(self):
        return diags([self.ld, self.dd, self.ud*ones(self.shape[1]-1, float)], [-3, -1, 1], shape=self.shape)
    
class BTTmat(object):
    """Matrix for inner product (p, T)_w = BTTmat * p_hat
    
    where p_hat is a vector of coefficients for a Chebyshev basis
    and T is a Chebyshev basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0], K.shape[0])
        N = shape[0]
        ck = ones(N, int)
        ck[0] = 2
        if quad == "GL": ck[-1] = 2        
        self.dd = pi/2*ck
    
    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            c[:] = self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[:].shape)*v[:]  
        else:
            c[:] = self.dd*v[:]
        return c
    
    def diags(self):
        return diags([self.dd], [0], shape=self.shape)

class BNNmat(object):
    """Matrix for inner product (p, phi_N)_w = BNNmat * p_hat
    
    where p_hat is a vector of coefficients for a Shen Neumann basis
    and phi_N is a Shen Neumann basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-3, K.shape[0]-3)
        N = shape[0]+1        
        ck = ones(K.shape[0], int)
        if quad == "GL": ck[-1] = 2        
        self.dd = pi/2*(1+ck[3:]*(K[1:N]/(K[1:N]+2))**4)
        self.ud = -pi/2*(K[1:(N-2)]/(K[1:(N-2)]+2))**2
        self.ld = -pi/2*((K[3:N]-2)/(K[3:N]))**2
    
    def matvec(self, v):
        N = self.shape[0]+1
        c = zeros(v.shape, dtype=v.dtype)
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

class BDDmat(object):
    """Matrix for inner product (u, phi)_w = BDDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        N = shape[0] 
        ck = ones(K.shape[0], int)
        ck[0] = 2
        if quad == "GL": ck[-1] = 2
        self.dd = pi/2*(ck[:-2]+ck[2:])
        self.ud = float(-pi/2)
        self.ld = float(-pi/2)
    
    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            c[:(N-2)] = self.ud*v[2:N]
            c[:N]    += self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[:N].shape)*v[:N]
            c[2:N]   += self.ld*v[:(N-2)] 
        else:
            c[:(N-2)] = self.ud*v[2:N]
            c[:N]    += self.dd*v[:N]
            c[2:N]   += self.ld*v[:(N-2)]
        return c
    
    def diags(self):
        return diags([self.ld, self.dd, self.ud], [-2, 0, 2], shape=self.shape)

class BDTmat(object):
    """Matrix for inner product (u, phi)_w = BDTmat * u_hat
    
    where u_hat is a vector of coefficients for a Chebyshev basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0])
        N = shape[0]
        ck = ones(N+2, int)
        ck[0] = 2
        if quad == "GL": ck[-1] = 2        
        self.dd = pi/2*ck[:N]
        self.ud = -pi/2*ck[2:]
    
    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            c[:N]  = self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[:N].shape)*v[:N]
            c[:N] += self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[2:].shape)*v[2:]
        else:
            c[:N]  = self.dd*v[:N]
            c[:N] += self.ud*v[2:]
        return c
    
    def diags(self):
        return diags([self.dd, self.ud], [0, 2], shape=self.shape)

class BTDmat(object):
    """Matrix for inner product (u, T)_w = BTDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and T is a Chebyshev basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0], K.shape[0]-2)
        N = shape[1]
        ck = ones(N+2, int)
        ck[0] = 2
        if quad == "GL": ck[-1] = 2        
        self.dd = pi/2*ck[:N]
        self.ld = -pi/2*ck[2:]
    
    def matvec(self, v):
        N = self.shape[1]
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            c[:N]  = self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[:N].shape)*v[:N]
            c[2:] += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[:N].shape)*v[:N]
        else:
            c[:N]  = self.dd*v[:N]
            c[2:] += self.ld*v[:N]
        return c
    
    def diags(self):
        return diags([self.ld, self.dd], [-2, 0], shape=self.shape)

class BTNmat(object):
    """Matrix for inner product (u, T)_w = BTNmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Neumann basis
    and T is a Chebyshev basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0], K.shape[0]-3)
        N = shape[0]-2
        ck = ones(shape[0], int)
        if quad == "GL": ck[-1] = 2        
        self.dd = pi/2
        self.ld = -pi/2*ck[3:]*((K[3:]-2)/K[3:])**2
    
    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            c[0] = 0
            c[1:N-2]  = self.dd*v[1:N-2]
            c[3:] += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[1:N-2].shape)*v[1:N-2]
        else:
            c[0] = 0
            c[1:N-2]  = self.dd*v[1:N-2]
            c[3:] += self.ld*v[1:N-2]
        return c
    
    def diags(self):
        return diags([self.ld, self.dd], [-3, -1], shape=self.shape)


# Derivative matrices
class CDNmat(object):
    """Matrix for inner product (p', phi)_w = CDNmat * p_hat
    
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
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            #c[:(N-1)] = self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[1:N].shape)*v[1:N]
            #c[2:N]   += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[1:(N-1)].shape)*v[1:(N-1)]
            CDNmat_matvec(self.ud, self.ld, v, c)
        else:
            c[:(N-1)] = self.ud*v[1:N]
            c[2:N]   += self.ld*v[1:(N-1)]
        return c

class CDDmat(object):
    """Matrix for inner product (u', phi) = (phi', phi) u_hat =  CDDmat * u_hat
    
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
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            #c[:N-1] = self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[1:N].shape)*v[1:N]
            #c[1:N] += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[:(N-1)].shape)*v[:(N-1)]
            CDDmat_matvec(self.ud, self.ld, v, c)
        else:
            c[:N-1] = self.ud*v[1:N]
            c[1:N] += self.ld*v[:(N-1)]
        return c

    def diags(self):
        return diags([self.ld, self.dd, self.ud], [-1, 0, 1], shape=self.shape)

class CNDmat(object):
    """Matrix for inner product (u', phi_N) = (phi', phi_N) u_hat =  CNDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi_N is a Shen Neumann basis.
    """

    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-3, K.shape[0]-2)
        N = shape[1]
        self.ld = -(K[1:N]+1)*pi
        self.ud = [-(2-K[1:(N-1)]**2/(K[1:(N-1)]+2)**2*(K[1:(N-1)]+3))*pi]
        for i in range(3, N-1, 2):
            self.ud.append(-(1-K[1:(N-i)]**2/(K[1:(N-i)]+2)**2)*2*pi)    

    def matvec(self, v):
        N = self.shape[1]
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            C = self.diags().toarray()
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    c[1:N,i,j] = np.dot(C, v[:N,i,j])
        else:
            c[1:N] = np.dot(self.diags().toarray(), v[:N])
        return c

    def diags(self):
        return diags([self.ld]+self.ud, range(0, self.shape[1], 2), shape=self.shape)

class CTDmat(object):
    """Matrix for inner product (u', T) = (phi', T) u_hat =  CTDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and T is a Chebyshev basis.
    """

    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0], K.shape[0]-2)
        N = shape[0]
        self.ld = -(K[1:N]+1)*pi
        self.ud = [-2*pi]
        for i in range(3, N-2, 2):
            self.ud.append(-2*pi)    

    def matvec(self, v):
        N = self.shape[1]
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            C = self.diags().toarray()
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    c[:,i,j] = np.dot(C, v[:N,i,j])

        else:
            c[:] = np.dot(self.diags().toarray(), v[:N])
        return c

    def diags(self):
        ud = []
        for i, u in enumerate(self.ud):
            ud.append(u*ones(self.shape[1]-2*i-1))
        return diags([self.ld]+ud, range(-1, self.shape[1], 2), shape=self.shape)
    
class CDTmat(object):
    """Matrix for inner product (p', phi) = (T', phi) p_hat = CDTmat * p_hat
    
    where p_hat is a vector of coefficients for a Chebyshev basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        self.shape  = (K.shape[0]-2, K.shape[0])
        N = self.shape[0]
        self.ld = []
        self.ud = pi*(K[:N]+1)

    def matvec(self, v):
        N = self.shape[0]
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            c[:N] = self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[1:-1].shape)*v[1:-1]
        else:
            c[:N] = self.ud*v[1:-1]
        return c        

    def diags(self):
        return diags(self.ud, [1], shape=self.shape)
    
    
class ADDmat(object):
    """Matrix for inner product -(u'', phi) = -(phi'', phi) u_hat = ADDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        N = shape[0]
        self.dd = 2*np.pi*(K[:N]+1)*(K[:N]+2)   
        self.ud = []
        for i in range(2, N, 2):
            self.ud.append(np.array(4*np.pi*(K[:-(i+2)]+1)))    

        self.ld = None
        
    def matvec(self, v):
        N = self.shape[0]
        c = np.zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            raise NotImplementedError
        else:
            c[:N] = np.dot(self.diags().toarray(), v[:N])
        return c

    def diags(self):
        N = self.shape[0]
        return diags([self.dd] + self.ud, range(0, N, 2))


class ANNmat(object):
    """Matrix for inner product -(u'', phi_N) = -(phi_N'', phi_N) p_hat = ANNmat * p_hat
    
    where u_hat is a vector of coefficients for a Shen Neumann basis
    and phi is a Shen Neumann basis.
    """

    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-3, K.shape[0]-3)
        N = shape[0]+1
        self.dd = 2*pi*K[1:N]**2*(K[1:N]+1)/(K[1:N]+2)        
        self.ud = []
        for i in range(2, N-1, 2):
            self.ud.append(np.array(4*np.pi*(K[1:-(i+2)]+i)**2*(K[1:-(i+2)]+1)/(K[1:-(i+2)]+2)**2))    

        self.ld = None
        
    def matvec(self, v):
        N = self.shape[0]+1
        c = np.zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            raise NotImplementedError
        else:
            c[1:N] = np.dot(self.diags().toarray(), v[1:N])
        return c

    def diags(self):
        N = self.shape[0]
        return diags([self.dd] + self.ud, range(0, N, 2))

