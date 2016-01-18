import numpy as np
from cbcdns import config
from SFTc import CDNmat_matvec, BDNmat_matvec, CDDmat_matvec, SBBmat_matvec, SBBmat_matvec3D, Biharmonic_matvec, Biharmonic_matvec3D, Tridiagonal_matvec, Tridiagonal_matvec3D, Pentadiagonal_matvec, Pentadiagonal_matvec3D, CBD_matvec3D, CBD_matvec
from scipy.sparse import diags

pi, zeros, ones, array = np.pi, np.zeros, np.ones, np.array
float, complex = np.float64, np.complex128

class BaseMatrix(object):

    def __init__(self):
        self.return_array = None
    
    def matvec(self, v):
        pass
    
    def diags(self):
        pass
        
    def get_return_array(self, v):        
        if self.return_array is None:
            self.return_array = np.zeros(v.shape, dtype=v.dtype)
        else:
            if not self.return_array.shape == v.shape:
                self.return_array = np.zeros(v.shape, dtype=v.dtype)
        self.return_array[:] = 0
        return self.return_array
    
# Mass matrices    
class BNDmat(BaseMatrix):
    """Matrix for inner product (p, phi)_w = BNDmat * p_hat
    
    where p_hat is a vector of coefficients for a Shen Neumann basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, quad, **kwargs):
        BaseMatrix.__init__(self)
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
        c = self.get_return_array(v)
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
    

class BDNmat(BaseMatrix):
    """Matrix for inner product (p, phi_N)_w = BDNmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi_N is a Shen Neumann basis.
    """

    def __init__(self, K, quad, **kwargs):
        BaseMatrix.__init__(self)
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
        c = self.get_return_array(v)
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
    
class BTTmat(BaseMatrix):
    """Matrix for inner product (p, T)_w = BTTmat * p_hat
    
    where p_hat is a vector of coefficients for a Chebyshev basis
    and T is a Chebyshev basis.
    """

    def __init__(self, K, quad, **kwargs):
        BaseMatrix.__init__(self)
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0], K.shape[0])
        N = shape[0]
        ck = ones(N, int)
        ck[0] = 2
        if quad == "GL": ck[-1] = 2        
        self.dd = pi/2*ck
    
    def matvec(self, v):
        N = self.shape[0]
        c = self.get_return_array(v)
        if len(v.shape) > 1:
            c[:] = self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[:].shape)*v[:]  
        else:
            c[:] = self.dd*v[:]
        return c
    
    def diags(self):
        return diags([self.dd], [0], shape=self.shape)

class BNNmat(BaseMatrix):
    """Matrix for inner product (p, phi_N)_w = BNNmat * p_hat
    
    where p_hat is a vector of coefficients for a Shen Neumann basis
    and phi_N is a Shen Neumann basis.
    """

    def __init__(self, K, quad, **kwargs):
        BaseMatrix.__init__(self)
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
        c = self.get_return_array(v)
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

class BDDmat(BaseMatrix):
    """Matrix for inner product (u, phi)_w = BDDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, quad, **kwargs):
        BaseMatrix.__init__(self)
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
        c = self.get_return_array(v)
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

class BDTmat(BaseMatrix):
    """Matrix for inner product (u, phi)_w = BDTmat * u_hat
    
    where u_hat is a vector of coefficients for a Chebyshev basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, quad, **kwargs):
        BaseMatrix.__init__(self)
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
        c = self.get_return_array(v)
        if len(v.shape) > 1:
            c[:N]  = self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[:N].shape)*v[:N]
            c[:N] += self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[2:].shape)*v[2:]
        else:
            c[:N]  = self.dd*v[:N]
            c[:N] += self.ud*v[2:]
        return c
    
    def diags(self):
        return diags([self.dd, self.ud], [0, 2], shape=self.shape)

class BTDmat(BaseMatrix):
    """Matrix for inner product (u, T)_w = BTDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and T is a Chebyshev basis.
    """

    def __init__(self, K, quad, **kwargs):
        BaseMatrix.__init__(self)
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
        c = self.get_return_array(v)
        if len(v.shape) > 1:
            c[:N]  = self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[:N].shape)*v[:N]
            c[2:] += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[:N].shape)*v[:N]
        else:
            c[:N]  = self.dd*v[:N]
            c[2:] += self.ld*v[:N]
        return c
    
    def diags(self):
        return diags([self.ld, self.dd], [-2, 0], shape=self.shape)

class BTNmat(BaseMatrix):
    """Matrix for inner product (u, T)_w = BTNmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Neumann basis
    and T is a Chebyshev basis.
    """

    def __init__(self, K, quad, **kwargs):
        BaseMatrix.__init__(self)
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0], K.shape[0]-3)
        N = shape[0]-2
        ck = ones(shape[0], int)
        if quad == "GL": ck[-1] = 2        
        self.dd = pi/2
        self.ld = -pi/2*ck[3:]*((K[3:]-2)/K[3:])**2
    
    def matvec(self, v):
        N = self.shape[0]
        c = self.get_return_array(v)
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

class BBBmat(BaseMatrix):
    
    def __init__(self, K, quad):
        BaseMatrix.__init__(self)
        N = K.shape[0]-4
        self.shape = (N, N)
        ck = ones(N)
        ckp = ones(N)
        ck[0] = 2
        if quad == "GL": ckp[-1] = 2
        k = K[:N].astype(float)
        self.dd = (ck + 4*((k+2)/(k+3))**2 + ckp*((k+1)/(k+3))**2)*pi/2.
        self.ud = -((k[:-2]+2)/(k[:-2]+3) + (k[:-2]+4)*(k[:-2]+1)/((k[:-2]+5)*(k[:-2]+3)))*pi
        self.uud = (k[:-4]+1)/(k[:-4]+3)*pi/2
        self.ld = self.ud
        self.lld = self.uud
        
    def matvec(self, v):
        c = self.get_return_array(v)
        N = self.shape[0]
        if len(v.shape) > 1:
            #vv = v[:-4]
            #c[:N] = self.dd.repeat(array(v.shape[1:]).prod()).reshape(vv.shape) * vv[:]
            #c[:N-2] += self.ud.repeat(array(v.shape[1:]).prod()).reshape(vv[2:].shape) * vv[2:]
            #c[:N-4] += self.uud.repeat(array(v.shape[1:]).prod()).reshape(vv[4:].shape) * vv[4:]
            #c[2:N]  += self.ld.repeat(array(v.shape[1:]).prod()).reshape(vv[:-2].shape) * vv[:-2]
            #c[4:N]  += self.lld.repeat(array(v.shape[1:]).prod()).reshape(vv[:-4].shape) * vv[:-4]
            Pentadiagonal_matvec3D(v, c, self.lld, self.ld, self.dd, self.ud, self.uud)
            
        else:
            #vv = v[:-4]
            #c[:N] = self.dd * vv[:]
            #c[:N-2] += self.ud * vv[2:]
            #c[:N-4] += self.uud * vv[4:]
            #c[2:N]  += self.ld * vv[:-2]
            #c[4:N]  += self.lld * vv[:-4]
            Pentadiagonal_matvec(v, c, self.lld, self.ld, self.dd, self.ud, self.uud)

        return c
    
    def diags(self):
        return diags([self.lld, self.ld, self.dd, self.ud, self.uud], range(-4, 6, 2), shape=self.shape)

class BBDmat(BaseMatrix):
    
    def __init__(self, K, quad):
        BaseMatrix.__init__(self)
        N = K.shape[0]-4
        self.shape = (N, N+2)
        ck = ones(N)
        ckp = ones(N)
        ck[0] = 2
        if quad == "GL": ckp[-1] = 2
        k = K[:N].astype(float)
        a = 2*(k+2)/(k+3)
        b = (k[:N]+1)/(k[:N]+3)
        self.dd = (ck + a)*pi/2.
        self.ld = -pi/2
        self.ud = -(a+b*ckp)*pi/2
        self.uud = b[:-2]*pi/2
        
    def matvec(self, v):
        c = self.get_return_array(v)
        N = self.shape[0]
        if len(v.shape) > 1:
            vv = v[:-2]
            c[:N] = self.dd.repeat(array(v.shape[1:]).prod()).reshape(vv[:-2].shape) * vv[:-2]
            c[:N] += self.ud.repeat(array(v.shape[1:]).prod()).reshape(vv[2:].shape) * vv[2:]
            c[:N-2] += self.uud.repeat(array(v.shape[1:]).prod()).reshape(vv[4:].shape) * vv[4:]
            c[2:N]  += self.ld * vv[:-4]
            
        else:
            vv = v[:-2]
            c[:N] = self.dd * vv[:-2]
            c[:N] += self.ud * vv[2:]
            c[:N-2] += self.uud * vv[4:]
            c[2:N]  += self.ld * vv[:-4]
        return c
    
    def diags(self):
        return diags([self.ld, self.dd, self.ud, self.uud], range(-2, 5, 2), shape=self.shape)
    
# Derivative matrices
class CDNmat(BaseMatrix):
    """Matrix for inner product (p', phi)_w = CDNmat * p_hat
    
    where p_hat is a vector of coefficients for a Shen Neumann basis
    and phi is a Shen Dirichlet basis.
    """
    
    def __init__(self, K, **kwargs):
        BaseMatrix.__init__(self)
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-3)
        N = shape[0]
        self.ud = (K[:(N-1)]+1)*pi
        self.ld = -((K[2:N]-1)/(K[2:N]+1))**2*(K[2:N]+1)*pi
        
    def matvec(self, v):
        N = self.shape[0]
        c = self.get_return_array(v)
        if len(v.shape) > 1:
            #c[:(N-1)] = self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[1:N].shape)*v[1:N]
            #c[2:N]   += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[1:(N-1)].shape)*v[1:(N-1)]
            CDNmat_matvec(self.ud, self.ld, v, c)
        else:
            c[:(N-1)] = self.ud*v[1:N]
            c[2:N]   += self.ld*v[1:(N-1)]
        return c

class CDDmat(BaseMatrix):
    """Matrix for inner product (u', phi) = (phi', phi) u_hat =  CDDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, **kwargs):
        BaseMatrix.__init__(self)
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-2)
        N = shape[0]
        self.ud = (K[:(N-1)]+1)*pi
        self.ld = -(K[1:N]+1)*pi
        
    def matvec(self, v):
        N = self.shape[0]
        c = self.get_return_array(v)
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

class CNDmat(BaseMatrix):
    """Matrix for inner product (u', phi_N) = (phi', phi_N) u_hat =  CNDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi_N is a Shen Neumann basis.
    """

    def __init__(self, K, **kwargs):
        BaseMatrix.__init__(self)
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-3, K.shape[0]-2)
        N = shape[1]
        self.ld = -(K[1:N]+1)*pi
        self.ud = [-(2-K[1:(N-1)]**2/(K[1:(N-1)]+2)**2*(K[1:(N-1)]+3))*pi]
        for i in range(3, N-1, 2):
            self.ud.append(-(1-K[1:(N-i)]**2/(K[1:(N-i)]+2)**2)*2*pi)    

    def matvec(self, v):
        N = self.shape[1]
        c = self.get_return_array(v)
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

class CTDmat(BaseMatrix):
    """Matrix for inner product (u', T) = (phi', T) u_hat =  CTDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and T is a Chebyshev basis.
    """

    def __init__(self, K, **kwargs):
        BaseMatrix.__init__(self)
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0], K.shape[0]-2)
        N = shape[0]
        self.ld = -(K[1:N]+1)*pi
        self.ud = [-2*pi]
        for i in range(3, N-2, 2):
            self.ud.append(-2*pi)    

    def matvec(self, v):
        N = self.shape[1]
        c = self.get_return_array(v)
        #c = np.zeros(v.shape, dtype=v.dtype)
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
    
class CDTmat(BaseMatrix):
    """Matrix for inner product (p', phi) = (T', phi) p_hat = CDTmat * p_hat
    
    where p_hat is a vector of coefficients for a Chebyshev basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, **kwargs):
        BaseMatrix.__init__(self)
        assert len(K.shape) == 1
        self.shape  = (K.shape[0]-2, K.shape[0])
        N = self.shape[0]
        self.ld = []
        self.ud = pi*(K[:N]+1)

    def matvec(self, v):
        N = self.shape[0]
        c = self.get_return_array(v)
        if len(v.shape) > 1:
            c[:N] = self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[1:-1].shape)*v[1:-1]
        else:
            c[:N] = self.ud*v[1:-1]
        return c        

    def diags(self):
        return diags(self.ud, [1], shape=self.shape)
    
class CBDmat(BaseMatrix):
    """Matrix for inner product (u', phi) = (phi', phi) u_hat =  CBDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Biharmonic basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, **kwargs):
        BaseMatrix.__init__(self)
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-4, K.shape[0]-2)
        N = shape[0]
        self.ld = -(K[1:N]+1)*pi
        self.ud = 2*(K[:N+1]+1)*pi     #  N+1 because of diags.
        self.uud = -(K[:N-1]+1)*pi

    def matvec(self, v):
        N1, N2 = self.shape
        c = self.get_return_array(v)
        if len(v.shape) > 1:
            #c[1:N1] = self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[:N2-3].shape)*v[:N2-3]
            #c[:N1] += self.ud[:N1].repeat(array(v.shape[1:]).prod()).reshape(v[1:N2-1].shape)*v[1:N2-1]
            #c[:N1-1]+= self.uud.repeat(array(v.shape[1:]).prod()).reshape(v[3:N2].shape)*v[3:N2]
            CBD_matvec3D(v, c, self.ld, self.ud, self.uud)
        else:
            #c[1:N1] = self.ld * v[:N2-3]
            #c[:N1] += self.ud[:N1] * v[1:N2-1]
            #c[:N1-1] += self.uud * v[3:N2]
            CBD_matvec(v, c, self.ld, self.ud, self.uud)
        return c

    def diags(self):
        return diags([self.ld, self.ud, self.uud], [-1, 1, 3], shape=self.shape)
    

class CDBmat(BaseMatrix):
    """Matrix for inner product (u', phi) = (phi', phi) u_hat =  CDBmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi is a Shen Biharmonic basis.
    """

    def __init__(self, K, **kwargs):
        BaseMatrix.__init__(self)
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0]-2, K.shape[0]-4)
        N = shape[0]
        self.lld = (K[3:N]-2)*(K[3:N]+1)/K[3:N]*pi
        self.ld = -2*(K[1:N]+1)**2/(K[1:N]+2)*pi
        self.ud = (K[:N-3]+1)*pi

    def matvec(self, v):
        N, M = self.shape
        c = self.get_return_array(v)
        if len(v.shape) > 1:
            c[3:N] = self.lld.repeat(array(v.shape[1:]).prod()).reshape(v[:M-1].shape) * v[:M-1]
            c[1:N-1] += self.ld[:M].repeat(array(v.shape[1:]).prod()).reshape(v[:M].shape) * v[:M]
            c[:N-3] += self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[1:M].shape) * v[1:M]
            
        else:
            c[3:N] = self.lld * v[:M-1]
            c[1:N-1] += self.ld[:M] * v[:M]
            c[:N-3] += self.ud * v[1:M]
        return c

    def diags(self):
        return diags([self.lld, self.ld, self.ud], [-3, -1, 1], shape=self.shape)

    
class ABBmat(BaseMatrix):
    """Matrix for inner product (u'', phi) = (phi'', phi) u_hat =  ABBmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Biharmonic basis
    and phi is a Shen Biharmonic basis.
    """

    def __init__(self, K):
        BaseMatrix.__init__(self)
        N = K.shape[0]-4
        self.shape = (N, N)
        ki = K[:N]
        k = K[:N].astype(float)
        i = -4*(ki+1)*(k+2)**2
        self.dd = i * pi / (k+3.)
        i = 2*(ki[:-2]+1)*(ki[:-2]+2)
        self.ud = i*pi
        i = 2*(ki[2:]-1)*(ki[2:]+2)
        self.ld = i * pi
        #self.dd = -4*(k+1)/(k+3)*(k+2)**2*pi
        #self.ud = 2*(k[:-2]+1)*(k[:-2]+2)*pi
        #self.ld = 2*(k[2:]-1)*(k[2:]+2)*pi
        
    def matvec(self, v):
        N = self.shape[0]
        c = self.get_return_array(v)
        if len(v.shape) > 1:
            #c[:N] = self.dd.repeat(array(v.shape[1:]).prod()).reshape(v[:N].shape) * v[:N]
            #c[:N-2] += self.ud.repeat(array(v.shape[1:]).prod()).reshape(v[2:N].shape) * v[2:N]
            #c[2:N] += self.ld.repeat(array(v.shape[1:]).prod()).reshape(v[:N-2].shape) * v[:N-2]
            Tridiagonal_matvec3D(v, c, self.ld, self.dd, self.ud)
            
        else:
            #c[:N] = self.dd * v[:N]
            #c[:N-2] += self.ud * v[2:N]
            #c[2:N] += self.ld * v[:N-2]
            Tridiagonal_matvec(v, c, self.ld, self.dd, self.ud)
        return c
        
    def diags(self):
        return diags([self.ld, self.dd, self.ud], [-2, 0, 2], shape=self.shape)
    
class ADDmat(BaseMatrix):
    """Matrix for inner product -(u'', phi) = -(phi'', phi) u_hat = ADDmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet basis
    and phi is a Shen Dirichlet basis.
    """

    def __init__(self, K, **kwargs):
        BaseMatrix.__init__(self)
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


class ANNmat(BaseMatrix):
    """Matrix for inner product -(u'', phi_N) = -(phi_N'', phi_N) p_hat = ANNmat * p_hat
    
    where u_hat is a vector of coefficients for a Shen Neumann basis
    and phi is a Shen Neumann basis.
    """

    def __init__(self, K, **kwargs):
        BaseMatrix.__init__(self)
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

class ATTmat(BaseMatrix):
    """Matrix for inner product -(u'', phi) = -(phi'', phi) p_hat = ATTmat * p_hat
    
    where p_hat is a vector of coefficients for a Chebyshev basis
    and phi is a Chebyshev basis.
    """

    def __init__(self, K, **kwargs):
        BaseMatrix.__init__(self)
        assert len(K.shape) == 1
        self.shape = shape = (K.shape[0], K.shape[0])
        N = shape[0]
        self.ud = []
        for j in range(2, N, 2):
            self.ud.append(np.array(K[j:]*(K[j:]**2-K[:-j]**2))*np.pi/2.)    

        self.ld = None
        
    def matvec(self, v):
        N = self.shape[0]
        c = np.zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            m = self.diags().toarray()
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    c[:N, i, j] = np.dot(m, v[:N])
        else:
            c[:N] = np.dot(self.diags().toarray(), v[:N])
        return c

    def diags(self):
        N = self.shape[0]
        return diags(self.ud, range(2, N, 2))

class SBBmat(BaseMatrix):
    """Matrix for inner product (u'''', phi) = (phi'''', phi) u_hat =  SBBmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Biharmonic basis
    and phi is a Shen Biharmonic basis.
    """
    
    def __init__(self, K):
        BaseMatrix.__init__(self)
        N = K.shape[0]-4
        self.shape = (N, N)
        k = K[:N].astype(float)
        ki = K[:N]
        i = 8*(ki+1)**2*(ki+2)*(ki+4)
        self.dd = i * pi
        #self.dd = 8.*(k+1.)**2*(k+2.)*(k+4.)*pi
        self.ud = []
        for j in range(2, N, 2):
            i = 8*(ki[:-j]+1)*(ki[:-j]+2)*(ki[:-j]*(ki[:-j]+4)+3*(ki[j:]+2)**2)
            self.ud.append(np.array(i*pi/(k[j:]+3)))
            #self.ud.append(np.array(8./(k[j:]+3.)*pi*(k[:-j]+1.)*(k[:-j]+2.)*(k[:-j]*(k[:-j]+4.)+3.*(k[j:]+2.)**2)))

        self.ld = None
        self.return_array = None
        
    def matvec(self, v):
        N = self.shape[0]
        c = self.get_return_array(v)
        if len(v.shape) > 1:
            SBBmat_matvec3D(v, c, self.dd)
        else:
            SBBmat_matvec(v, c, self.dd)
            #c[:N] = np.dot(self.diags().toarray(), v[:N])
        return c
            
    def diags(self):
        return diags([self.dd]+self.ud, range(0, self.shape[0], 2), shape=self.shape)

class BiharmonicCoeff(BaseMatrix):
    
    def __init__(self, K, a0, alfa, beta, quad="GL"):
        BaseMatrix.__init__(self)
        self.quad = quad
        self.N = K.shape[0]-4
        N = K.shape[0]-4
        self.shape = (N, N)
        self.S = SBBmat(K)
        self.B = BBBmat(K, self.quad)
        self.A = ABBmat(K)
        self.a0 = a0
        self.alfa = alfa
        self.beta = beta
                
    def matvec(self, v):
        N = self.shape[0]
        #c = np.zeros(v.shape, dtype=v.dtype)
        c = self.get_return_array(v)
        if len(v.shape) > 1:
            Biharmonic_matvec3D(v, c, self.a0, self.alfa, self.beta, self.S.dd, self.S.ud[0], 
                                self.S.ud[1], self.A.ld, self.A.dd, self.A.ud,
                                self.B.lld, self.B.ld, self.B.dd, self.B.ud, self.B.uud)
        else:
            Biharmonic_matvec(v, c, self.a0, self.alfa, self.beta, self.S.dd, self.S.ud[0], 
                                self.S.ud[1], self.A.ld, self.A.dd, self.A.ud,
                                self.B.lld, self.B.ld, self.B.dd, self.B.ud, self.B.uud)
        return c
    
    def diags(self):
        raise NotImplementedError
        #return diags([self.ldd, self.ld, self.dd]+self.ud, range(-4, self.shape()[0], 2), shape=self.shape())
    
    