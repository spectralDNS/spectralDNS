"""
Solve the Orr-Sommerfeld eigenvalue problem

FIXME Should use Shen basis for fourth order problem

"""
from scipy.linalg import eig
from numpy import ones, cos, arange, pi, dot, eye, real, imag, resize, transpose, \
    float, newaxis, sum, abs, max, complex, linspace, argmax, argmin, zeros, squeeze, \
    seterr, array, hstack, argpartition
from numpy.linalg import inv
from scipy.special import orthogonal
from numpy.polynomial import chebyshev as n_cheb
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
import warnings
seterr(divide='ignore')


try:
    from pylab import find, plot, figure, show, axis

except ImportError:
    warnings.warn("matplotlib not installed")

def cheb_derivative_matrix(N):
    x = arange(N)
    y = -cos(x/(N - 1.)*pi)
    c = ones(N, float)
    c[0] = 2.
    c[-1] = 2.
    c = c*(-1.)**x
    X = transpose(resize(y, (N, N)))
    dX = X - transpose(X)
    D = dot(c[:,newaxis], 1./c[newaxis, :])/(dX + eye(N))
    D = D - eye(N)*sum(transpose(D), 0)
    return D

class OrrSommerfeld(object):
    def __init__(self,**kwargs):
        self.par={'alfa':1.,
                  'Re':8000.,
                  'N':20,
                  'eigval':1,
                  'order':None}
        self.par.update(**kwargs)
        [setattr(self, name, val) for name, val in self.par.iteritems()]
        if(self.order == None):
            self.order = self.N
        self.y = -cos(arange(self.N)*1./(self.N - 1.)*pi)
        #self.D=derivative_matrix(self.y,self.order)
        self.D = cheb_derivative_matrix(self.N)
        self.D2 = dot(self.D, self.D)
        self.D3 = dot(self.D, self.D2)
        self.D4 = dot(self.D2, self.D2)
        self.yI = eye(self.N)*self.y
        self.I = eye(self.N)
        self.OS = self.D2-self.alfa**2*self.I
        self.b = eye(self.N)*2.
        self.II = (self.I - self.yI**2)
        self.S = eye(self.N)
        self.S = self.S*1./(1. - hstack((0, self.y[1:-1], 0))**2)
        self.S[0, 0]=0
        self.S[-1, -1]=0
        print 'Solving the Orr-Sommerfeld eigenvalue problem...'
        print 'Re = ', self.par['Re'], ' and alfa = ', self.par['alfa']
        self.eigvals, self.eigvectors = self.solve()
        self.nx = [argpartition(imag(self.eigvals), -1)[-self.par['eigval']]]
        self.eigval = self.eigvals[self.nx][0]
        #print 'Least stable eigenvalue = ', self.eigval
        print 'Eigenvalue = ', self.eigval
        self.create_interpolation_arrays()
        
    def create_interpolation_arrays(self):
        # Normalize and create necessary arrays for interpolation
        self.phi = zeros(self.N, complex)
        self.phi[1: -1] = squeeze(self.eigvectors[:, self.nx]) 
        self.phi = self.phi/max(abs(real(self.phi)))
        # Compute and store derivative of phi
        self.dphidy = dot(self.D, self.phi)
        self.dphidy[:self.N:self.N-1] = 0.
        #self.dphidy[-1]=0.
        # Create interpolation vector and barycentric weights for later use
        self.f = zeros(self.N, float)
        self.w = (-1.)**(arange(self.N))
        self.w[:self.N:(self.N-1)]/=2.

    def assemble(self):
        """
        Assemble matrixes for the eigenvalue problem: Au=LBu
        
        Set basis function 
        
        p(y)=(1-y**2)*q(y)
        
        where q is a polynomial of degree <= N, q(+-1)=0. Then p is a polynomial of
        degree <= N+2, with p(+-1)=0 and p'(+-1)=0. The fourth derivative of p is
        
        p^(4)=(1-y**2)q^(4)-8*y*q^(3)-12*q^(2)
        
        and the fourth derivative matrix P4 becomes
        
        P4 = (1-y**2)*D^4-8*y*D^3-12*D^2
        
        where D is the regular derivative matrix of q
        
        """
        P4 = dot(dot(self.II, self.D4) - 8.*dot(self.yI, self.D3) - 12.*self.D2, self.S)
        #P2=dot(dot(self.II,self.D2)-4.*dot(self.yI,self.D)-2.*self.I,self.S)
        P2 = self.D2
        OS = P2 - self.alfa**2*self.I
        self.Ai = P4 - 2.*self.alfa**2*P2 + self.alfa**4*self.I \
                  - 1j*self.alfa*self.Re*(dot(self.II, OS) + 2*self.I)
        self.A = self.Ai[1:-1, 1:-1]
        self.Bi = -1j*self.alfa*self.Re*OS
        self.B = self.Bi[1:-1, 1:-1]	

    def solve(self):
        self.assemble()
        #return eig(self.A,self.B)
        return eig(dot(inv(self.B), self.A))
        
    def interp(self, x):
        """Barycentric interpolation from self.y to a point x."""
        x2 = x - self.y
        if any(abs(x2) < 1.e-15):
            self.f[:] = 0.
            self.f[argmin(abs(x2))] = 1.
        else:
            ss = sum(self.w/x2)
            self.f[:] = self.w/x2/ss


class Amat(LinearOperator):
    """Matrix for inner product -(u'', (phi*w)'') = -(phi'', (phi*w)'') u_hat = Amat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet/Neumann basis
    and phi is a Shen Dirichlet/Neumann basis.
    """

    def __init__(self, K, **kwargs):
        assert len(K.shape) == 1
        shape = (K.shape[0]-4, K.shape[0]-4)
        N = shape[0]
        LinearOperator.__init__(self, shape, None, **kwargs)
        self.dd = 8*pi*(K[:N]+1)**2*(K[:N]+2)*(K[:N]+4)   
        self.ud = []
        for i in range(2, N, 2):
            self.ud.append(8*pi*(K[:-(i+4)]+1)*(K[:-(i+4)]+2)*(K[:-(i+4)]*(K[:-(i+4)]+4)+3*(arange(i, N)+2)**2)/((arange(i, N)+3)))

        self.ld = None
        
    def matvec(self, v):
        raise NotImplementedError

    def diags(self):
        N = self.shape[0]
        return diags([self.dd] + self.ud, range(0, N, 2))

class Bmat(LinearOperator):
    """Matrix for inner product (u, phi) = (phi_j, phi_k) u_hat_j = Bmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet/Neumann basis
    and phi is a Shen Dirichlet/Neumann basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        shape = (K.shape[0]-4, K.shape[0]-4)
        N = shape[0]
        LinearOperator.__init__(self, shape, None, **kwargs)
        ck = ones(K.shape)
        if quad == "GC": ck[N-1] = 2
        self.dd = pi/2*(ck[:-4] + 4*(K[:N]+2)**2/(K[:N]+3)**2 + (K[:N]+1)**2/(K[:N]+3)**2)           
        self.ud = [-pi*((K[:N-2]+2)/(K[:N-2]+3) + (K[:N-2]+4)/(K[:N-2]+5)*(K[:N-2]+1)/(K[:N-2]+3)),
                   pi/2*(K[:N-4]+1)/(K[:N-4]+3)]
        
        self.ld = [pi/2*(K[:N-4]+1)/(K[:N-4]+3),
                   -pi*((K[:N-2]+2)/(K[:N-2]+3) + (K[:N-2]+4)/(K[:N-2]+5)*(K[:N-2]+1)/(K[:N-2]+3))]
        
    def matvec(self, v):
        raise NotImplementedError

    def diags(self):
        N = self.shape[0]
        return diags(self.ld + [self.dd] + self.ud, [-4, -2, 0, 2, 4])


class Cmat(LinearOperator):
    """Matrix for inner product (u', (phi*w)') = (phi'_j, (phi_k*w)') u_hat_j = Cmat * u_hat
    
    where u_hat is a vector of coefficients for a Shen Dirichlet/Neumann basis
    and phi is a Shen Dirichlet/Neumann basis.
    """

    def __init__(self, K, quad, **kwargs):
        assert len(K.shape) == 1
        shape = (K.shape[0]-4, K.shape[0]-4)
        N = shape[0]
        LinearOperator.__init__(self, shape, None, **kwargs)
        ck = ones(K.shape)
        if quad == "GC": ck[N-1] = 2
        self.dd = -4*pi*(K[:N]+1)/(K[:N]+3)*(K[:N]+2)**2
        self.ud = [2*pi*(K[:N-2]+1)*(K[:N-2]+2)]        
        self.ld = [2*pi*(K[2:N]-1)*(K[2:N]+2)]
        
    def matvec(self, v):
        raise NotImplementedError

    def diags(self):
        N = self.shape[0]
        return diags(self.ld + [self.dd] + self.ud, [-2, 0, 2])

## Use sympy to compute a rhs, given an analytical solution
#from numpy import *
#from sympy import chebyshevt, Symbol, sin, cos, lambdify, sqrt as Sqrt
#import scipy.sparse.linalg as la

#x = Symbol("x")
#u = (1-x**2)*sin(2*pi*x)
#f = u.diff(x, 4) - u.diff(x, 2) + u
#N = 40
#k = arange(N).astype(float)
#Am = Amat(k)
#Bm = Bmat(k, "GC")
#Cm = Cmat(k, "GC")

#points = n_cheb.chebpts2(N)[::-1]
#weights = zeros((N))+pi/(N-1)
#weights[0] /= 2
#weights[-1] /= 2

#fj = array([f.subs(x, j) for j in points], dtype=float)
#V = (n_cheb.chebvander(points, N-5).T 
     #- (2*(k[:N-4]+2)/(k[:N-4]+3))[:, newaxis]*n_cheb.chebvander(points, N-3)[:, 2:].T
     #+ ((k[:N-4]+1)/(k[:N-4]+3))[:, newaxis]*n_cheb.chebvander(points, N-1)[:, 4:].T)

#fk = dot(V, fj*weights)
#uk = la.spsolve(Am.diags()-Cm.diags()+Bm.diags(), fk)
#uj = dot(V.T, uk)
#uq = array([u.subs(x, j) for j in points], dtype=float)



class OrrSommerfeldShen(object):
    def __init__(self,**kwargs):
        self.par={'alfa':1.,
                  'Re':8000.,
                  'N':20,
                  'order':None}
        self.par.update(**kwargs)
        [setattr(self, name, val) for name, val in self.par.iteritems()]
        
    def assemble(self):
        k = arange(self.N).astype(float)
        Am = Amat(k)
        Bm = Bmat(k, "GC")
        Cm = Cmat(k, "GC")
        weights = zeros((self.N))+pi/(self.N-1)
        weights[0] /= 2
        weights[-1] /= 2
        A = Am.diags().toarray()
        B = Bm.diags().toarray()
        C = Cm.diags().toarray()
        points = n_cheb.chebpts2(self.N)[::-1]
        self.B = -1j*self.alfa*self.Re*(C+self.alfa**2*B)
        self.A = (A + (2*self.alfa**2+1j*self.alfa*self.Re*(1-points**2))*C + 
                  (self.alfa**4+1j*self.alfa**3*self.Re*(1-points**2)-2*1j*self.alfa*self.Re)*B)

    def solve(self):
        self.assemble()
        #return eig(self.A,self.B)
        return eig(dot(inv(self.B), self.A))


def ploteig(an,ev,nd=20):
    N = len(an)
    x = linspace(-1, 1, nd)
    v = zeros(nd, float)
    for i, a in enumerate(an):
        T = orthogonal.chebyt(2*i)
        v[:] += a*T(x)
    plot(x, v, 'o-')
    return x, v

if __name__=='__main__':
    print 'Solving the Orr-Sommerfeld eigenvalue problem...'
    # This needs to be solved with relatively high resolution
    z = OrrSommerfeld(N=60, Re=8000., alfa=1.0)
    sol = z.solve()
    nx = find(imag(sol[0]) > 0)
    eigval = sol[0][nx] # Eigenvalue of least stable mode
    eigv = sol[1][:, nx[-1]]
    print 'Least stable eigenvalue = ',eigval	
