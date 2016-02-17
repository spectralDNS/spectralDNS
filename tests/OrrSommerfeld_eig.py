"""
Solve the Orr-Sommerfeld eigenvalue problem

FIXME Should use Shen basis for fourth order problem

"""
from scipy.linalg import eig
from numpy import ones, cos, arange, pi, dot, eye, real, imag, resize, transpose, float, newaxis, sum, abs, max, complex, linspace, argmax, argmin, zeros, squeeze, seterr, array, hstack
from numpy.linalg import inv
from numpy.polynomial import chebyshev as n_cheb
seterr(divide='ignore')

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
        self.nx = [argmax(imag(self.eigvals))]
        self.eigval = self.eigvals[self.nx][0]
        print 'Least stable eigenvalue = ', self.eigval
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

