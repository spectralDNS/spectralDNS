"""
Solve the Orr-Sommerfeld eigenvalue problem

FIXME Should use Shen basis for fourth order problem

"""
from scipy.linalg import eig
#from numpy.linalg import eig
from numpy import ones, cos, arange, pi, dot, eye, real, imag, resize, transpose, \
    float, newaxis, sum, abs, max, complex, linspace, argmax, argmin, zeros, squeeze, \
    seterr, array, hstack, argsort, ndim
from numpy.linalg import inv
from numpy.polynomial import chebyshev as n_cheb
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
import six
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
                  'order':None}
        self.par.update(**kwargs)
        [setattr(self, name, val) for name, val in six.iteritems(self.par)]
        if(self.order == None):
            self.order = self.N
        self.y = -cos(arange(self.N)*1./(self.N - 1.)*pi)
        #self.y = -n_cheb.chebpts2(self.N)
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
        
    def get_eigval(self, nx, verbose=False):
        """Get the chosen eigenvalue
        
        Args:
            nx       The chosen eigenvalue. nx=1 corresponds to the one with the
                     largest imaginary part, nx=2 the second largest etc.

            verbose  Print the value of the chosen eigenvalue

        """
        indices = argsort(imag(self.eigvals))
        indi = indices[-array(nx)]
        eigval = self.eigvals[indi]
        if verbose:
            ev = list(eigval) if ndim(eigval) else [eigval]
            indi = list(indi) if ndim(indi) else [indi]
            for i, (e, v) in enumerate(zip(ev, indi)):
                print('Eigenvalue {} ({}) = {:2.12e}'.format(i+1, v, e))

        return indi, eigval
        
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

    def solve(self, verbose=False):
        if verbose:
            print('Solving the Orr-Sommerfeld eigenvalue problem...')
            print('Re = '+str(self.par['Re'])+' and alfa = '+str(self.par['alfa']))
        
        self.assemble()
        self.eigvals, self.eigvectors = eig(self.A,self.B)
        #self.eigvals, self.eigvectors = eig(dot(inv(self.B), self.A))

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
        
    def interp(self, x, nx, verbose=False):
        """Barycentric interpolation from self.y to a point x."""
        self.nx, self.eigval = self.get_eigval(nx, verbose)
        if not hasattr(self, 'phi'):
            self.create_interpolation_arrays()
        
        x2 = x - self.y
        if any(abs(x2) < 1.e-15):
            self.f[:] = 0.
            self.f[argmin(abs(x2))] = 1.
        else:
            ss = sum(self.w/x2)
            self.f[:] = self.w/x2/ss


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Orr Sommerfeld parameters')
    parser.add_argument('--N', type=int, default=120,
                        help='Number of discretization points')
    parser.add_argument('--Re', default=8000.0, type=float,
                        help='Reynolds number')
    parser.add_argument('--alfa', default=1.0, type=float,
                        help='Parameter')
    parser.add_argument('--plot', dest='plot', action='store_true', help='Plot eigenvalues')
    parser.set_defaults(plot=False)
    args = parser.parse_args()
    #z = OrrSommerfeld(N=120, Re=5772.2219, alfa=1.02056)
    z = OrrSommerfeld(**vars(args))
    z.solve()
    if args.plot:
        plt.figure()
        ev = z.eigvals*z.alfa
        plt.plot(ev.imag, ev.real, 'o')
        plt.axis([-10, 0.1, 0, 1])
        plt.show()
