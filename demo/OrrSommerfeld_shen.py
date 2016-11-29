"""
Solve the Orr-Sommerfeld eigenvalue problem

Using Shen's biharmonic basis

"""
from scipy.linalg import eig
#from numpy.linalg import eig
import numpy as np
from numpy.linalg import inv
from numpy.polynomial import chebyshev as n_cheb
from spectralDNS.shen.shentransform import ShenBiharmonicBasis, ShenDirichletBasis, \
    ChebyshevTransform
from spectralDNS.shen.Matrices import CDBmat, ABBmat, SBBmat, BBBmat
import six
import warnings
np.seterr(divide='ignore')

try:
    from matplotlib import pyplot as plt
    from pylab import find

except ImportError:
    warnings.warn("matplotlib not installed")


class OrrSommerfeld(object):
    def __init__(self,**kwargs):
        self.par={'alfa':1.,
                  'Re':8000.,
                  'N':80,
                  'quad': 'GC'}
        self.par.update(**kwargs)
        [setattr(self, name, val) for name, val in six.iteritems(self.par)]
        self.P4 = np.zeros(0)
        self.T4x = np.zeros(0)
        self.SB = None

    def interp(self, y, eigval=1, same_mesh=False, verbose=False):
        """Interpolate solution eigenvector and it's derivative onto y
        """
        N = self.N
        nx, eigval = self.get_eigval(eigval, verbose)
        phi_hat = np.zeros(N, np.complex)
        phi_hat[:-4] = np.squeeze(self.eigvectors[:, nx])
        if same_mesh:            
            phi = np.zeros_like(phi_hat)
            dphidy = np.zeros_like(phi_hat)
            if self.SB is None:
                self.SB = ShenBiharmonicBasis(quad=self.quad)
                self.SD = ShenDirichletBasis(quad=self.quad)
                self.CDB = CDBmat(np.arange(N).astype(np.float))

            phi = self.SB.ifst(phi_hat, phi)
            dphidy_hat = self.CDB.matvec(phi_hat)
            dphidy_hat = self.SD.Solver(dphidy_hat)
            dphidy = self.SD.ifst(dphidy_hat, dphidy)
            
        else:
            # Recompute interpolation matrices if necessary
            if not len(self.P4) == len(y):
                k = np.arange(N).astype(np.float)[:-4] # Wavenumbers
                V = n_cheb.chebvander(y, N-1)
                P4 = np.zeros((len(y), N))
                P4[:, :-4] = V[:, :-4] - (2*(k+2)/(k+3))*V[:, 2:-2] + ((k+1)/(k+3))*V[:, 4:]
                D = np.zeros((N, N))
                D[:-1, :] = n_cheb.chebder(np.eye(N), 1)
                T1 = np.dot(V, D)
                T4x = np.zeros((len(y), N))
                T4x[:, :-4] = T1[:, :-4] - (2*(k+2)/(k+3))*T1[:, 2:-2] + ((k+1)/(k+3))*T1[:, 4:]
                self.P4 = P4
                self.T4x = T4x
            phi = np.dot(self.P4, phi_hat)
            dphidy = np.dot(self.T4x, phi_hat)

        return phi, dphidy

    def assemble(self):
        N = self.N
        CT = ChebyshevTransform(quad=self.quad)
        x, w = self.x, self.w = CT.points_and_weights(N)
        V = n_cheb.chebvander(x, N-1)
        D2 = np.zeros((N, N))
        D2[:-2, :] = n_cheb.chebder(np.eye(N), 2)
        D4 = np.zeros((N, N))
        D4[:-4,:] = n_cheb.chebder(np.eye(N), 4)

        # Matrices of coefficients for second and fourth derivatives
        T2 = np.dot(V, D2)
        T4 = np.dot(V, D4)

        # Trial function
        k = np.arange(N).astype(np.float)[:-4] # Wavenumbers
        P4 = np.zeros((N, N))
        P4[:, :-4] = V[:, :-4] - (2*(k+2)/(k+3))*V[:, 2:-2] + ((k+1)/(k+3))*V[:, 4:]

        # Second derivatives 
        T2x = np.zeros((N, N))
        T2x[:, :-4] = T2[:, :-4] - (2*(k+2)/(k+3))*T2[:, 2:-2] + ((k+1)/(k+3))*T2[:, 4:]

        # Fourth derivatives
        T4x = np.zeros((N, N))
        T4x[:, :-4] = T4[:, :-4] - (2*(k+2)/(k+3))*T4[:, 2:-2] + ((k+1)/(k+3))*T4[:, 4:]

        # (u'', v)
        #K = np.dot(w*P4.T, T2x)
        SB = ShenBiharmonicBasis(quad=self.quad)
        K = np.zeros((N, N))
        K = SB.fastShenScalar(T2x, K)
        #K[:-4, :-4] = ABBmat(np.arange(N).astype(np.float)).diags().toarray()

        # ((1-x**2)u, v)
        xx = (1-x**2).repeat(N).reshape((N, N))
        #K1 = np.dot(w*P4.T, xx*P4)  # Alternative: K1 = np.dot(w*P4.T, ((1-x**2)*P4.T).T)
        K1 = np.zeros((N, N))
        K1 = SB.fastShenScalar(xx*P4, K1)
        #K1 = extract_diagonal_matrix(K1) # For improved roundoff
        
        # ((1-x**2)u'', v)
        #K2 = np.dot(w*P4.T, xx*T2x)
        K2 = np.zeros((N, N))
        K2 = SB.fastShenScalar(xx*T2x, K2)
        #K2 = extract_diagonal_matrix(K2) # For improved roundoff
        
        # (u'''', v)
        #Q = np.dot(w*P4.T, T4x)
        Q = np.zeros((self.N, self.N))
        Q[:-4, :-4] = SBBmat(np.arange(N).astype(np.float)).diags().toarray()

        # (u, v)
        #M = np.dot(w*P4.T, P4)
        M = np.zeros((self.N, self.N))
        M[:-4, :-4] = BBBmat(np.arange(N).astype(np.float), self.quad).diags().toarray()
        
        Re = self.Re
        a = self.alfa
        self.B = -Re*a*1j*(K-a**2*M)
        self.A = Q-2*a**2*K+a**4*M - 2*a*Re*1j*M - 1j*a*Re*(K2-a**2*K1)

    def solve(self, verbose=False):
        """Solve the Orr-Sommerfeld eigenvalue problem
        """
        if verbose:
            print('Solving the Orr-Sommerfeld eigenvalue problem...')
            print('Re = '+str(self.par['Re'])+' and alfa = '+str(self.par['alfa']))
        self.assemble()
        self.eigvals, self.eigvectors = eig(self.A[:-4, :-4],self.B[:-4, :-4])
        #self.eigvals, self.eigvectors = eig(np.dot(inv(self.B[:-4, :-4]), self.A[:-4, :-4]))

    def get_eigval(self, nx, verbose=False):
        """Get the chosen eigenvalue
        
        Args:
            nx       The chosen eigenvalue. nx=1 corresponds to the one with the
                     largest imaginary part, nx=2 the second largest etc.

            verbose  Print the value of the chosen eigenvalue

        """
        indices = np.argsort(np.imag(self.eigvals))
        indi = indices[-np.array(nx)]
        eigval = self.eigval = self.eigvals[indi]
        if verbose:
            ev = list(eigval) if np.ndim(eigval) else [eigval]
            indi = list(indi) if np.ndim(indi) else [indi]
            for i, (e, v) in enumerate(zip(ev, indi)):
                print('Eigenvalue {} ({}) = {:2.16e}'.format(i+1, v, e))
        return indi, eigval


def extract_diagonal_matrix(M, tol=1e-8):
    """Return matrix with essentially zero diagonals nulled out
    """
    Mc = np.zeros_like(M)
    du = []
    dl = []
    dd = M.diagonal()
    for i in range(1, M.shape[1]):
        u = M.diagonal(i)
        l = M.diagonal(-i)
        if abs(u).max() > tol:
            du.append((i, u))
        if abs(l).max() > tol:
            dl.append((i, l))
    
    np.fill_diagonal(Mc, dd)
    for (i, ud) in du:
        np.fill_diagonal(Mc[:, i:], ud)
    for (i, ld) in dl:
        np.fill_diagonal(Mc[i:, :], ld)
    return Mc


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Orr Sommerfeld parameters')
    parser.add_argument('--N', type=int, default=120,
                        help='Number of discretization points')
    parser.add_argument('--Re', default=8000.0, type=float,
                        help='Reynolds number')
    parser.add_argument('--alfa', default=1.0, type=float,
                        help='Parameter')
    parser.add_argument('--quad', default='GC', type=str, choices=('GC', 'GL'),
                        help='Discretization points: GC: Gauss-Chebyshev, GL: Gauss-Lobatto')
    parser.add_argument('--plot', dest='plot', action='store_true', help='Plot eigenvalues')
    parser.set_defaults(plot=False)
    args = parser.parse_args()
    #z = OrrSommerfeld(N=120, Re=5772.2219, alfa=1.02056)
    z = OrrSommerfeld(**vars(args))
    z.solve()
    if args.Re == 8000.0 and args.alfa == 1.0 and args.N > 80:
        d = z.get_eigval(1)
        assert abs(d[1] - (0.24707506017508621+0.0026644103710965817j)) < 1e-12
    
    if args.plot:
        plt.figure()
        ev = z.eigvals*z.alfa
        plt.plot(ev.imag, ev.real, 'o')
        plt.axis([-10, 0.1, 0, 1])
        plt.show()
