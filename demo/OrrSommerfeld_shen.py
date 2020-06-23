"""
Solve the Orr-Sommerfeld eigenvalue problem

Using Shen's biharmonic basis

"""
import warnings
from scipy.linalg import eig
#from numpy.linalg import eig
#from numpy.linalg import inv
import numpy as np
import sympy as sp
from shenfun import FunctionSpace, Function, Dx
from shenfun.spectralbase import inner_product
from shenfun.matrixbase import extract_diagonal_matrix

np.seterr(divide='ignore')

#pylint: disable=no-member

try:
    from matplotlib import pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")

class OrrSommerfeld(object):
    def __init__(self, alfa=1., Re=8000., N=80, quad='GC', **kwargs):
        kwargs.update(dict(alfa=alfa, Re=Re, N=N, quad=quad))
        vars(self).update(kwargs)
        self.x, self.w = None, None

    def interp(self, y, eigvals, eigvectors, eigval=1, verbose=False):
        """Interpolate solution eigenvector and it's derivative onto y

        Parameters
        ----------
            y : array
                Interpolation points
            eigvals : array
                All computed eigenvalues
            eigvectors : array
                All computed eigenvectors
            eigval : int, optional
                The chosen eigenvalue, ranked with descending imaginary
                part. The largest imaginary part is 1, the second
                largest is 2, etc.
            verbose : bool, optional
                Print information or not
        """
        nx, eigval = self.get_eigval(eigval, eigvals, verbose)
        SB = FunctionSpace(self.N, 'C', bc='Biharmonic', quad=self.quad, dtype='D')
        phi_hat = Function(SB)
        phi_hat[:-4] = np.squeeze(eigvectors[:, nx])
        phi = phi_hat.eval(y)
        dphidy = Dx(phi_hat, 0, 1).eval(y)
        return eigval, phi, dphidy

    def assemble(self):
        N = self.N
        SB = FunctionSpace(N, 'C', bc='Biharmonic', quad=self.quad)
        SB.plan((N, N), 0, np.float, {})

        # (u'', v)
        K = inner_product((SB, 0), (SB, 2))

        # ((1-x**2)u, v)
        x = sp.symbols('x', real=True)
        K1 = inner_product((SB, 0), (SB, 0), measure=(1-x**2))

        # ((1-x**2)u'', v)
        K2 = inner_product((SB, 0), (SB, 2), measure=(1-x**2))

        # (u'''', v)
        Q = inner_product((SB, 0), (SB, 4))

        # (u, v)
        M = inner_product((SB, 0), (SB, 0))

        Re = self.Re
        a = self.alfa
        B = -Re*a*1j*(K-a**2*M)
        A = Q-2*a**2*K+a**4*M - 2*a*Re*1j*M - 1j*a*Re*(K2-a**2*K1)
        return A.diags().toarray(), B.diags().toarray()

    def solve(self, verbose=False):
        """Solve the Orr-Sommerfeld eigenvalue problem
        """
        if verbose:
            print('Solving the Orr-Sommerfeld eigenvalue problem...')
            print('Re = '+str(self.Re)+' and alfa = '+str(self.alfa))
        A, B = self.assemble()
        return eig(A, B)
        # return eig(np.dot(inv(B), A))

    @staticmethod
    def get_eigval(nx, eigvals, verbose=False):
        """Get the chosen eigenvalue

        Parameters
        ----------
            nx : int
                The chosen eigenvalue. nx=1 corresponds to the one with the
                largest imaginary part, nx=2 the second largest etc.
            eigvals : array
                Computed eigenvalues
            verbose : bool, optional
                Print the value of the chosen eigenvalue. Default is False.

        """
        indices = np.argsort(np.imag(eigvals))
        indi = indices[-1*np.array(nx)]
        eigval = eigvals[indi]
        if verbose:
            ev = list(eigval) if np.ndim(eigval) else [eigval]
            indi = list(indi) if np.ndim(indi) else [indi]
            for i, (e, v) in enumerate(zip(ev, indi)):
                print('Eigenvalue {} ({}) = {:2.16e}'.format(i+1, v, e))
        return indi, eigval

if __name__ == '__main__':
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
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print results')
    parser.set_defaults(plot=False)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    #z = OrrSommerfeld(N=120, Re=5772.2219, alfa=1.02056)
    z = OrrSommerfeld(**vars(args))
    evals, evectors = z.solve(args.verbose)
    d = z.get_eigval(1, evals, args.verbose)

    if args.Re == 8000.0 and args.alfa == 1.0 and args.N > 80:
        assert abs(d[1] - (0.24707506017508621+0.0026644103710965817j)) < 1e-12

    if args.plot:
        plt.figure()
        evi = evals*z.alfa
        plt.plot(evi.imag, evi.real, 'o')
        plt.axis([-10, 0.1, 0, 1])
        plt.show()
