"""
Solve the Orr-Sommerfeld eigenvalue problem

Using Shen's biharmonic basis

"""
import warnings
import six
from scipy.linalg import eig
#from numpy.linalg import eig
#from numpy.linalg import inv
import numpy as np
from shenfun.chebyshev.bases import ShenBiharmonicBasis, \
    ShenDirichletBasis
from shenfun.spectralbase import inner_product
from shenfun.matrixbase import extract_diagonal_matrix

np.seterr(divide='ignore')

try:
    from matplotlib import pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")


class OrrSommerfeld(object):
    def __init__(self, **kwargs):
        self.par = {'alfa':1.,
                    'Re':8000.,
                    'N':80,
                    'quad': 'GC'}
        self.par.update(**kwargs)
        for name, val in six.iteritems(self.par):
            setattr(self, name, val)
        self.P4 = np.zeros(0)
        self.T4x = np.zeros(0)
        self.SB, self.SD, self.CDB = (None,)*3
        self.x, self.w = None, None

    def interp(self, y, eigvals, eigvectors, eigval=1, same_mesh=False, verbose=False):
        """Interpolate solution eigenvector and it's derivative onto y

        args:
            y            Interpolation points
            eigvals      All computed eigenvalues
            eigvectors   All computed eigenvectors
        kwargs:
            eigval       The chosen eigenvalue, ranked with descending imaginary
                         part. The largest imaginary part is 1, the second
                         largest is 2, etc.
            same_mesh    Boolean. Whether or not to interpolate to the same
                         quadrature points as used for computing the eigenvectors
            verbose      Boolean. Print information or not
        """
        N = self.N
        nx, eigval = self.get_eigval(eigval, eigvals, verbose)
        phi_hat = np.zeros(N, np.complex)
        phi_hat[:-4] = np.squeeze(eigvectors[:, nx])
        if same_mesh:
            phi = np.zeros_like(phi_hat)
            dphidy = np.zeros_like(phi_hat)
            if self.SB is None:
                self.SB = ShenBiharmonicBasis(N, quad=self.quad, plan=True)
                self.SD = ShenDirichletBasis(N, quad=self.quad, plan=True)
                self.CDB = inner_product((self.SD, 0), (self.SB, 1))

            phi = self.SB.ifst(phi_hat, phi)
            dphidy_hat = self.CDB.matvec(phi_hat)
            dphidy_hat = self.SD.apply_inverse_mass(dphidy_hat)
            dphidy = self.SD.backward(dphidy_hat, dphidy)

        else:
            # Recompute interpolation matrices if necessary
            if not len(self.P4) == len(y):
                SB = ShenBiharmonicBasis(N, quad=self.quad)
                V = SB.vandermonde(y)
                self.P4 = SB.get_vandermonde_basis(V)
                self.T4x = SB.get_vandermonde_basis_derivative(V, 1)
            phi = np.dot(self.P4, phi_hat)
            dphidy = np.dot(self.T4x, phi_hat)

        return eigval, phi, dphidy

    def assemble(self):
        N = self.N
        SB = ShenBiharmonicBasis(N, quad=self.quad)
        SB.plan((N, N), 0, np.float, {})

        CT = SB.CT
        x, w = self.x, self.w = SB.points_and_weights(N)
        V = SB.vandermonde(x)

        # Trial function
        P4 = SB.get_vandermonde_basis(V)

        # Second derivatives
        T2x = SB.get_vandermonde_basis_derivative(V, 2)

        # (u'', v)
        K = np.zeros((N, N))
        K[:-4, :-4] = inner_product((SB, 0), (SB, 2)).diags().toarray()

        # ((1-x**2)u, v)
        xx = np.broadcast_to((1-x**2)[:, np.newaxis], (N, N))
        #K1 = np.dot(w*P4.T, xx*P4)  # Alternative: K1 = np.dot(w*P4.T, ((1-x**2)*P4.T).T)
        K1 = np.zeros((N, N))
        K1 = SB.scalar_product(xx*P4, K1)
        K1 = extract_diagonal_matrix(K1).diags().toarray() # For improved roundoff

        # ((1-x**2)u'', v)
        K2 = np.zeros((N, N))
        K2 = SB.scalar_product(xx*T2x, K2)
        K2 = extract_diagonal_matrix(K2).diags().toarray() # For improved roundoff

        # (u'''', v)
        Q = np.zeros((self.N, self.N))
        Q[:-4, :-4] = inner_product((SB, 0), (SB, 4)).diags().toarray()

        # (u, v)
        M = np.zeros((self.N, self.N))
        M[:-4, :-4] = inner_product((SB, 0), (SB, 0)).diags().toarray()

        Re = self.Re
        a = self.alfa
        B = -Re*a*1j*(K-a**2*M)
        A = Q-2*a**2*K+a**4*M - 2*a*Re*1j*M - 1j*a*Re*(K2-a**2*K1)
        return A, B

    def solve(self, verbose=False):
        """Solve the Orr-Sommerfeld eigenvalue problem
        """
        if verbose:
            print('Solving the Orr-Sommerfeld eigenvalue problem...')
            print('Re = '+str(self.par['Re'])+' and alfa = '+str(self.par['alfa']))
        A, B = self.assemble()
        return eig(A[:-4, :-4], B[:-4, :-4])
        # return eig(np.dot(inv(B[:-4, :-4]), A[:-4, :-4]))

    @staticmethod
    def get_eigval(nx, eigvals, verbose=False):
        """Get the chosen eigenvalue

        Args:
            nx       The chosen eigenvalue. nx=1 corresponds to the one with the
                     largest imaginary part, nx=2 the second largest etc.
            eigvals  Computed eigenvalues

            verbose  Print the value of the chosen eigenvalue

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
    eigvals, eigvectors = z.solve(args.verbose)
    d = z.get_eigval(1, eigvals, args.verbose)
    if args.Re == 8000.0 and args.alfa == 1.0 and args.N > 80:
        assert abs(d[1] - (0.24707506017508621+0.0026644103710965817j)) < 1e-12

    if args.plot:
        plt.figure()
        ev = eigvals*z.alfa
        plt.plot(ev.imag, ev.real, 'o')
        plt.axis([-10, 0.1, 0, 1])
        plt.show()
