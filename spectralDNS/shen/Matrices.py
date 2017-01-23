from __future__ import division
import numpy as np
from .Matvec import CDNmat_matvec, BDNmat_matvec, CDDmat_matvec, SBBmat_matvec, \
    SBBmat_matvec3D, Biharmonic_matvec, Biharmonic_matvec3D, Tridiagonal_matvec, \
    Tridiagonal_matvec3D, Pentadiagonal_matvec, Pentadiagonal_matvec3D, \
    CBD_matvec3D, CBD_matvec, CDB_matvec3D, ADDmat_matvec, Helmholtz_matvec3D, \
    Helmholtz_matvec, BBD_matvec3D, Tridiagonal_matvec

from .shentransform import ChebyshevTransform, ShenDirichletBasis, ShenNeumannBasis, \
    ShenBiharmonicBasis
from spectralDNS.utilities import inheritdocstrings
import numpy.polynomial.chebyshev as cheb
from scipy.sparse import diags
from collections import OrderedDict
import six
from copy import deepcopy

pi, zeros, ones, array = np.pi, np.zeros, np.ones, np.array
float, complex = np.float64, np.complex128

class SparseMatrix(dict):
    """Base class for sparse matrices

    The data is stored as a dictionary, where keys and values are,
    respectively, the offsets and values of the diagonal.

    A tridiagonal matrix of shape N x N could be created as

    >>> d = {-1: 1,
              0: -2,
              1: 1}

    >>> SparseMatrix(d, (N, N))

    In case of variable values, store the entire diagonal
    For an N x N matrix use:

    >>> d = {-1: ones(N-1),
              0: -2*ones(N),
              1: ones(N-1)}

    >>> SparseMatrix(d, (N, N))

    """

    def __init__(self, d, shape):
        dict.__init__(self, d)
        self.shape = shape
        self._diags = None

    def matvec(self, v, c, format='dia'):
        """Matrix vector product

        Returns c = dot(self, v)

        args:
            v    (input)         Numpy array of ndim=1 or 3
            c    (output)        Numpy array of same ndim as v

        kwargs:
            format  ('csr',      Choice for computation
                     'dia',      format = 'csr' or 'dia' uses sparse matrices
                     'python')   from scipy.sparse and their built in matvec.
                                 format = 'python' uses numpy and vectorization
                                 May be overloaded in subclass, for example
                                 with a Cython matvec

        """
        assert v.shape == c.shape
        N, M = self.shape
        c.fill(0)
        if len(v.shape) > 1:
            if format == 'python':
                for key, val in six.iteritems(self):
                    if key < 0:
                        for i in range(v.shape[1]):
                            for j in range(v.shape[2]):
                                c[-key:min(N, M-key), i, j] += val*v[:min(M, N+key), i, j]
                    else:
                        for i in range(v.shape[1]):
                            for j in range(v.shape[2]):
                                c[:min(N, M-key), i, j] += val*v[key:min(M, N+key), i, j]

            else:
                if not format in ('csr', 'dia'): # Fallback on 'csr'. Should probably throw warning
                    format = 'csr'
                diags = self.diags(format=format)
                for i in range(v.shape[1]):
                    for j in range(v.shape[2]):
                        c[:N, i, j] = diags.dot(v[:M, i, j])

        else:
            if format == 'python':
                for key, val in six.iteritems(self):
                    if key < 0:
                        c[-key:min(N, M-key)] += val*v[:min(M, N+key)]
                    else:
                        c[:min(N, M-key)] += val*v[key:min(M, N+key)]

            else:
                if not format in ('csr', 'dia'):
                    format = 'csr'
                diags = self.diags(format=format)
                c[:N] = diags.dot(v[:M])

        return c

    def diags(self, format='dia'):
        """Return a regular sparse matrix of specified format

        kwargs:
            format  ('dia', 'csr', 'csc')

        """
        if self._diags is None:
            self._diags = diags(list(self.values()), list(self.keys()),
                                shape=self.shape, format=format)

        if self._diags.format != format:
            self._diags = diags(list(self.values()), list(self.keys()),
                                shape=self.shape, format=format)

        return self._diags

    def __imul__(self, y):
        """self.__imul__(y) <==> self*=y"""
        assert isinstance(y, (np.float, np.int))
        for key, val in six.iteritems(self):
            # Check if symmetric
            if key < 0 and (-key) in self:
                if id(self[key]) == id(self[-key]):
                    continue
            self[key] *= y

        return self

    def __mul__(self, y):
        """Returns copy of self.__mul__(y) <==> self*y"""
        f = SparseMatrix(deepcopy(dict(self)), self.shape)
        assert isinstance(y, (np.float, np.int))
        for key, val in six.iteritems(f):
            # Check if symmetric
            if key < 0 and (-key) in f:
                if id(f[key]) == id(f[-key]):
                    continue
            f[key] *= y
        return f

    def __rmul__(self, y):
        """Returns copy of self.__rmul__(y) <==> y*self"""
        return self.__mul__(y)

    def __div__(self, y):
        """Returns copy self.__div__(y) <==> self/y"""
        f = SparseMatrix(deepcopy(dict(self)), self.shape)
        assert isinstance(y, (np.float, np.int))
        for key, val in six.iteritems(f):
            # Check if symmetric
            if key < 0 and (-key) in f:
                if id(f[key]) == id(f[-key]):
                    continue
            f[key] /= y
        return f

    def __truediv__(self, y):
        """Returns copy self.__div__(y) <==> self/y"""
        return self.__div__(y)

    def __add__(self, d):
        """Return copy of self.__add__(y) <==> self+d"""
        f = SparseMatrix(deepcopy(dict(self)), self.shape)
        assert isinstance(d, dict)
        assert d.shape == self.shape
        for key, val in six.iteritems(d):
            if key in f:
                # Check if symmetric and make copy if necessary
                if (-key) in f:
                    if id(f[key]) == id(f[-key]):
                        f[-key] = deepcopy(f[key])
                f[key] += val
            else:
                f[key] = val

        return f

    def __iadd__(self, d):
        """self.__iadd__(d) <==> self += d"""
        assert isinstance(d, dict)
        assert d.shape == self.shape
        for key, val in six.iteritems(d):
            if key in self:
                # Check if symmetric and make copy if necessary
                if (-key) in self:
                    if id(self[key]) == id(self[-key]):
                        self[-key] = deepcopy(self[key])
                self[key] += val
            else:
                self[key] = val

        return self

    def __sub__(self, d):
        """Return copy of self.__add__(y) <==> self+d"""
        f = SparseMatrix(deepcopy(dict(self)), self.shape)
        assert isinstance(d, dict)
        assert d.shape == self.shape
        for key, val in six.iteritems(d):
            if key in f:
                # Check if symmetric and make copy if necessary
                if (-key) in f:
                    if id(f[key]) == id(f[-key]):
                        f[-key] = deepcopy(f[key])
                f[key] -= val
            else:
                f[key] = -val

        return f

    def __isub__(self, d):
        """self.__isub__(d) <==> self -= d"""
        assert isinstance(d, dict)
        assert d.shape == self.shape
        for key, val in six.iteritems(d):
            if key in self:
                # Check if symmetric and make copy if necessary
                if (-key) in self:
                    if id(self[key]) == id(self[-key]):
                        self[-key] = deepcopy(self[key])
                self[key] -= val
            else:
                self[key] = -val

        return self


@inheritdocstrings
class ShenMatrix(SparseMatrix):
    """Base class for Shen matrices

    args:
        d                            Dictionary, where keys are the diagonal
                                     offsets and values the diagonals
        N      integer               Length of main diagonal
        trial  (basis, derivative)   tuple, where basis is an instance of
                                     one of
                                         - ChebyshevTransform
                                         - ShenDirichletBasis
                                         - ShenBiharmonicBasis
                                         - ShenNeumannBasis
                                     derivative is an integer, and represents
                                     the number of times the trial function
                                     should be differentiated
        test   basis                 One of the above basis functions
        scale  float                 Scale matrix with this constant


    Shen matrices are assumed to be sparse diagonal. The matrices are
    scalar products of trial and test functions from one of four function
    spaces

    Chebyshev basis and space of first kind

        T_k,
        span(T_k) for k = 0, 1, ..., N

    For homogeneous Dirichlet boundary conditions:

        phi_k = T_k - T_{k+2},
        span(phi_k) for k = 0, 1, ..., N-2

    For homogeneous Neumann boundary conditions:

        phi_k = T_k - (k/(k+2))**2T_{k+2},
        span(phi_k) for k = 1, 2, ..., N-2

    For Biharmonic basis with both homogeneous Dirichlet
    and Neumann:

        psi_k = T_k - 2(k+2)/(k+3)*T_{k+2} + (k+1)/(k+3)*T_{k+4},
        span(psi_k) for k = 0, 1, ..., N-4

    The scalar product is computed as a weighted inner product with
    w=1/sqrt(1-x**2) the weights.

    Mass matrix for Dirichlet basis:

        (phi_k, phi_j)_w = \int_{-1}^{1} phi_k phi_j w dx

    Stiffness matrix for Dirichlet basis:

        (phi_k'', phi_j)_w = \int_{-1}^{1} phi_k'' phi_j w dx

    etc.

    The matrix can be automatically created using, e.g., for the mass
    matrix of the Dirichlet space

      M = ShenMatrix({}, 16, (ShenDirichletBasis(), 0), ShenDirichletBasis())

    where the first (ShenDirichletBasis, 0) represents the trial function and
    the second the test function. The stiffness matrix can be obtained as

      A = ShenMatrix({}, 16, (ShenDirichletBasis(), 2), ShenDirichletBasis())

    where (ShenDirichletBasis, 2) signals that we use the second derivative
    of this trial function.

    The automatically created matrices may be overloaded with more exactly
    computed diagonals.

    Note that matrices with the Neumann basis are stored using index space
    k = 0, 1, ..., N-2, i.e., including the zero index. This is used for
    simplicity, and needs to be accounted for by users. For example, to
    solve the Poisson equation:

        from spectralDNS.shen.shentransform import ShenNeumannBasis
        import numpy as np
        from sympy import Symbol, sin, pi
        M = 32
        SN = ShenNeumannBasis('GC')
        x = Symbol("x")
        # Define an exact solution to compute rhs
        u = (1-x**2)*sin(pi*x)
        f = -u.diff(x, 2)
        points, weights = SN.points_and_weights(M, SN.quad)
        # Compute exact function on quadrature points
        uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
        fj = np.array([f.subs(x, h) for h in points], dtype=np.float)
        # Subtract mean
        fj -= np.dot(fj, weights)/weights.sum()
        uj -= np.dot(uj, weights)/weights.sum()
        # Get the stiffness matrix
        A = ShenMatrix({}, M, (SN, 2), SN, scale=-1)
        # Compute rhs scalar product
        f_hat = np.zeros(M)
        f_hat = SN.scalar_product(fj, f_hat)
        # Solve
        u_hat = np.zeros(M)
        s = slice(1, M-2)
        u_hat[s] = np.linalg.solve(A.diags().toarray()[s, s], f_hat[s])
        # Compare with exact solution
        u0 = np.zeros(M)
        u0 = SN.ifst(u_hat, u0)
        assert np.allclose(u0, uj)


    """
    def __init__(self, d, N,  trial, test, scale=1.0):
        self.trialfunction, self.derivative = trial
        self.testfunction = test
        self.N = N
        self.scale = scale
        shape = self.get_shape()
        if d == {}:
            D = self.get_dense_matrix()[:shape[0], :shape[1]]
            d = extract_diagonal_matrix(D)
        SparseMatrix.__init__(self, d, shape)
        if not round(scale-1.0, 8) == 0:
            self *= scale

    def get_shape(self):
        """Return shape of matrix"""
        return (self.testfunction.get_shape(self.N),
                self.trialfunction.get_shape(self.N))

    def get_ck(self, N, quad):
        ck = ones(N, int)
        ck[0] = 2
        if quad == "GL": ck[-1] = 2
        return ck

    def get_dense_matrix(self):
        """Return dense matrix automatically computed from basis"""
        N = self.N
        x, w = self.testfunction.points_and_weights(N, self.trialfunction.quad)
        V = self.testfunction.vandermonde(x, N)
        test = self.testfunction.get_vandermonde_basis(V)
        trial = self.trialfunction.get_vandermonde_basis_derivative(V, self.derivative)
        return np.dot(w*test.T, trial)

    def test(self):
        """Test for matrix.

        Test that automatically created matrix is the same as the one created

        """
        N, M = self.shape
        D = self.get_dense_matrix()[:N, :M]
        Dsp = extract_diagonal_matrix(D)
        Dsp *= self.scale
        for key, val in six.iteritems(self):
            assert np.allclose(val, Dsp[key])

def extract_diagonal_matrix(M, tol=1e-8):
    """Return matrix with essentially zero diagonals nulled out
    """
    du = []
    dl = []
    d = {}
    for i in range(M.shape[1]):
        u = M.diagonal(i).copy()
        if abs(u).max() > tol:
            d[i] = u

    for i in range(1, M.shape[0]):
        l = M.diagonal(-i).copy()
        if abs(l).max() > tol:
            d[-i] = l

    return SparseMatrix(d, M.shape)

@inheritdocstrings
class BDDmat(ShenMatrix):
    """Matrix for inner product B_{kj}=(phi_j, phi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and phi is a Shen Dirichlet basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        d = {0: pi/2*(ck[:-2]+ck[2:]),
             2: array([-pi/2])}
        d[-2] = d[2]
        trial = ShenDirichletBasis(quad=quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), trial)

    def matvec(self, v, c, format='cython'):
        N, M = self.shape
        c.fill(0)
        if len(v.shape) > 1:
            ld = self[-2]*ones(M-2)
            if format == 'cython':
                Tridiagonal_matvec3D(v, c, ld, self[0], ld)

            elif format == 'self':
                c[:(N-2)] = self[2]*v[2:N]
                c[:N]    += self[0].repeat(array(v.shape[1:]).prod()).reshape(v[:N].shape)*v[:N]
                c[2:N]   += self[-2]*v[:(N-2)]

            else:
                c = super(BDDmat, self).matvec(v, c, format=format)

        else:
            if format == 'cython':
                ld = self[-2]*ones(M-2)
                Tridiagonal_matvec(v, c, ld, self[0], ld)

            elif format == 'self':
                c[:(N-2)] = self[2]*v[2:M]
                c[:N]    += self[0]*v[:M]
                c[2:N]   += self[-2]*v[:(M-2)]

            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        return c


@inheritdocstrings
class BNDmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 1, 2, ..., N-2

    psi is the Shen Dirichlet basis and phi is a Shen Neumann basis.

    For simplicity, the matrix is stored including the zero index row (k=0)

    """

    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        k = K[:N-2].astype(float)
        d = {-2: -pi/2,
              0: pi/2.*(ck[:-2]+ck[2:]*(k/(k+2))**2),
              2: -pi/2*(k[:N-4]/(k[:N-4]+2))**2}
        trial = ShenDirichletBasis(quad=quad)
        test  = ShenNeumannBasis(quad=quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), test)

    def matvec(self, v, c, format='csr'):
        c = ShenMatrix.matvec(self, v, c, format=format)
        c[0] = 0
        return c


@inheritdocstrings
class BDNmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 0, 1, ..., N-2

    psi is the Shen Dirichlet basis and phi is a Shen Neumann basis.

    For simplicity, the matrix is stored including the zero index column (j=0)

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        k = K[:N-2].astype(float)
        d = {-2: -pi/2*(k[:N-4]/(k[:N-4]+2))**2,
              0:  pi/2.*(ck[:-2]+ck[2:]*(k/(k+2))**2),
              2: -pi/2}
        trial = ShenNeumannBasis(quad=quad)
        test  = ShenDirichletBasis(quad=quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), test)


@inheritdocstrings
class BTTmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (T_j, T_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N

    and T_j is the jth order Chebyshev function of the first kind.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        trial = ChebyshevTransform(quad)
        ShenMatrix.__init__(self, {0: pi/2*ck}, N, (trial, 0), trial)

    def matvec(self, v, c, format='self'):
        N = self.shape[0]
        c.fill(0)
        if len(v.shape) > 1:
            if format == 'self':
                c[:] = self[0].repeat(array(v.shape[1:]).prod()).reshape(v[:].shape)*v
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        else:
            if format == 'self':
                c[:] = self[0]*v
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        return c


@inheritdocstrings
class BNNmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 1, 2, ..., N-2

    and phi is the Shen Neumann basis.

    The matrix is stored including the zero index row and column

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        k = K[:-2].astype(float)
        d = {0: pi/2*(ck[:-2]+ck[2:]*(k[:]/(k[:]+2))**4),
             2: -pi/2*((k[2:]-2)/(k[2:]))**2}
        d[-2] = d[2]
        trial = ShenNeumannBasis(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), trial)

    def matvec(self, v, c, format='csr'):
        c = ShenMatrix.matvec(self, v, c, format=format)
        c[0] = 0
        return c


class BDTmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (T_j, phi_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N-2

    phi is the Shen Dirichlet basis and T is the Chebyshev basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        d = {0: pi/2*ck[:N-2],
             2: -pi/2*ck[2:]}
        trial = ChebyshevTransform(quad)
        test = ShenDirichletBasis(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), test)


class BTDmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, T_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N

    phi is the Shen Dirichlet basis and T is the Chebyshev basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        d = {-2: -pi/2*ck[2:],
              0: pi/2*ck[:N-2]}
        test  = ChebyshevTransform(quad)
        trial = ShenDirichletBasis(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), test)


class BTNmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, T_k)_w

    where

        j = 1, 2, ..., N-2 and k = 0, 1, ..., N

    phi is the Shen Neumann basis and T is the Chebyshev basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        d = {-2: -pi/2*ck[2:]*((K[2:]-2)/K[2:])**2,
              0: pi/2*ck[:-2]}
        trial = ShenNeumannBasis(quad)
        test = ChebyshevTransform(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), test)


class BBBmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, psi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-4

    and phi is the Shen Biharmonic basis.

    """

    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        k = K[:N-4].astype(float)

        d = {4: (k[:-4]+1)/(k[:-4]+3)*pi/2,
             2: -((k[:-2]+2)/(k[:-2]+3) + (k[:-2]+4)*(k[:-2]+1)/((k[:-2]+5)*(k[:-2]+3)))*pi,
             0: (ck[:N-4] + 4*((k+2)/(k+3))**2 + ck[4:]*((k+1)/(k+3))**2)*pi/2.}
        d[-2] = d[2]
        d[-4] = d[4]
        trial = ShenBiharmonicBasis(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), trial)

    def matvec(self, v, c, format='cython'):
        c.fill(0)
        N = self.shape[0]
        if len(v.shape) > 1:
            if format == 'self':
                vv = v[:-4]
                c[:N] = self[0].repeat(array(v.shape[1:]).prod()).reshape(vv.shape) * vv[:]
                c[:N-2] += self[2].repeat(array(v.shape[1:]).prod()).reshape(vv[2:].shape) * vv[2:]
                c[:N-4] += self[4].repeat(array(v.shape[1:]).prod()).reshape(vv[4:].shape) * vv[4:]
                c[2:N]  += self[-2].repeat(array(v.shape[1:]).prod()).reshape(vv[:-2].shape) * vv[:-2]
                c[4:N]  += self[-4].repeat(array(v.shape[1:]).prod()).reshape(vv[:-4].shape) * vv[:-4]

            elif format == 'cython':
                Pentadiagonal_matvec3D(v, c, self[-4], self[-2], self[0], self[2], self[4])

            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        else:
            if format == 'self':
                vv = v[:-4]
                c[:N] = self[0] * vv[:]
                c[:N-2] += self[2] * vv[2:]
                c[:N-4] += self[4] * vv[4:]
                c[2:N]  += self[-2] * vv[:-2]
                c[4:N]  += self[-4] * vv[:-4]

            elif format == 'cython':
                Pentadiagonal_matvec(v, c, self[-4], self[-2], self[0], self[2], self[4])

            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        return c


#import decimal
#class BBBmat(BaseMatrix):

    #def __init__(self, K, quad='GC'):
        #BaseMatrix.__init__(self)
        #N = K.shape[0]-4
        #self.shape = (N, N)
        #ck = ones(N)
        #ckp = ones(N)
        #ck[0] = 2
        #if quad == "GL": ckp[-1] = 2
        #k = K[:N].astype(float)

        #self.dd = (ck + 4*((k+2)/(k+3))**2 + ckp*((k+1)/(k+3))**2)*pi/2.
        #self.ud = -((k[:-2]+2)/(k[:-2]+3) + (k[:-2]+4)*(k[:-2]+1)/((k[:-2]+5)*(k[:-2]+3)))*pi
        #self.uud = (k[:-4]+1)/(k[:-4]+3)*pi/2
        ##kd = array(map(decimal.Decimal, K[:N]))
        ##ckd = array(map(decimal.Decimal, [1,]*N))
        ##ckdp = array(map(decimal.Decimal, [1,]*N))
        ##ckd[0] = decimal.Decimal(2)
        ##if quad == "GL": ckdp[-1] = decimal.Decimal(2)
        ##one = decimal.Decimal(1)
        ##two = decimal.Decimal(2)
        ##three = decimal.Decimal(3)
        ##four = decimal.Decimal(4)
        ##five = decimal.Decimal(5)
        ##PI = decimal.Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089")

        ###dd = (ckd + four*((kd+two)/(kd+three))**two + ckdp*((kd+one)/(kd+three))**two)*PI/two
        ##dd = (ckd + four*((kd+two)*(kd+two)/(kd+three)/(kd+three)) + ckdp*((kd+one)*(kd+one)/(kd+three)/(kd+three)))*PI/two
        ##ud = -((kd[:-2]+two)/(kd[:-2]+three) + (kd[:-2]+four)*(kd[:-2]+one)/((kd[:-2]+five)*(kd[:-2]+three)))*PI
        ##uud = (kd[:-4]+one)/(kd[:-4]+three)*PI/two

        ###self.dd = dd.astype(float)
        ###self.ud = ud.astype(float)
        ###self.uud = uud.astype(float)
        ##self.dd = dd
        ##self.ud = ud
        ##self.uud = uud

        #self.ld = self.ud
        #self.lld = self.uud

    #def matvec(self, v):
        #c = self.get_return_array(v)
        #N = self.shape[0]
        #if len(v.shape) > 1:
            ##vv = v[:-4]
            ##c[:N] = self.dd.repeat(array(v.shape[1:]).prod()).reshape(vv.shape) * vv[:]
            ##c[:N-2] += self.ud.repeat(array(v.shape[1:]).prod()).reshape(vv[2:].shape) * vv[2:]
            ##c[:N-4] += self.uud.repeat(array(v.shape[1:]).prod()).reshape(vv[4:].shape) * vv[4:]
            ##c[2:N]  += self.ld.repeat(array(v.shape[1:]).prod()).reshape(vv[:-2].shape) * vv[:-2]
            ##c[4:N]  += self.lld.repeat(array(v.shape[1:]).prod()).reshape(vv[:-4].shape) * vv[:-4]
            #Pentadiagonal_matvec3D(v, c, self.lld, self.ld, self.dd, self.ud, self.uud)

        #else:
            ##vv = v[:-4]
            ##c[:N] = self.dd * vv[:]
            ##c[:N-2] += self.ud * vv[2:]
            ##c[:N-4] += self.uud * vv[4:]
            ##c[2:N]  += self.ld * vv[:-2]
            ##c[4:N]  += self.lld * vv[:-4]
            #Pentadiagonal_matvec(v, c, self.lld, self.ld, self.dd, self.ud, self.uud)

        #return c

    #def diags(self):
        #return diags([self.lld, self.ld, self.dd, self.ud, self.uud], range(-4, 6, 2), shape=self.shape)

class BBDmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-4

    and phi is the Shen Dirichlet basis and psi the Shen Biharmonic basis.

    """


    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        k = K[:N-4].astype(float)
        a = 2*(k+2)/(k+3)
        b = (k[:N-4]+1)/(k[:N-4]+3)
        d = {-2: -pi/2,
              0: (ck[:N-4] + a)*pi/2,
              2: -(a+b*ck[4:])*pi/2,
              4: b[:-2]*pi/2}
        trial = ShenDirichletBasis(quad)
        test = ShenBiharmonicBasis(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), test)

    def matvec(self, v, c, format='cython'):
        c.fill(0)
        N = self.shape[0]
        if len(v.shape) > 1:
            if format == 'self':
                vv = v[:-2]
                c[:N] = self[0].repeat(array(v.shape[1:]).prod()).reshape(vv[:-2].shape) * vv[:-2]
                c[:N] += self[2].repeat(array(v.shape[1:]).prod()).reshape(vv[2:].shape) * vv[2:]
                c[:N-2] += self[4].repeat(array(v.shape[1:]).prod()).reshape(vv[4:].shape) * vv[4:]
                c[2:N]  += self[-2] * vv[:-4]

            elif format == 'cython':
                BBD_matvec3D(v, c, self[-2], self[0], self[2], self[4])

            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        else:
            if format == 'self':
                vv = v[:-2]
                c[:N] = self[0] * vv[:-2]
                c[:N] += self[2] * vv[2:]
                c[:N-2] += self[4] * vv[4:]
                c[2:N]  += self[-2] * vv[:-4]
            else:
                if format == 'cython': format = 'csr'
                c = ShenMatrix.matvec(self, v, c, format=format)

        return c

# Derivative matrices
class CDNmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (psi'_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 0, 1, ..., N-2

    and phi is the Shen Dirichlet basis and psi the Shen Neumann basis.

    For simplicity, the matrix is stored including the zero index row (k=0)

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        k = K[:N-2].astype(float)
        d = {-1: -((k[1:]-1)/(k[1:]+1))**2*(k[1:]+1)*pi,
              1: (k[:-1]+1)*pi}
        trial = ShenNeumannBasis()
        test = ShenDirichletBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), test)

    def matvec(self, v, c, format='cython'):
        N = self.shape[0]
        if len(v.shape) > 1:
            if format == 'cython':
                CDNmat_matvec(self[1], self[-1][1:], v, c)
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)
        else:
            if format == 'cython': format = 'csr'
            c = ShenMatrix.matvec(self, v, c, format=format)

        return c


class CDDmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (phi'_j, phi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and phi is the Shen Dirichlet basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {-1: -(K[1:N-2]+1)*pi,
              1: (K[:(N-3)]+1)*pi}
        trial = ShenDirichletBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), trial)

    def matvec(self, v, c, format='cython'):
        N = self.shape[0]
        c.fill(0)
        if len(v.shape) > 1:
            if format == 'self':
                c[:N-1] = self[1].repeat(array(v.shape[1:]).prod()).reshape(v[1:N].shape)*v[1:N]
                c[1:N] += self[-1].repeat(array(v.shape[1:]).prod()).reshape(v[:(N-1)].shape)*v[:(N-1)]
            elif format == 'cython':
                CDDmat_matvec(self[1], self[-1], v, c)
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)
        else:
            if format == 'self':
                c[:N-1] = self[1]*v[1:N]
                c[1:N] += self[-1]*v[:(N-1)]
            else:
                if format == 'cython': format='csr'
                c = ShenMatrix.matvec(self, v, c, format=format)

        return c


class CNDmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (phi'_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 1, 2, ..., N-2

    and phi is the Shen Dirichlet basis and psi the Shen Neumann basis.

    For simplicity, the matrix is stored including the zero index coloumn (j=0)

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        k = K[:N-2].astype(float)
        d = {-1: -(k[1:]+1)*pi,
              1: -(2-k[:-1]**2/(k[:-1]+2)**2*(k[:-1]+3))*pi}
        for i in range(3, N-1, 2):
            d[i] = -(1-k[:-i]**2/(k[:-i]+2)**2)*2*pi
        trial = ShenDirichletBasis()
        test = ShenNeumannBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), test)

    def matvec(self, v, c, format='csr'):
        c = ShenMatrix.matvec(self, v, c, format=format)
        c[0] = 0
        return c


class CTDmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (phi'_j, T_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N

    phi is the Shen Dirichlet basis and T is the Chebyshev basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {-1: -(K[1:N-1]+1)*pi,
              1: -2*pi}
        for i in range(3, N-2, 2):
            d[i] = -2*pi
        trial = ShenDirichletBasis()
        test = ChebyshevTransform()
        ShenMatrix.__init__(self, d, N, (trial, 1), test)


class CDTmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (T'_j, phi_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N-2

    phi is the Shen Dirichlet basis and T is the Chebyshev basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {1: pi*(K[:N-2]+1)}
        trial = ChebyshevTransform()
        test = ShenDirichletBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), test)


class CBDmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (phi'_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-4

    phi is the Shen Dirichlet basis and psi the Shen Biharmonic basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {-1: -(K[1:N-4]+1)*pi,
              1: 2*(K[:N-4]+1)*pi,
              3: -(K[:N-5]+1)*pi}
        trial = ShenDirichletBasis()
        test = ShenBiharmonicBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), test)

    def matvec(self, v, c, format='cython'):
        N, M = self.shape
        c.fill(0)
        if len(v.shape) > 1:
            if format == 'self':
                c[1:N] = self[-1].repeat(array(v.shape[1:]).prod()).reshape(v[:M-3].shape)*v[:M-3]
                c[:N] += self[1].repeat(array(v.shape[1:]).prod()).reshape(v[1:M-1].shape)*v[1:M-1]
                c[:N-1]+= self[3].repeat(array(v.shape[1:]).prod()).reshape(v[3:M].shape)*v[3:M]
            elif format == 'cython':
                CBD_matvec3D(v, c, self[-1], self[1], self[3])
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)
        else:
            if format == 'self':
                c[1:N] = self[-1] * v[:M-3]
                c[:N] += self[1] * v[1:M-1]
                c[:N-1] += self[3] * v[3:M]
            elif format == 'cython':
                CBD_matvec(v, c, self[-1], self[1], self[3])
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)
        return c


class CDBmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (psi'_j, phi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-2

    phi is the Shen Dirichlet basis and psi the Shen Biharmonic basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {-3: (K[3:-2]-2)*(K[3:-2]+1)/K[3:-2]*pi,
             -1: -2*(K[1:-3]+1)**2/(K[1:-3]+2)*pi,
              1: (K[:-5]+1)*pi}
        trial = ShenBiharmonicBasis()
        test = ShenDirichletBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), test)

    def matvec(self, v, c, format='cython'):
        N, M = self.shape
        c.fill(0)
        if len(v.shape) > 1:
            if format == 'self':
                c[3:N] = self[-3].repeat(array(v.shape[1:]).prod()).reshape(v[:M-1].shape) * v[:M-1]
                c[1:N-1] += self[-1].repeat(array(v.shape[1:]).prod()).reshape(v[:M].shape) * v[:M]
                c[:N-3] += self[1].repeat(array(v.shape[1:]).prod()).reshape(v[1:M].shape) * v[1:M]
            elif format == 'cython':
                CDB_matvec3D(v, c, self[-3], self[-1], self[1])
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        else:
            if format == 'self':
                c[3:N] = self[-3] * v[:M-1]
                c[1:N-1] += self[-1] * v[:M]
                c[:N-3] += self[1] * v[1:M]
            else:
                if format == 'cython': format = 'csr'
                c = ShenMatrix.matvec(self, v, c, format=format)

        return c


class ABBmat(ShenMatrix):
    """Stiffness matrix for inner product A_{kj} = -(psi''_j, psi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-4

    and psi is the Shen Biharmonic basis.

    """

    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        ki = K[:N-4]
        k = K[:N-4].astype(float)
        d = {-2: 2*(ki[2:]-1)*(ki[2:]+2)*pi,
              0: -4*((ki+1)*(ki+2)**2)/(k+3)*pi,
              2: 2*(ki[:-2]+1)*(ki[:-2]+2)*pi}
        trial = ShenBiharmonicBasis()
        ShenMatrix.__init__(self, d, N, (trial, 2), trial)

    def matvec(self, v, c, format='cython'):
        N = self.shape[0]
        c.fill(0)
        if len(v.shape) > 1:
            if format == 'self':
                c[:N] = self[0].repeat(array(v.shape[1:]).prod()).reshape(v[:N].shape) * v[:N]
                c[:N-2] += self[2].repeat(array(v.shape[1:]).prod()).reshape(v[2:N].shape) * v[2:N]
                c[2:N] += self[-2].repeat(array(v.shape[1:]).prod()).reshape(v[:N-2].shape) * v[:N-2]
            elif format == 'cython':
                Tridiagonal_matvec3D(v, c, self[-2], self[0], self[2])
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        else:
            if format == 'self':
                c[:N] = self[0] * v[:N]
                c[:N-2] += self[2] * v[2:N]
                c[2:N] += self[-2] * v[:N-2]
            elif format == 'cython':
                Tridiagonal_matvec(v, c, self[-2], self[0], self[2])
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        return c


class ADDmat(ShenMatrix):
    """Stiffness matrix for inner product A_{kj} = -(psi''_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and psi is the Shen Dirichlet basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {0: -2*pi*(K[:N-2]+1)*(K[:N-2]+2)}
        for i in range(2, N-2, 2):
            d[i] = -4*pi*(K[:-(i+2)]+1)
        trial = ShenDirichletBasis()
        ShenMatrix.__init__(self, d, N, (trial, 2), trial, -1.0)

    def matvec(self, v, c, format='cython'):
        N = self.shape[0]
        c = zeros(v.shape, dtype=v.dtype)
        if len(v.shape) > 1:
            if format == 'cython': format = 'csr'
            c = ShenMatrix.matvec(self, v, c, format=format)

        else:
            if format == 'cython':
                ADDmat_matvec(v, c, self[0])
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        return c


class ANNmat(ShenMatrix):
    """Stiffness matrix for inner product A_{kj} = -(phi''_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 1, 2, ..., N-2

    and phi is the Shen Neumann basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        k = K[:-2].astype(float)
        d = {0: -2*pi*k**2*(k+1)/(k+2)}
        for i in range(2, N-2, 2):
            d[i] = -4*pi*(k[:-i]+i)**2*(k[:-i]+1)/(k[:-i]+2)**2
        trial = ShenNeumannBasis()
        ShenMatrix.__init__(self, d, N, (trial, 2), trial, -1.0)

    def matvec(self, v, c, format='csr'):
        c = ShenMatrix.matvec(self, v, c, format=format)
        c[0] = 0
        return c


class ATTmat(ShenMatrix):
    """Stiffness matrix for inner product A_{kj} = -(psi''_j, psi_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N

    and psi is the Chebyshev basis.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {}
        for j in range(2, N, 2):
            d[j] = K[j:]*(K[j:]**2-K[:-j]**2)*pi/2.
        trial = ChebyshevTransform()
        ShenMatrix.__init__(self, d, N, (trial, 2), trial, -1.0)


class SBBmat(ShenMatrix):
    """Biharmonic matrix for inner product S_{kj} = -(psi''''_j, psi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-4

    and psi is the Shen Biharmonic basis.

    """

    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        k = K[:N-4].astype(float)
        ki = K[:N-4]
        i = 8*(ki+1)**2*(ki+2)*(ki+4)
        d = {0: i * pi}
        for j in range(2, N-4, 2):
            i = 8*(ki[:-j]+1)*(ki[:-j]+2)*(ki[:-j]*(ki[:-j]+4)+3*(ki[j:]+2)**2)
            d[j] = array(i*pi/(k[j:]+3))
        trial = ShenBiharmonicBasis()
        ShenMatrix.__init__(self, d, N, (trial, 4), trial)

    def matvec(self, v, c, format='cython'):
        N = self.shape[0]
        c.fill(0)
        if len(v.shape) > 1:
            if format == 'cython':
                SBBmat_matvec3D(v, c, self[0])
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)
        else:
            if format == 'cython':
                SBBmat_matvec(v, c, self[0])
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        return c

mass_matrix = {
    'ChebyshevTransform': BTTmat,
    'ShenDirichletBasis': BDDmat,
    'ShenNeumannBasis': BNNmat,
    'ShenBiharmonicBasis': BBBmat
}


class BiharmonicCoeff(object):

    def __init__(self, K, a0, alfa, beta, quad="GL"):
        self.quad = quad
        N = K.shape[0]
        self.shape = (N-4, N-4)
        self.S = SBBmat(K)
        self.B = BBBmat(K, self.quad)
        self.A = ABBmat(K)
        self.a0 = a0
        self.alfa = alfa
        self.beta = beta

    def matvec(self, v, c):
        N = self.shape[0]
        #c = zeros(v.shape, dtype=v.dtype)
        c.fill(0)
        if len(v.shape) > 1:
            Biharmonic_matvec3D(v, c, self.a0, self.alfa, self.beta, self.S[0], self.S[2],
                                self.S[4], self.A[-2], self.A[0], self.A[2],
                                self.B[-4], self.B[-2], self.B[0], self.B[2], self.B[4])
        else:
            Biharmonic_matvec(v, c, self.a0, self.alfa, self.beta, self.S[0], self.S[2],
                                self.S[4], self.A[-2], self.A[0], self.A[2],
                                self.B[-4], self.B[-2], self.B[0], self.B[2], self.B[4])
        return c


class HelmholtzCoeff(object):

    def __init__(self, K, alfa, beta, quad="GL"):
        """alfa*ADD + beta*BDD
        """
        self.quad = quad
        N = self.N = K.shape[0]-2
        self.shape = (N, N)
        self.B = BDDmat(K, self.quad)
        self.A = ADDmat(K)
        self.alfa = alfa
        self.beta = beta

    def matvec(self, v, c):
        N = self.shape[0]
        c.fill(0)
        if len(v.shape) > 1:
            Helmholtz_matvec3D(v, c, self.alfa, self.beta, self.A[0], self.A[2], self.B[0])
        else:
            Helmholtz_matvec(v, c, self.alfa, self.beta, self.A[0], self.A[2], self.B[0])
        return c

