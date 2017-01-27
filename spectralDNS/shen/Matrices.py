from .Matvec import Biharmonic_matvec, Biharmonic_matvec3D, Helmholtz_matvec3D, \
    Helmholtz_matvec

from shenfun.chebyshev.bases import ChebyshevBasis, ShenDirichletBasis, ShenNeumannBasis, \
    ShenBiharmonicBasis
from shenfun.chebyshev.matrices import BDDmat, ADDmat, BBBmat, ABBmat, SBBmat


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
        #c = zeros(v.shape, dtype=v.dtype)
        c.fill(0)
        if len(v.shape) > 1:
            Biharmonic_matvec3D(v, c, self.a0, self.alfa, self.beta, self.S[0],
                                self.S[2], self.S[4], self.A[-2], self.A[0],
                                self.A[2], self.B[-4], self.B[-2], self.B[0],
                                self.B[2], self.B[4])
        else:
            Biharmonic_matvec(v, c, self.a0, self.alfa, self.beta, self.S[0],
                              self.S[2], self.S[4], self.A[-2], self.A[0],
                              self.A[2], self.B[-4], self.B[-2], self.B[0],
                              self.B[2], self.B[4])
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
        c.fill(0)
        if len(v.shape) > 1:
            Helmholtz_matvec3D(v, c, self.alfa, self.beta, self.A[0], self.A[2], self.B[0])
        else:
            Helmholtz_matvec(v, c, self.alfa, self.beta, self.A[0], self.A[2], self.B[0])
        return c

