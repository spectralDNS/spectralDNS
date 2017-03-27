from .Matvec import Biharmonic_matvec, Biharmonic_matvec3D, Helmholtz_matvec3D, \
    Helmholtz_matvec
from shenfun.chebyshev import bases
from shenfun import inner_product


class BiharmonicCoeff(object):

    def __init__(self, K, a0, alfa, beta, quad="GL"):
        self.quad = quad
        N = K.shape[0]
        self.shape = (N-4, N-4)
        SB = bases.ShenBiharmonicBasis(N, quad)
        self.S = inner_product((SB, 0), (SB, 4))
        self.B = inner_product((SB, 0), (SB, 0))
        self.A = inner_product((SB, 0), (SB, 2))
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
        SD = bases.ShenDirichletBasis(N+2, quad)
        self.B = inner_product((SD, 0), (SD, 0))
        self.A = inner_product((SD, 0), (SD, 2))
        self.alfa = alfa
        self.beta = beta

    def matvec(self, v, c):
        c.fill(0)
        if len(v.shape) > 1:
            Helmholtz_matvec3D(v, c, self.alfa, self.beta, self.A[0], self.A[2], self.B[0])
        else:
            Helmholtz_matvec(v, c, self.alfa, self.beta, self.A[0], self.A[2], self.B[0])
        return c
