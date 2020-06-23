import numpy as np
from shenfun.chebyshev import bases
from shenfun.spectralbase import inner_product
from shenfun.optimization.cython import Matvec

__all__ = ('BiharmonicCoeff', 'HelmholtzCoeff')

class BiharmonicCoeff(object):

    def __init__(self, N, a0, alfa, beta, axis, quad="GL"):
        self.quad = quad
        self.shape = (N-4, N-4)
        SB = bases.ShenBiharmonic(N, quad)
        self.S = inner_product((SB, 0), (SB, 4))
        self.B = inner_product((SB, 0), (SB, 0))
        self.A = inner_product((SB, 0), (SB, 2))
        self.axis = axis
        self.a0 = a0
        self.alfa = alfa
        self.beta = beta

    def matvec(self, v, c):
        c.fill(0)
        Matvec.Biharmonic_matvec(v, c, self.a0, self.alfa, self.beta, self.S[0],
                                 self.S[2], self.S[4], self.A[-2], self.A[0],
                                 self.A[2], self.B[-4], self.B[-2], self.B[0],
                                 self.B[2], self.B[4], self.axis)
        return c

class HelmholtzCoeff(object):

    def __init__(self, N, alfa, beta, axis, quad="GL"):
        """alfa*ADD + beta*BDD
        """
        self.quad = quad
        self.shape = (N-2, N-2)
        SD = bases.ShenDirichlet(N, quad)
        self.B = inner_product((SD, 0), (SD, 0))
        self.A = inner_product((SD, 0), (SD, 2))
        self.axis = axis
        self.alfa = np.broadcast_to(alfa, beta.shape).copy()
        self.beta = beta

    def matvec(self, v, c):
        c.fill(0)
        Matvec.Helmholtz_matvec(v, c, self.alfa, self.beta, self.A[0],
                                self.A[2], self.B[0], self.axis)
        return c
