from numpy import zeros, ones, arange, pi, float, complex, int, complex128, array
from shenfun.spectralbase import inner_product
from shenfun.chebyshev import bases
from . import LUsolve, Matvec
from scipy.linalg import lu_factor, lu_solve, solve, solve_banded, decomp_cholesky
import scipy.sparse.linalg as la_solve

class Helmholtz(object):
    """Helmholtz solver -u'' + alfa*u = b

    args:
        N          integer       Size of problem in real space
        alfa    float or array   Coefficients. Just one value for 1D problems
                                 and a 2D array for 3D problems.
        basis                    ShenNeumannBasis or ShenDirichletBasis

    """

    def __init__(self, N, alfa, basis):
        # Prepare LU Helmholtz solver for velocity
        self.N = N
        self.alfa = alfa
        self.basis = basis
        self.neumann = True if isinstance(basis, bases.ShenNeumannBasis) else False
        quad = basis.quad
        M = (N-4)//2 if self.neumann else (N-3)//2
        self.s = basis.slice()
        if hasattr(alfa, "__len__"):
            Ny, Nz = alfa.shape
            self.u0 = zeros((2, M+1, Ny, Nz), float)   # Diagonal entries of U
            self.u1 = zeros((2, M, Ny, Nz), float)     # Diagonal+1 entries of U
            self.u2 = zeros((2, M-1, Ny, Nz), float)   # Diagonal+2 entries of U
            self.L  = zeros((2, M, Ny, Nz), float)     # The single nonzero row of L
            LUsolve.LU_Helmholtz_3D(N, self.neumann, quad=="GL", self.alfa, self.u0, self.u1, self.u2, self.L)
        else:
            self.u0 = zeros((2, M+1), float)   # Diagonal entries of U
            self.u1 = zeros((2, M), float)     # Diagonal+1 entries of U
            self.u2 = zeros((2, M-1), float)   # Diagonal+2 entries of U
            self.L  = zeros((2, M), float)     # The single nonzero row of L
            LUsolve.LU_Helmholtz_1D(N, self.neumann, quad=="GL", self.alfa, self.u0, self.u1, self.u2, self.L)
        if not self.neumann:
            self.B = inner_product((basis, 0), (basis, 0))
            self.A = inner_product((basis, 0), (basis, 2))

    def __call__(self, u, b):
        s = self.s
        if isinstance(self.basis, bases.ShenNeumannBasis):
            s = slice(1, self.s.stop)
        if len(u.shape) > 1:
            LUsolve.Solve_Helmholtz_3D_n(self.N, self.neumann, b[s], u[s], self.u0, self.u1, self.u2, self.L)
        else:
            if u.dtype == complex128:
                LUsolve.Solve_Helmholtz_1D(self.N, self.neumann, b[s].real, u[s].real, self.u0, self.u1, self.u2, self.L)
                LUsolve.Solve_Helmholtz_1D(self.N, self.neumann, b[s].imag, u[s].imag, self.u0, self.u1, self.u2, self.L)
            else:
                LUsolve.Solve_Helmholtz_1D(self.N, self.neumann, b[s], u[s], self.u0, self.u1, self.u2, self.L)
        return u

    def matvec(self, v, c):
        assert self.neumann is False
        c[:] = 0
        if len(v.shape) > 1:
            Matvec.Helmholtz_matvec3D(v, c, 1.0, self.alfa**2, self.A[0], self.A[2], self.B[0])
        else:
            Matvec.Helmholtz_matvec(v, c, 1.0, self.alfa**2, self.A[0], self.A[2], self.B[0])
        return c


class Biharmonic(object):
    """Biharmonic solver

      a0*u'''' + alfa*u'' + beta*u = b

    args:
        N            integer        Size of problem in real space
        a0           float          Coefficient
        alfa, beta float/arrays     Coefficients. Just one value for 1D problems
                                    and 2D arrays for 3D problems.
    kwargs:
        quad        ('GL', 'GC')    Chebyshev-Gauss-Lobatto or Chebyshev-Gauss
        solver ('cython', 'python') Choose implementation

    """

    def __init__(self, N, a0, alfa, beta, quad="GL", solver="cython"):
        self.quad = quad
        self.solver = solver
        k = arange(N)
        SB = bases.ShenBiharmonicBasis(N, quad=quad)
        self.S = S = inner_product((SB, 0), (SB, 4))
        self.B = B = inner_product((SB, 0), (SB, 0))
        self.A = A = inner_product((SB, 0), (SB, 2))
        self.a0 = a0
        self.alfa = alfa
        self.beta = beta
        if not solver == "scipy":
            sii, siu, siuu = S[0], S[2], S[4]
            ail, aii, aiu = A[-2], A[0], A[2]
            bill, bil, bii, biu, biuu = B[-4], B[-2], B[0], B[2], B[4]
            M = sii[::2].shape[0]

        if hasattr(beta, "__len__"):
            Ny, Nz = beta.shape
            if solver == "scipy":
                self.Le = Le = []
                self.Lo = Lo = []
                #self.AA = []
                for i in range(Ny):
                    Lej = []
                    Loj = []
                    #AA = []
                    for j in range(Nz):
                        AA = a0*S.diags().toarray() + alfa[i, j]*A.diags().toarray() + beta[i, j]*B.diags().toarray()
                        Ae = AA[::2, ::2]
                        Ao = AA[1::2, 1::2]
                        Lej.append(lu_factor(Ae))
                        Loj.append(lu_factor(Ao))
                        #AA.append((a0*S + alfa*A + beta*B).diags('csr'))
                    Le.append(Lej)
                    Lo.append(Loj)
                    #self.AA.append(AA)
            else:
                self.u0 = zeros((2, M, Ny, Nz))
                self.u1 = zeros((2, M, Ny, Nz))
                self.u2 = zeros((2, M, Ny, Nz))
                self.l0 = zeros((2, M, Ny, Nz))
                self.l1 = zeros((2, M, Ny, Nz))
                self.ak = zeros((2, M, Ny, Nz))
                self.bk = zeros((2, M, Ny, Nz))
                LUsolve.LU_Biharmonic_3D_n(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu, self.u0, self.u1, self.u2, self.l0, self.l1)
                LUsolve.Biharmonic_factor_pr_3D(self.ak, self.bk, self.l0, self.l1)

        else:
            if solver == "scipy":
                #AA = a0*S.diags().toarray() + alfa*A.diags().toarray() + beta*B.diags().toarray()
                #Ae = AA[::2, ::2]
                #Ao = AA[1::2, 1::2]
                #self.Le = lu_factor(Ae)
                #self.Lo = lu_factor(Ao)
                self.AA = (a0*S + alfa*A + beta*B).diags('csr')
            else:
                self.u0 = zeros((2, M))
                self.u1 = zeros((2, M))
                self.u2 = zeros((2, M))
                self.l0 = zeros((2, M))
                self.l1 = zeros((2, M))
                self.ak = zeros((2, M))
                self.bk = zeros((2, M))
                LUsolve.LU_Biharmonic_1D(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu, self.u0, self.u1, self.u2, self.l0, self.l1)
                LUsolve.Biharmonic_factor_pr(self.ak, self.bk, self.l0, self.l1)

    def __call__(self, u, b):
        if len(u.shape) == 3:
            Ny, Nz = u.shape[1:]
            if self.solver == "scipy":
                for i in range(Ny):
                    for j in range(Nz):
                        u[:-4:2, i, j] = lu_solve(self.Le[i][j], b[:-4:2, i, j])
                        u[1:-4:2, i, j] = lu_solve(self.Lo[i][j], b[1:-4:2, i, j])
                        #u[:-4, i, j].real = la_solve.spsolve(self.AA[i][j], b[:-4, i, j].real)
                        #u[:-4, i, j].imag = la_solve.spsolve(self.AA[i][j], b[:-4, i, j].imag)

            else:
                LUsolve.Solve_Biharmonic_3D_n(b, u, self.u0, self.u1, self.u2, self.l0, self.l1, self.ak, self.bk, self.a0)
                #LUsolve.Solve_Biharmonic_3D_com(b, u, self.u0, self.u1, self.u2, self.l0, self.l1, self.ak, self.bk, self.a0)
        else:
            if self.solver == "scipy":
                #u[:-4:2] = lu_solve(self.Le, b[:-4:2])
                #u[1:-4:2] = lu_solve(self.Lo, b[1:-4:2])
                u[:-4] = la_solve.spsolve(self.AA, b[:-4])
                #u[:-4] = solve(self.AA, b[:-4])
            else:
                LUsolve.Solve_Biharmonic_1D(b, u, self.u0, self.u1, self.u2, self.l0, self.l1, self.ak, self.bk, self.a0)

        return u

    def matvec(self, v, c):
        N = v.shape[0]
        c[:] = 0
        if len(v.shape) > 1:
            Matvec.Biharmonic_matvec3D(v, c, self.a0, self.alfa, self.beta, self.S[0], self.S[2],
                                self.S[4], self.A[-2], self.A[0], self.A[2],
                                self.B[-4], self.B[-2], self.B[0], self.B[2], self.B[4])
        else:
            Matvec.Biharmonic_matvec(v, c, self.a0, self.alfa, self.beta, self.S[0], self.S[2],
                                self.S[4], self.A[-2], self.A[0], self.A[2],
                                self.B[-4], self.B[-2], self.B[0], self.B[2], self.B[4])
        return c

