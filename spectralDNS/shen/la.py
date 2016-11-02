from numpy import zeros, ones, arange, pi, float, complex, int, complex128, array
from Matrices import BBBmat, SBBmat, ABBmat, BDDmat, BNNmat, ADDmat
import SFTc
from scipy.linalg import lu_factor, lu_solve, solve, solve_banded, decomp_cholesky
import scipy.sparse.linalg as la_solve
import decimal

class Helmholtz(object):

    def __init__(self, N, alfa, quad="GL", neumann=False):
        # Prepare LU Helmholtz solver for velocity
        self.N = N
        self.alfa = alfa
        self.neumann = neumann
        M = (N-4)/2 if neumann else (N-3)/2
        if hasattr(alfa, "__len__"):
            Ny, Nz = alfa.shape
            self.u0 = zeros((2, M+1, Ny, Nz), float)   # Diagonal entries of U
            self.u1 = zeros((2, M, Ny, Nz), float)     # Diagonal+1 entries of U
            self.u2 = zeros((2, M-1, Ny, Nz), float)   # Diagonal+2 entries of U
            self.L  = zeros((2, M, Ny, Nz), float)     # The single nonzero row of L 
            self.s = slice(1, N-2) if neumann else slice(0, N-2) 
            SFTc.LU_Helmholtz_3D(N, neumann, quad=="GL", self.alfa, self.u0, self.u1, self.u2, self.L)  
        else:
            self.u0 = zeros((2, M+1), float)   # Diagonal entries of U
            self.u1 = zeros((2, M), float)     # Diagonal+1 entries of U
            self.u2 = zeros((2, M-1), float)   # Diagonal+2 entries of U
            self.L  = zeros((2, M), float)     # The single nonzero row of L 
            self.s = slice(1, N-2) if neumann else slice(0, N-2) 
            SFTc.LU_Helmholtz_1D(N, neumann, quad=="GL", self.alfa, self.u0, self.u1, self.u2, self.L)  
        if not neumann:
            self.B = BDDmat(arange(N), quad)
            self.A = ADDmat(arange(N))

    def __call__(self, u, b):
        if len(u.shape) > 1:
            SFTc.Solve_Helmholtz_3D_n(self.N, self.neumann, b[self.s], u[self.s], self.u0, self.u1, self.u2, self.L)
        else:
            if u.dtype == complex128:
                SFTc.Solve_Helmholtz_1D(self.N, self.neumann, b[self.s].real, u[self.s].real, self.u0, self.u1, self.u2, self.L)
                SFTc.Solve_Helmholtz_1D(self.N, self.neumann, b[self.s].imag, u[self.s].imag, self.u0, self.u1, self.u2, self.L)
            else:
                SFTc.Solve_Helmholtz_1D(self.N, self.neumann, b[self.s], u[self.s], self.u0, self.u1, self.u2, self.L)
        return u
    
    def matvec(self, v, c):
        assert self.neumann is False
        c[:] = 0
        if len(v.shape) > 1:
            SFTc.Helmholtz_matvec3D(v, c, 1.0, self.alfa**2, self.A.dd, self.A.ud[0], self.B.dd)
        else:
            SFTc.Helmholtz_matvec(v, c, 1.0, self.alfa**2, self.A.dd, self.A.ud[0], self.B.dd)
        return c
    

class TDMA(object):
    
    def __init__(self, quad="GL", neumann=False):
        self.quad = quad
        self.neumann = neumann
        self.dd = zeros(0)
        self.ud = zeros(0)
        
    def init(self, N):
        kk = arange(N).astype(float)
        if self.neumann:            
            self.M = BNNmat(kk, self.quad)
            self.dd = self.M.dd
            self.ud = self.M.ud
            self.L = zeros(N-4)
            SFTc.TDMA_SymLU(self.dd, self.ud, self.L)
            self.s = slice(1, N-2)
            
        else:
            self.M = BDDmat(kk, self.quad)
            self.dd = self.M.dd
            self.ud = ones(N-4)*self.M.ud
            self.L = zeros(N-4)
            self.s = slice(0, N-2) 
            SFTc.TDMA_SymLU(self.dd, self.ud, self.L)

    def __call__(self, u):
        N = u.shape[0]
        if not self.dd.shape[0] == u.shape[0]:
            self.init(N)
        if len(u.shape) == 3:
            #SFTc.TDMA_3D(self.ud, self.dd, self.dd.copy(), self.ud.copy(), u[self.s])
            SFTc.TDMA_SymSolve3D(self.dd, self.ud, self.L, u[self.s])
        elif len(u.shape) == 1:
            #SFTc.TDMA_1D(self.ud, self.dd, self.dd.copy(), self.ud.copy(), u[self.s])
            SFTc.TDMA_SymSolve(self.dd, self.ud, self.L, u[self.s])
        else:
            raise NotImplementedError
        return u

class PDMA(object):
    
    def __init__(self, quad="GL", solver="cython"):
        self.quad = quad
        self.B = zeros(0)
        self.solver = solver
        
    def init(self, N):
        self.B = BBBmat(arange(N), self.quad)
        if self.solver == "cython":
            self.d0, self.d1, self.d2 = self.B.dd.copy(), self.B.ud.copy(), self.B.uud.copy()
            SFTc.PDMA_SymLU(self.d0, self.d1, self.d2)
            #self.SymLU(self.d0, self.d1, self.d2)
            ##self.d0 = self.d0.astype(float)
            ##self.d1 = self.d1.astype(float)
            ##self.d2 = self.d2.astype(float)
        else:
            #self.L = lu_factor(self.B.diags().toarray())
            self.d0, self.d1, self.d2 = self.B.dd.copy(), self.B.ud.copy(), self.B.uud.copy()
            #self.A = zeros((9, N-4))
            #self.A[0, 4:] = self.d2
            #self.A[2, 2:] = self.d1
            #self.A[4, :] = self.d0
            #self.A[6, :-2] = self.d1
            #self.A[8, :-4] = self.d2
            self.A = zeros((5, N-4))
            self.A[0, 4:] = self.d2
            self.A[2, 2:] = self.d1
            self.A[4, :] = self.d0
            self.L = decomp_cholesky.cholesky_banded(self.A)
        
    def SymLU(self, d, e, f):
        n = d.shape[0]
        m = e.shape[0]
        k = n - m
        
        for i in xrange(n-2*k):
            lam = e[i]/d[i]
            d[i+k] -= lam*e[i]
            e[i+k] -= lam*f[i]
            e[i] = lam
            lam = f[i]/d[i]
            d[i+2*k] -= lam*f[i]
            f[i] = lam

        lam = e[n-4]/d[n-4]
        d[n-2] -= lam*e[n-4]
        e[n-4] = lam
        lam = e[n-3]/d[n-3]
        d[n-1] -= lam*e[n-3]
        e[n-3] = lam

    def SymSolve(self, d, e, f, b):
        n = d.shape[0]
        #bc = array(map(decimal.Decimal, b))
        bc = b
        
        bc[2] -= e[0]*bc[0]
        bc[3] -= e[1]*bc[1]    
        for k in range(4, n):
            bc[k] -= (e[k-2]*bc[k-2] + f[k-4]*bc[k-4])
    
        bc[n-1] /= d[n-1]
        bc[n-2] /= d[n-2]    
        bc[n-3] /= d[n-3] 
        bc[n-3] -= e[n-3]*bc[n-1]
        bc[n-4] /= d[n-4]
        bc[n-4] -= e[n-4]*bc[n-2]    
        for k in range(n-5,-1,-1):
            bc[k] /= d[k] 
            bc[k] -= (e[k]*bc[k+2] + f[k]*bc[k+4])
        b[:] = bc.astype(float)
        
    def __call__(self, u):
        N = u.shape[0]
        if not self.B.shape[0] == u.shape[0]:
            self.init(N)
        if len(u.shape) == 3:
            if self.solver == "cython":
                SFTc.PDMA_Symsolve3D(self.d0, self.d1, self.d2, u[:-4])
            else:
                b = u.copy()
                for i in range(u.shape[1]):
                    for j in range(u.shape[2]):
                        #u[:-4, i, j] = lu_solve(self.L, b[:-4, i, j])
                        u[:-4, i, j] = la_solve.spsolve(self.B.diags(), b[:-4, i, j])
                        
        elif len(u.shape) == 1:
            if self.solver == "cython":
                SFTc.PDMA_Symsolve(self.d0, self.d1, self.d2, u[:-4])
                #self.SymSolve(self.d0, self.d1, self.d2, u[:-4])
            else:
                b = u.copy()
                #u[:-4] = lu_solve(self.L, b[:-4])
                #u[:-4] = la_solve.spsolve(self.B.diags(), b[:-4])
                #u[:-4] = solve_banded((4, 4), self.A, b[:-4])
                u[:-4] = decomp_cholesky.cho_solve_banded((self.L, False), b[:-4])
        else:
            raise NotImplementedError
        
        return u

class Biharmonic(object):

    def __init__(self, N, a0, alfa, beta, quad="GL", solver="cython"):
        self.quad = quad
        self.solver = solver
        k = arange(N)
        self.S = S = SBBmat(k)
        self.B = B = BBBmat(k, self.quad)
        self.A = A = ABBmat(k)
        self.a0 = a0
        self.alfa = alfa
        self.beta = beta
        if not solver == "scipy":
            sii, siu, siuu = S.dd, S.ud[0], S.ud[1]
            ail, aii, aiu = A.ld, A.dd, A.ud
            bill, bil, bii, biu, biuu = B.lld, B.ld, B.dd, B.ud, B.uud
            M = sii[::2].shape[0]
        
        if hasattr(beta, "__len__"):
            Ny, Nz = beta.shape
            if solver == "scipy":
                self.Le = Le = []
                self.Lo = Lo = []
                for i in range(Ny):
                    Lej = []
                    Loj = []
                    for j in range(Nz):
                        AA = a0*S.diags().toarray() + alfa[i, j]*A.diags().toarray() + beta[i, j]*B.diags().toarray()
                        Ae = AA[::2, ::2]
                        Ao = AA[1::2, 1::2]
                        Lej.append(lu_factor(Ae))
                        Loj.append(lu_factor(Ao))
                    Le.append(Lej)
                    Lo.append(Loj)
            else:
                self.u0 = zeros((2, M, Ny, Nz))
                self.u1 = zeros((2, M, Ny, Nz))
                self.u2 = zeros((2, M, Ny, Nz))
                self.l0 = zeros((2, M, Ny, Nz))
                self.l1 = zeros((2, M, Ny, Nz))
                self.ak = zeros((2, M, Ny, Nz))
                self.bk = zeros((2, M, Ny, Nz))
                SFTc.LU_Biharmonic_3D_n(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu, self.u0, self.u1, self.u2, self.l0, self.l1)
                SFTc.Biharmonic_factor_pr_3D(self.ak, self.bk, self.l0, self.l1)

        else:
            if solver == "scipy":
                AA = a0*S.diags().toarray() + alfa*A.diags().toarray() + beta*B.diags().toarray()
                Ae = AA[::2, ::2]
                Ao = AA[1::2, 1::2]
                self.Le = lu_factor(Ae)
                self.Lo = lu_factor(Ao)
            else:
                self.u0 = zeros((2, M))
                self.u1 = zeros((2, M))
                self.u2 = zeros((2, M))
                self.l0 = zeros((2, M))
                self.l1 = zeros((2, M))
                self.ak = zeros((2, M))
                self.bk = zeros((2, M))
                SFTc.LU_Biharmonic_1D(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu, self.u0, self.u1, self.u2, self.l0, self.l1)
                SFTc.Biharmonic_factor_pr(self.ak, self.bk, self.l0, self.l1)
        
    def __call__(self, u, b):
        if len(u.shape) == 3:
            Ny, Nz = u.shape[1:]
            if self.solver == "scipy":
                for i in range(Ny):
                    for j in range(Nz):
                        u[:-4:2, i, j] = lu_solve(self.Le[i][j], b[:-4:2, i, j])
                        u[1:-4:2, i, j] = lu_solve(self.Lo[i][j], b[1:-4:2, i, j])
            else:
                SFTc.Solve_Biharmonic_3D_n(b, u, self.u0, self.u1, self.u2, self.l0, self.l1, self.ak, self.bk, self.a0)
        else:
            if self.solver == "scipy":
                u[:-4:2] = lu_solve(self.Le, b[:-4:2])
                u[1:-4:2] = lu_solve(self.Lo, b[1:-4:2])
            else:
                SFTc.Solve_Biharmonic_1D(b, u, self.u0, self.u1, self.u2, self.l0, self.l1, self.ak, self.bk, self.a0)

        return u
    
    def matvec(self, v, c):
        N = v.shape[0]
        c[:] = 0
        if len(v.shape) > 1:
            SFTc.Biharmonic_matvec3D(v, c, self.a0, self.alfa, self.beta, self.S.dd, self.S.ud[0], 
                                self.S.ud[1], self.A.ld, self.A.dd, self.A.ud,
                                self.B.lld, self.B.ld, self.B.dd, self.B.ud, self.B.uud)
        else:
            SFTc.Biharmonic_matvec(v, c, self.a0, self.alfa, self.beta, self.S.dd, self.S.ud[0], 
                                self.S.ud[1], self.A.ld, self.A.dd, self.A.ud,
                                self.B.lld, self.B.ld, self.B.dd, self.B.ud, self.B.uud)
        return c
        