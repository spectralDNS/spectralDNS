from cbcdns import config
from numpy import zeros, ones, arange, pi, float, complex, int, complex128
from Matrices import BBBmat, SBBmat, ABBmat
import SFTc
from scipy.linalg import lu_factor, lu_solve, solve

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

class TDMA(object):
    
    def __init__(self, quad="GL", neumann=False):
        self.quad = quad
        self.neumann = neumann
        self.a0 = None
        self.b0 = None
        self.c0 = None
        self.bc = None
        self.s = None
        
    def init(self, N):
        if self.neumann:
            kk = arange(N-2).astype(float)
            ck = ones(N-3, int)
            if self.quad == "GL": ck[-1] = 2
            self.a0 = ones(N-5, float)*(-pi/2)*(kk[1:-2]/(kk[1:-2]+2))**2
            self.b0 = pi/2*(1+ck*(kk[1:]/(kk[1:]+2))**4)
            self.c0 = self.a0.copy()
            self.bc = self.b0.copy()
            self.s = slice(1, N-2)
            
        else:
            ck = ones(N-2, int)
            ck[0] = 2
            if self.quad == "GL":
                ck[-1] = 2
            self.a0 = ones(N-4, float)*(-pi/2)
            self.b0 = (pi/2*(ck+1)).astype(float)
            self.c0 = self.a0.copy()
            self.bc = self.b0.copy()
            self.s = slice(0, N-2) 
        
    def __call__(self, u):
        N = u.shape[0]
        if self.a0 is None:
            self.init(N)
        if len(u.shape) == 3:
            SFTc.TDMA_3D(self.a0, self.b0, self.bc, self.c0, u[self.s])
        elif len(u.shape) == 1:
            SFTc.TDMA_1D(self.a0, self.b0, self.bc, self.c0, u[self.s])
        else:
            raise NotImplementedError
        return u

class PDMA(object):
    
    def __init__(self, quad="GL"):
        self.quad = quad
        self.B = None
        
    def init(self, N):
        self.B = BBBmat(arange(N).astype(float), self.quad)
        self.d0, self.d1, self.d2 = self.B.dd.copy(), self.B.ud.copy(), self.B.uud.copy()
        SFTc.PDMA_SymLU(self.d0, self.d1, self.d2)
        
    def __call__(self, u):
        N = u.shape[0]
        if self.B is None:
            self.init(N)
        if len(u.shape) == 3:
            SFTc.PDMA_Symsolve3D(self.d0, self.d1, self.d2, u[:-4])
        elif len(u.shape) == 1:
            SFTc.PDMA_Symsolve(self.d0, self.d1, self.d2, u[:-4])
        else:
            raise NotImplementedError
        return u

class Biharmonic(object):
    
    def __init__(self, N, a0, alfa, beta, quad="GL", solver="scipy"):
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
                SFTc.LU_Biharmonic_3D(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu, self.u0, self.u1, self.u2, self.l0, self.l1)
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
        