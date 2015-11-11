from cbcdns import config
from numpy import zeros, ones, arange, pi, float, complex, int
import SFTc

class Helmholtz(object):

    def __init__(self, N, alfa, quad="GL", neumann=False):
        # Prepare LU Helmholtz solver for velocity
        self.N = N
        self.alfa = alfa
        self.neumann = neumann
        M = (N-4)/2 if neumann else (N-3)/2
        Ny, Nz = alfa.shape
        self.u0 = zeros((2, M+1, Ny, Nz), float)   # Diagonal entries of U
        self.u1 = zeros((2, M, Ny, Nz), float)     # Diagonal+1 entries of U
        self.u2 = zeros((2, M-1, Ny, Nz), float)   # Diagonal+2 entries of U
        self.L  = zeros((2, M, Ny, Nz), float)     # The single nonzero row of L 
        self.s = slice(1, N-2) if neumann else slice(0, N-2) 
        SFTc.LU_Helmholtz_3D(N, neumann, quad=="GL", self.alfa, self.u0, self.u1, self.u2, self.L)  

    def __call__(self, u, b):
        SFTc.Solve_Helmholtz_3D_complex(self.N, self.neumann, b[self.s], u[self.s], self.u0, self.u1, self.u2, self.L)
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
            u[self.s] = SFTc.TDMA_3D(self.a0, self.b0, self.bc, self.c0, u[self.s])
        elif len(u.shape) == 1:
            u[self.s] = SFTc.TDMA_1D(self.a0, self.b0, self.c0, u[self.s])
        else:
            raise NotImplementedError
        return u
