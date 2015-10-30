from cbcdns import config
from numpy import zeros, ones, arange, pi
if config.precision == "double":
    import SFTc_double as SFTc
else:
    import SFTc_single as SFTc

class Helmholtz(object):

    def __init__(self, N, alfa, quad="GL", neumann=False):
        # Prepare LU Helmholtz solver for velocity
        self.N = N
        self.alfa = alfa
        self.neumann = neumann
        M = (N[0]-4)/2 if neumann else (N[0]-3)/2
        Ny, Nz = alfa.shape
        self.u0 = zeros((2, M+1, Ny, Nz))   # Diagonal entries of U
        self.u1 = zeros((2, M, Ny, Nz))     # Diagonal+1 entries of U
        self.u2 = zeros((2, M-1, Ny, Nz))   # Diagonal+2 entries of U
        self.L  = zeros((2, M, Ny, Nz))     # The single nonzero row of L 
        self.s = slice(1, N[0]-2) if neumann else slice(0, N[0]-2) 
        SFTc.LU_Helmholtz_3D(N[0], neumann, quad=="GL", self.alfa, self.u0, self.u1, self.u2, self.L)  

    def __call__(self, u, b):
        SFTc.Solve_Helmholtz_3D_complex(self.N[0], self.neumann, b[self.s], u[self.s], self.u0, self.u1, self.u2, self.L)
        return u

class TDMA(object):
    
    def __init__(self, N, quad="GL", neumann=False):
        if neumann:
            kk = arange(N[0]-2)
            ck = ones(N[0]-3)
            if quad == "GL": ck[-1] = 2
            self.a0 = ones(N[0]-5)*(-pi/2)*(kk[1:-2]/(kk[1:-2]+2))**2
            self.b0 = pi/2*(1+ck*(kk[1:]/(kk[1:]+2))**4)
            self.c0 = self.a0.copy()
            self.bc = self.b0.copy()
            self.s = slice(1, N[0]-2)
            
        else:
            ck = ones(N[0]-2)
            ck[0] = 2
            if quad == "GL":
                ck[-1] = 2
            self.a0 = ones(N[0]-4)*(-pi/2)
            self.b0 = pi/2*(ck+1)
            self.c0 = self.a0.copy()
            self.bc = self.b0.copy()
            self.s = slice(0, N[0]-2) 
        
    def __call__(self, u):
        u[self.s] = SFTc.TDMA_3D(self.a0, self.b0, self.bc, self.c0, u[self.s])
        return u