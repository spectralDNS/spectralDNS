from cbcdns import config
from numpy import zeros, ones, arange, pi, mgrid, float, complex, int
import SFTc
from ShenMatrices import B_matrix, C_matrix, A_matrix, D_matrix 


class Shen(object):
    
    def __init__(self, N, quad="GL", Neumann=False):
        # Prepare LU Helmholtz solver for velocity
        self.N = N
        self.Neumann = Neumann    

    def wavenumbers(self, N):
	    if isinstance(N, tuple):
		if len(N) == 1:
		    N = N[0]
	    if isinstance(N, int): 
		return arange(N-2).astype(float)
	    else:
		kk = mgrid[:N[0]-2, :N[1], :N[2]].astype(float)
		return kk[0]

    def chebNormalizationFactor(self, N, quad):
	if self.quad == "GL":
	    ck = ones(N[0]-2); ck[0] = 2
	elif self.quad == "GC":
	    ck = ones(N[0]-2); ck[0] = 2; ck[-1] = 2
	return ck

    def shenCoefficients(self, k, BC):
	"""
	Shen basis functions given by
	phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},
	satisfy the imposed boundary conditions for a unique set of {a_k, b_k}.  
	"""
	am = BC[0]; bm = BC[1]; cm = BC[2]
	ap = BC[3]; bp = BC[4]; cp = BC[5]

	detk = 2*am*ap + ((k + 1.)**2 + (k + 2.)**2)*(am*bp - ap*bm) - 2.*bm*bp*(k + 1.)**2*(k + 2.)**2

	Aa = am - bm*(k + 2.)**2; Ab= -ap - bp*(k + 2.)**2  
	Ac = am - bm*(k + 1.)**2; Ad= ap + bp*(k + 1.)**2

	y1 = -ap - bp*k**2 + cp; y2= -am + bm*k**2 + cm/((-1)**k) 

	ak = (1./detk)*(Aa*y1 + Ab*y2)
	bk = (1./detk)*(Ac*y1 + Ad*y2)

	return ak, bk

class UTDMA(Shen):
    
    def __init__(self, N, BC, quad="GL", Neumann=False, dim = "1"):
        self.quad = quad
        self.BC = BC
        self.Neumann = Neumann
        self.dim = dim
        self.N = N
        self.k = self.wavenumbers(self.N)
        self.ak, self.bk = self.shenCoefficients(self.k, self.BC)

    def __call__(self, u, b):
        if self.dim=="1":
            if self.Neumann:
                SFTc.UTDMA_1D_Neumann(self.bk, u, b)
            else:
                SFTc.UTDMA_1D(self.ak, self.bk, u, b)
        elif self.dim=="3":
            if self.Neumann:
                SFTc.UTDMA_Neumann(self.bk, u, b)
            else:
                SFTc.UTDMA(self.ak, self.bk, u, b)
        return b

class PDMA(Shen):        
    
    def __init__(self, N, BC, quad="GL", dim = "1"):
        self.quad = quad
        self.BC = BC
        self.k = self.wavenumbers(N)
        self.ak, self.bk = self.shenCoefficients(self.k, self.BC)          
        self.a0 = zeros(N[0]-2)
        self.b0 = zeros(N[0]-2)
        self.kk= zeros(N[0])
        self.Bm = B_matrix(self.kk, self.quad, self.ak, self.bk, self.ak, self.bk)
        self.a0[2:], self.b0[1:], self.c0, self.d0, self.e0 = self.Bm.diags()

    def __call__(self, u, b):
	if self.dim=="1":
	    SFTc.PDMA_1D(self.a0, self.b0, self.c0, self.d0, self.e0, u, b)
        elif self.dim=="3":
	    SFTc.PDMA(self.a0, self.b0, self.c0, self.d0, self.e0, u, b)
        return b
    

#class Helmholtz(object):

    #def __init__(self, N, alfa, quad="GL", neumann=False):
        ## Prepare LU Helmholtz solver for velocity
        #self.N = N
        #self.alfa = alfa
        #self.neumann = neumann
        #M = (N-4)/2 if neumann else (N-3)/2
        #Ny, Nz = alfa.shape
        #self.u0 = zeros((2, M+1, Ny, Nz), float)   # Diagonal entries of U
        #self.u1 = zeros((2, M, Ny, Nz), float)     # Diagonal+1 entries of U
        #self.u2 = zeros((2, M-1, Ny, Nz), float)   # Diagonal+2 entries of U
        #self.L  = zeros((2, M, Ny, Nz), float)     # The single nonzero row of L 
        #self.s = slice(1, N-2) if neumann else slice(0, N-2) 
        #SFTc.LU_Helmholtz_3D(N, neumann, quad=="GL", self.alfa, self.u0, self.u1, self.u2, self.L)  

 
    #def __call__(self, u, b):
        #SFTc.Solve_Helmholtz_3D_complex(self.N, self.neumann, b[self.s], u[self.s], self.u0, self.u1, self.u2, self.L)
        #return u

#class TDMA(object):
    
    #def __init__(self, quad="GL", neumann=False):
        #self.quad = quad
        #self.neumann = neumann
        #self.a0 = None
        #self.b0 = None
        #self.c0 = None
        #self.bc = None
        #self.s = None
        
    #def init(self, N):
        #if self.neumann:
            #kk = arange(N-2).astype(float)
            #ck = ones(N-3, int)
            #if self.quad == "GL": ck[-1] = 2
            #self.a0 = ones(N-5, float)*(-pi/2)*(kk[1:-2]/(kk[1:-2]+2))**2
            #self.b0 = pi/2*(1+ck*(kk[1:]/(kk[1:]+2))**4)
            #self.c0 = self.a0.copy()
            #self.bc = self.b0.copy()
            #self.s = slice(1, N-2)
            
        #else:
            #ck = ones(N-2, int)
            #ck[0] = 2
            #if self.quad == "GL":
                #ck[-1] = 2
            #self.a0 = ones(N-4, float)*(-pi/2)
            #self.b0 = (pi/2*(ck+1)).astype(float)
            #self.c0 = self.a0.copy()
            #self.bc = self.b0.copy()
            #self.s = slice(0, N-2) 
        
    #def __call__(self, u):
        #N = u.shape[0]
        #if self.a0 is None:
            #self.init(N)
        #if len(u.shape) == 3:
            #SFTc.TDMA_3D(self.a0, self.b0, self.bc, self.c0, u[self.s])
        #elif len(u.shape) == 1:
            #SFTc.TDMA_1D(self.a0, self.b0, self.bc, self.c0, u[self.s])
        #else:
            #raise NotImplementedError
        #return u



    
    