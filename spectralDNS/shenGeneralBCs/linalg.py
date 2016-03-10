from spectralDNS import config
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
	if self.quad == "GC":
	    ck = ones(N[0]-2); ck[0] = 2
	elif self.quad == "GL":
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
    

