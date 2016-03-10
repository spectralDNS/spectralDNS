import pytest
from spectralDNS.shen.shentransform import ShenDirichletBasis, ShenNeumannBasis, ChebyshevTransform, ShenBiharmonicBasis
from spectralDNS.shen.la import TDMA, Helmholtz, Biharmonic
from spectralDNS.shen.Matrices import BNNmat, BTTmat, BDDmat, CDDmat, CDNmat, BNDmat, CNDmat, BDNmat, ADDmat, ANNmat, CTDmat, BDTmat, CDTmat, BTDmat, BTNmat, BBBmat, ABBmat, SBBmat, CDBmat, CBDmat, ATTmat, BBDmat, HelmholtzCoeff
from spectralDNS.shen import SFTc
from scipy.linalg import solve

from spectralDNS import config
config.mesh = "channel"
config.solver = "IPCS"
from spectralDNS.mesh.channel import FastShenFourierTransform
from mpi4py import MPI

from sympy import chebyshevt, Symbol, sin, cos, pi
import numpy as np
import scipy.sparse.linalg as la

N = 12
x = Symbol("x")

@pytest.fixture(params=("NGC", "NGL", "DGC", "DGL", "CGC", "CGL", "BGL", "BGC"))
def ST(request):
    if request.param[0] == 'N':
        return ShenNeumannBasis(request.param[1:])
    elif request.param[0] == 'D':
        return ShenDirichletBasis(request.param[1:])
    elif request.param[0] == 'C':
        return ChebyshevTransform(request.param[1:])
    elif request.param[0] == 'B':
        return ShenBiharmonicBasis(request.param[1:])

@pytest.fixture(params=("NGC", "NGL", "DGC", "DGL"))
def ST2(request):
    if request.param[0] == 'N':
        return ShenNeumannBasis(request.param[1:])
    elif request.param[0] == 'D':
        return ShenDirichletBasis(request.param[1:])

@pytest.fixture(params=("GC", "GL"))
def SD(request):
    return ShenDirichletBasis(request.param)

@pytest.fixture(params=("GC", "GL"))
def SB(request):
    return ShenBiharmonicBasis(request.param)

@pytest.fixture(params=("GCGC", "GLGL"))
def SDSN(request):
    return (ShenDirichletBasis(request.param[:2]), ShenNeumannBasis(request.param[2:]))

@pytest.fixture(params=("GCGC", "GLGL"))
def SBSD(request):
    return (ShenBiharmonicBasis(request.param[:2]), ShenDirichletBasis(request.param[:2]))

@pytest.fixture(params=("GCGC1", "GLGL1", "GCGC2", "GLGL2"))
def S1S2(request):
    if request.param[-1] == "1":
        return (ShenDirichletBasis(request.param[:2]), ShenNeumannBasis(request.param[2:4]))
    elif request.param[-1] == "2":
        return (ShenNeumannBasis(request.param[2:4]), ShenDirichletBasis(request.param[:2]))

@pytest.fixture(params=("GCGC1", "GLGL1", "GCGC3"))
def SXSX(request):
    if request.param[-1] == "1":
        return (ShenDirichletBasis(request.param[:2]), ShenNeumannBasis(request.param[2:4]))
    elif request.param[-1] == "2":
        return (ShenNeumannBasis(request.param[2:4]), ShenDirichletBasis(request.param[:2]))
    elif request.param[-1] == "3":
        return (ShenDirichletBasis(request.param[2:4]), ShenDirichletBasis(request.param[:2]))
    elif request.param[-1] == "4":
        return (ShenNeumannBasis(request.param[2:4]), ShenNeumannBasis(request.param[:2]))

@pytest.fixture(params=("NGC", "NGL", "DGC", "DGL"))
def T(request):
    if request.param[0] == 'N':
        return TDMA(request.param[1:], True)
    elif request.param[0] == 'D':
        return TDMA(request.param[1:], False)

@pytest.fixture(params=("GCGC", "GLGL"))
def SDST(request):
    return (ShenDirichletBasis(request.param[:2]), ChebyshevTransform(request.param[2:]))

@pytest.fixture(params=("GCGC", "GLGL"))
def SBST(request):
    return (ShenBiharmonicBasis(request.param[:2]), ShenDirichletBasis(request.param[2:]))

@pytest.fixture(params=("GCGC1", "GLGL1", "GCGC2", "GLGL2"))
def SXST(request):
    if request.param[-1] == "1":
        return (ShenDirichletBasis(request.param[:2]), ChebyshevTransform(request.param[2:-1]))
    elif request.param[-1] == "2":
        return (ShenNeumannBasis(request.param[:2]), ChebyshevTransform(request.param[2:-1]))

def test_scalarproduct(ST):
    """Test fast scalar product against Vandermonde computed version"""
    points, weights = ST.points_and_weights(N)
    f = x*x+cos(pi*x)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)
    u0 = np.zeros(N)
    u1 = np.zeros(N)
    if ST.__class__.__name__ == "ChebyshevTransform":
        u0 = ST.fastChebScalar(fj, u0)
    else:
        u0 = ST.fastShenScalar(fj, u0)
    ST.fast_transform = False
    if ST.__class__.__name__ == "ChebyshevTransform":
        u1 = ST.fastChebScalar(fj, u1)
    else:
        u1 = ST.fastShenScalar(fj, u1)
    assert np.allclose(u1, u0)

def test_TDMA(T):
    from scipy.linalg import solve
    if T.neumann == True:
        B = BNNmat(np.arange(N).astype(np.float), T.quad)
        s = slice(1, N-2)
    else:
        B = BDDmat(np.arange(N).astype(np.float), T.quad)
        s = slice(0, N-2)
    Ba = B.diags().toarray()
    f = np.random.random(N)
    u = solve(Ba, f[s])

    u0 = f.copy()
    u0 = T(u0)
    assert np.allclose(u0[s], u)
    # Again
    u0 = f.copy()
    u0 = T(u0)
    assert np.allclose(u0[s], u)
    
    
    # Multidimensional version
    fc = f.repeat(16).reshape((N, 4, 4))
    fc = T(fc)    
    assert np.allclose(fc[s, 2, 2], u)
    fc = f.repeat(16).reshape((N, 4, 4))
    fc = T(fc)    
    assert np.allclose(fc[s, 2, 2], u)
    
def test_BNNmat(ST):
    points, weights = ST.points_and_weights(N)
    f_hat = np.zeros(N)
    fj = np.random.random(N)    
    u0 = np.zeros(N)
    if ST.__class__.__name__ == "ShenNeumannBasis":
        B = BNNmat(np.arange(N).astype(np.float), ST.quad)
        f_hat = ST.fst(fj, f_hat)
        fj = ST.ifst(f_hat, fj)
        u0 = ST.fastShenScalar(fj, u0)
        f_hat = ST.fst(fj, f_hat)
    elif ST.__class__.__name__ == "ShenDirichletBasis":
        B = BDDmat(np.arange(N).astype(np.float), ST.quad)
        f_hat = ST.fst(fj, f_hat)
        fj = ST.ifst(f_hat, fj)
        u0 = ST.fastShenScalar(fj, u0)
        f_hat = ST.fst(fj, f_hat)
    elif ST.__class__.__name__ == "ShenBiharmonicBasis":
        B = BBBmat(np.arange(N).astype(np.float), ST.quad)
        f_hat = ST.fst(fj, f_hat)
        fj = ST.ifst(f_hat, fj)
        u0 = ST.fastShenScalar(fj, u0)
        f_hat = ST.fst(fj, f_hat)        
    else:
        B = BTTmat(np.arange(N).astype(np.float), ST.quad)
        f_hat = ST.fct(fj, f_hat)
        fj = ST.ifct(f_hat, fj)
        u0 = ST.fastChebScalar(fj, u0)
        f_hat = ST.fct(fj, f_hat)

    u2 = B.matvec(f_hat)
    #from IPython import embed; embed()
    assert np.allclose(u2, u0)
    
    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    f_hat = np.zeros((N, 4, 4), dtype=np.complex)
    if ST.__class__.__name__ in ("ChebyshevTransform"):
        u0 = ST.fastChebScalar(fj, u0)
        f_hat = ST.fct(fj, f_hat)
    else:
        u0 = ST.fastShenScalar(fj, u0)
        f_hat = ST.fst(fj, f_hat)
    u2 = B.matvec(f_hat)
    assert np.allclose(u2, u0)

#test_BNNmat(ShenBiharmonicBasis("GL"))

def test_BDNmat(S1S2):
    S1, S2 = S1S2
    
    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = S2.fst(fj, f_hat)
    fj = S2.ifst(f_hat, fj)
    
    if S1.__class__.__name__ == "ShenNeumannBasis":
        B = BNDmat(np.arange(N).astype(np.float), S1.quad)
    else:
        B = BDNmat(np.arange(N).astype(np.float), S1.quad)
    
    f_hat = S2.fst(fj, f_hat)
    u2 = B.matvec(f_hat)
    u0 = np.zeros(N)
    u0 = S1.fastShenScalar(fj, u0)
    
    assert np.allclose(u0, u2)
    
    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    f_hat = f_hat.repeat(16).reshape((N, 4, 4)) + 1j*f_hat.repeat(16).reshape((N, 4, 4))
    
    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = S1.fastShenScalar(fj, u0)    
    u2 = B.matvec(f_hat)
    assert np.linalg.norm(u2-u0)/(N*16) < 1e-12    
    

def test_BDTmat(SDST):
    SD, ST = SDST
    
    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = ST.fct(fj, f_hat)
    fj = ST.ifct(f_hat, fj)
    
    B = BDTmat(np.arange(N).astype(np.float), SD.quad)
    
    f_hat = ST.fct(fj, f_hat)
    u2 = B.matvec(f_hat)
    u0 = np.zeros(N)
    u0 = SD.fastShenScalar(fj, u0)
    
    #from IPython import embed; embed()
    assert np.allclose(u0, u2)
    
    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    f_hat = f_hat.repeat(16).reshape((N, 4, 4)) + 1j*f_hat.repeat(16).reshape((N, 4, 4))
    
    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = SD.fastShenScalar(fj, u0)    
    u2 = B.matvec(f_hat)
    assert np.linalg.norm(u2-u0)/(N*16) < 1e-12    

#test_BDTmat((ShenDirichletBasis("GL"), ShenNeumannBasis("GL")))

def test_BBDmat(SBSD):
    SB, SD = SBSD
    
    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = SD.fst(fj, f_hat)
    fj = SD.ifst(f_hat, fj)
    
    B = BBDmat(np.arange(N).astype(np.float), SB.quad)
    
    f_hat = SD.fst(fj, f_hat)
    u2 = B.matvec(f_hat)
    u0 = np.zeros(N)
    u0 = SB.fastShenScalar(fj, u0)
    
    #from IPython import embed; embed()
    assert np.allclose(u0, u2)
    
    # Multidimensional version
    fj = fj.repeat(N*N).reshape((N, N, N)) + 1j*fj.repeat(N*N).reshape((N, N, N))
    f_hat = f_hat.repeat(N*N).reshape((N, N, N)) + 1j*f_hat.repeat(N*N).reshape((N, N, N))
        
    u0 = np.zeros((N, N, N), dtype=np.complex)
    u0 = SB.fastShenScalar(fj, u0)    
    u2 = B.matvec(f_hat)
    assert np.linalg.norm(u2-u0)/(N*N*N) < 1e-12    
    
    FST = FastShenFourierTransform(np.array([N, N, N]), np.array([2*pi, 2*pi, 2*pi]), MPI)
    f_hat = np.zeros(FST.complex_shape(), dtype=np.complex)
    fj = np.random.random((N, N, N))
    f_hat = FST.fst(fj, f_hat, SD)
    fj = FST.ifst(f_hat, fj, SD)
    
    z0 = B.matvec(f_hat)
    z1 = z0.copy()*0
    z1 = FST.fss(fj, z1, SB)
    assert np.linalg.norm(z1-z0)/(N*N*N) < 1e-12    


#test_BBDmat((ShenBiharmonicBasis("GL"), ShenDirichletBasis("GL")))

def test_BTXmat(SXST):
    SX, ST = SXST
    
    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = SX.fst(fj, f_hat)
    fj = SX.ifst(f_hat, fj)
    
    if SX.__class__.__name__ == "ShenDirichletBasis":
        B = BTDmat(np.arange(N).astype(np.float), ST.quad)
    if SX.__class__.__name__ == "ShenNeumannBasis":
        B = BTNmat(np.arange(N).astype(np.float), ST.quad)
    
    f_hat = SX.fst(fj, f_hat)
    u2 = B.matvec(f_hat)
    u0 = np.zeros(N)
    u0 = ST.fastChebScalar(fj, u0)
    
    #from IPython import embed; embed()
    assert np.allclose(u0, u2)
    
    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    f_hat = f_hat.repeat(16).reshape((N, 4, 4)) + 1j*f_hat.repeat(16).reshape((N, 4, 4))
    
    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = ST.fastChebScalar(fj, u0)    
    u2 = B.matvec(f_hat)
    assert np.linalg.norm(u2-u0)/(N*16) < 1e-12    

#test_BTXmat((ShenDirichletBasis("GL"), ChebyshevTransform("GL")))

def test_transforms(ST):
    points, weights = ST.points_and_weights(N)
    fj = np.random.random(N)    

    # Project function to space first
    if not ST.__class__.__name__ == "ChebyshevTransform":
        f_hat = np.zeros(N)
        f_hat = ST.fst(fj, f_hat)
        fj = ST.ifst(f_hat, fj)

    # Then check if transformations work as they should
    u0 = np.zeros(N)    
    u1 = np.zeros(N)
    if ST.__class__.__name__ == "ChebyshevTransform":
        u0 = ST.fct(fj, u0)
        u1 = ST.ifct(u0, u1)
    else:
        u0 = ST.fst(fj, u0)
        u1 = ST.ifst(u0, u1)

    #from IPython import embed; embed()
    assert np.allclose(fj, u1)
    
    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    u0 = np.zeros((N, 4, 4), dtype=np.complex)    
    u1 = np.zeros((N, 4, 4), dtype=np.complex)
    if ST.__class__.__name__ == "ChebyshevTransform":
        u0 = ST.fct(fj, u0)
        u1 = ST.ifct(u0, u1)
    else:
        u0 = ST.fst(fj, u0)
        u1 = ST.ifst(u0, u1)
    
    assert np.allclose(fj, u1)

#test_transforms(ShenBiharmonicBasis("GC"))

def test_FST(ST):
    FST = FastShenFourierTransform(np.array([N, 4, 4]), np.array([2*pi, 2*pi, 2*pi]), MPI)
    points, weights = ST.points_and_weights(N)
    fj = np.random.random((N,4,4))    
    f_hat = fj.copy()
    
    if not ST.__class__.__name__ == "ChebyshevTransform":
        f_hat = ST.fst(fj, f_hat)
        fj = ST.ifst(f_hat, fj)

    # Then check if transformations work as they should
    u_hat = np.zeros((N,4,3), dtype=np.complex)
    u0 = np.zeros((N,4,4))
    if ST.__class__.__name__ == "ChebyshevTransform":
        u_hat = FST.fct(fj, u_hat, ST)
        u0 = FST.ifct(u_hat, u0, ST)
    else:
        u_hat = FST.fst(fj, u_hat, ST)
        u0 = FST.ifst(u_hat, u0, ST)

    #from IPython import embed; embed()
    assert np.allclose(fj, u0)

#test_FST(ShenBiharmonicBasis("GC"))    
    
def test_CDDmat(SD):
    M = 256
    u = (1-x**2)*sin(np.pi*6*x)
    dudx = u.diff(x, 1)
    points, weights = SD.points_and_weights(M)
    dudx_j = np.array([dudx.subs(x, h) for h in points], dtype=np.float)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    
    dudx_j = np.zeros(M)
    u_hat = np.zeros(M)
    u_hat = SD.fst(uj, u_hat)
    uj = SD.ifst(u_hat, uj)
    u_hat = SD.fst(uj, u_hat)
    
    uc_hat = np.zeros(M)
    uc_hat = SD.fct(uj, uc_hat)
    du_hat = np.zeros(M)
    dudx_j = SD.fastChebDerivative(uj, dudx_j)
    
    Cm = CDDmat(np.arange(M).astype(np.float))
    TDMASolver = TDMA(SD.quad, False)
    
    cs = Cm.matvec(u_hat)
    
    # Should equal (but not exact so use extra resolution)
    cs2 = np.zeros(M)
    cs2 = SD.fastShenScalar(dudx_j, cs2)
    
    assert np.allclose(cs, cs2)
    
    cs = TDMASolver(cs)
    du = np.zeros(M)
    du = SD.ifst(cs, du)

    assert np.linalg.norm(du-dudx_j)/M < 1e-10
    
    # Multidimensional version
    u3_hat = u_hat.repeat(4*4).reshape((M, 4, 4)) + 1j*u_hat.repeat(4*4).reshape((M, 4, 4))    
    cs = Cm.matvec(u3_hat)
    cs2 = np.zeros((M, 4, 4), dtype=np.complex)
    du3 = dudx_j.repeat(4*4).reshape((M, 4, 4)) + 1j*dudx_j.repeat(4*4).reshape((M, 4, 4))
    cs2 = SD.fastShenScalar(du3, cs2)
    
    assert np.allclose(cs, cs2, 1e-10)
    
    cs = TDMASolver(cs)
    d3 = np.zeros((M, 4, 4), dtype=np.complex)
    d3 = SD.ifst(cs, d3)

    #from IPython import embed; embed()
    assert np.linalg.norm(du3-d3)/(M*16) < 1e-10
        
def test_CXXmat(SXSX):
    S1, S2 = SXSX
    
    if S1.__class__.__name__ == "ShenDirichletBasis" and S2.__class__.__name__ == "ShenDirichletBasis":
        Cm = CDDmat(np.arange(N).astype(np.float))    
    elif S1.__class__.__name__ == "ShenDirichletBasis":
        Cm = CDNmat(np.arange(N).astype(np.float))    
    elif S1.__class__.__name__ == "ShenNeumannBasis":
        Cm = CNDmat(np.arange(N).astype(np.float))    
    
    fj = np.random.randn(N)
    # project to S2
    f_hat = np.zeros(N)
    f_hat = S2.fst(fj, f_hat)
    fj = S2.ifst(f_hat, fj)
    
    # Check S1.fss(f) equals Cm*S2.fst(f)
    f_hat = S2.fst(fj, f_hat)
    cs = Cm.matvec(f_hat)
    df = np.zeros(N)
    df = S2.fastChebDerivative(fj, df)
    cs2 = np.zeros(N)
    cs2 = S1.fastShenScalar(df, cs2)
    
    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)
    
    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((N, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((N, 4, 4))    
    df = df.repeat(4*4).reshape((N, 4, 4)) + 1j*df.repeat(4*4).reshape((N, 4, 4))    
    cs = Cm.matvec(f_hat)
    cs2 = np.zeros((N, 4, 4), dtype=np.complex)
    cs2 = S1.fastShenScalar(df, cs2)
    
    assert np.allclose(cs, cs2)
    

def test_CDTmat(SDST):
    SD, ST = SDST
    
    Cm = CDTmat(np.arange(N).astype(np.float))    
    
    fj = np.random.randn(N)
    # project to ST
    f_hat = np.zeros(N)
    f_hat = ST.fct(fj, f_hat)
    fj = ST.ifct(f_hat, fj)
    
    # Check SD.fss(f) equals Cm*ST.fst(f)
    f_hat = ST.fct(fj, f_hat)
    cs = Cm.matvec(f_hat)
    df = np.zeros(N)
    df = ST.fastChebDerivative(fj, df)
    cs2 = np.zeros(N)
    cs2 = SD.fastShenScalar(df, cs2)
    
    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)
    
    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((N, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((N, 4, 4))    
    df = df.repeat(4*4).reshape((N, 4, 4)) + 1j*df.repeat(4*4).reshape((N, 4, 4))    
    cs = Cm.matvec(f_hat)
    cs2 = np.zeros((N, 4, 4), dtype=np.complex)
    cs2 = SD.fastShenScalar(df, cs2)
    
    assert np.allclose(cs, cs2)


def test_CTDmat(SDST):
    SD, ST = SDST
    
    Cm = CTDmat(np.arange(N).astype(np.float))    
    
    fj = np.random.randn(N)
    # project to SD
    f_hat = np.zeros(N)
    f_hat = SD.fst(fj, f_hat)
    fj = SD.ifst(f_hat, fj)
    
    # Check if ST.fcs(f') equals Cm*SD.fst(f)
    f_hat = SD.fst(fj, f_hat)
    cs = Cm.matvec(f_hat)
    df = np.zeros(N)
    df = SD.fastChebDerivative(fj, df)
    cs2 = np.zeros(N)
    cs2 = ST.fastChebScalar(df, cs2)
    
    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)
    
    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((N, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((N, 4, 4))    
    df = df.repeat(4*4).reshape((N, 4, 4)) + 1j*df.repeat(4*4).reshape((N, 4, 4))    
    cs = Cm.matvec(f_hat)
    cs2 = np.zeros((N, 4, 4), dtype=np.complex)
    cs2 = ST.fastChebScalar(df, cs2)
    
    assert np.allclose(cs, cs2)

def test_CDBmat(SBST):
    SB, SD = SBST
    
    M = 8*N
    Cm = CDBmat(np.arange(M).astype(np.float))    
    
    x = Symbol("x")
    u = sin(2*pi*x)**2
    f = u.diff(x, 1)

    points, weights = SB.points_and_weights(M) 

    uj = np.array([u.subs(x, j) for j in points], dtype=float)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points
    
    # project to SB
    f_hat = np.zeros(M)
    f_hat = SB.fst(uj, f_hat)
    uj = SB.ifst(f_hat, uj)
    
    # Check if SD.fss(f') equals Cm*SD.fst(f)
    f_hat = SB.fst(uj, f_hat)
    cs = Cm.matvec(f_hat)
    
    df = np.zeros(M)
    df = SB.fastChebDerivative(uj, df)
    cs2 = np.zeros(M)
    cs2 = SD.fastShenScalar(df, cs2)
    
    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)
    
    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((M, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((M, 4, 4))    
    df = df.repeat(4*4).reshape((M, 4, 4)) + 1j*df.repeat(4*4).reshape((M, 4, 4))    
    cs = Cm.matvec(f_hat)
    cs2 = np.zeros((M, 4, 4), dtype=np.complex)
    cs2 = SD.fastShenScalar(df, cs2)
    
    assert np.allclose(cs, cs2)

#test_CDBmat((ShenBiharmonicBasis("GC"), ShenDirichletBasis("GC")))

def test_CBDmat(SBST):
    SB, SD = SBST
    
    M = 4*N
    Cm = CBDmat(np.arange(M).astype(np.float))    
    
    x = Symbol("x")
    u = sin(12*pi*x)**2
    f = u.diff(x, 1)

    points, weights = SD.points_and_weights(M) 

    uj = np.array([u.subs(x, j) for j in points], dtype=float)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points
    
    # project to SD
    f_hat = np.zeros(M)
    f_hat = SD.fst(uj, f_hat)
    uj = SD.ifst(f_hat, uj)
    
    # Check if SB.fss(f') equals Cm*SD.fst(f)
    f_hat = SD.fst(uj, f_hat)
    cs = Cm.matvec(f_hat)
    
    df = np.zeros(M)
    df = SD.fastChebDerivative(uj, df)
    cs2 = np.zeros(M)
    cs2 = SB.fastShenScalar(df, cs2)
    
    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)
    
    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((M, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((M, 4, 4))    
    df = df.repeat(4*4).reshape((M, 4, 4)) + 1j*df.repeat(4*4).reshape((M, 4, 4))    
    cs = Cm.matvec(f_hat)
    cs2 = np.zeros((M, 4, 4), dtype=np.complex)
    cs2 = SB.fastShenScalar(df, cs2)
    
    assert np.allclose(cs, cs2)

#test_CBDmat((ShenBiharmonicBasis("GL"), ShenDirichletBasis("GL")))

def test_Mult_Div():
    
    SD = ShenDirichletBasis("GC")
    SN = ShenDirichletBasis("GC")
    
    Cm = CNDmat(np.arange(N).astype(np.float))
    Bm = BNDmat(np.arange(N).astype(np.float), "GC")
    
    uk = np.random.randn((N))+np.random.randn((N))*1j
    vk = np.random.randn((N))+np.random.randn((N))*1j
    wk = np.random.randn((N))+np.random.randn((N))*1j
    
    b = np.zeros(N, dtype=np.complex)
    uk0 = np.zeros(N, dtype=np.complex)
    vk0 = np.zeros(N, dtype=np.complex)
    wk0 = np.zeros(N, dtype=np.complex)
    
    uk0 = SD.fst(uk, uk0)
    uk  = SD.ifst(uk0, uk)
    uk0 = SD.fst(uk, uk0)
    vk0 = SD.fst(vk, vk0)
    vk  = SD.ifst(vk0, vk)
    vk0 = SD.fst(vk, vk0)
    wk0 = SD.fst(wk, wk0)
    wk  = SD.ifst(wk0, wk)
    wk0 = SD.fst(wk, wk0)

    SFTc.Mult_Div_1D(N, 7, 7, uk0[:N-2], vk0[:N-2], wk0[:N-2], b[1:N-2])
        
    uu = Cm.matvec(uk0)
    uu += 1j*7*Bm.matvec(vk0) + 1j*7*Bm.matvec(wk0)
    
    #from IPython import embed; embed()
    assert np.allclose(uu, b)
    
    uk0 = uk0.repeat(4*4).reshape((N,4,4)) + 1j*uk0.repeat(4*4).reshape((N,4,4))
    vk0 = vk0.repeat(4*4).reshape((N,4,4)) + 1j*vk0.repeat(4*4).reshape((N,4,4))
    wk0 = wk0.repeat(4*4).reshape((N,4,4)) + 1j*wk0.repeat(4*4).reshape((N,4,4))
    b = np.zeros((N,4,4), dtype=np.complex)
    m = np.zeros((4,4))+7
    n = np.zeros((4,4))+7 
    SFTc.Mult_Div_3D(N, m, n, uk0[:N-2], vk0[:N-2], wk0[:N-2], b[1:N-2])
    
    uu = Cm.matvec(uk0)
    uu += 1j*7*Bm.matvec(vk0) + 1j*7*Bm.matvec(wk0)
    
    assert np.allclose(uu, b)

#test_Mult_Div()

def test_ADDmat(ST2):
    M = 2*N
    u = (1-x**2)*sin(np.pi*x)
    f = -u.diff(x, 2)
    
    points, weights = ST2.points_and_weights(M)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    fj = np.array([f.subs(x, h) for h in points], dtype=np.float)
    
    if ST2.__class__.__name__ == "ShenDirichletBasis":
        A = ADDmat(np.arange(M).astype(np.float))
        s = slice(0, M-2)
    elif ST2.__class__.__name__ == "ShenNeumannBasis":
        A = ANNmat(np.arange(M).astype(np.float))
        s = slice(1, M-2)
        fj -= np.dot(fj, weights)/weights.sum()
        uj -= np.dot(uj, weights)/weights.sum()
        
    f_hat = np.zeros(M)
    f_hat = ST2.fastShenScalar(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[s] = la.spsolve(A.diags(), f_hat[s])
    
    u0 = np.zeros(M)
    u0 = ST2.ifst(u_hat, u0)
        
    #from IPython import embed; embed()
    assert np.allclose(u0, uj)
    
    u1 = np.zeros(M)
    u1 = ST2.fst(uj, u1)
    c = A.matvec(u1)
    
    assert np.allclose(c, f_hat)
    
#test_ADDmat(ShenDirichletBasis("GL"))


def test_SBBmat(SB):
    M = 6*N
    u = sin(4*pi*x)**2
    f = u.diff(x, 4)
    
    points, weights = SB.points_and_weights(M)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    fj = np.array([f.subs(x, h) for h in points], dtype=np.float)
    
    A = SBBmat(np.arange(M).astype(np.float))
    f_hat = np.zeros(M)
    f_hat = SB.fastShenScalar(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[:-4] = la.spsolve(A.diags(), f_hat[:-4])
    
    u0 = np.zeros(M)
    u0 = SB.ifst(u_hat, u0)
        
    assert np.allclose(u0, uj)
    
    u1 = np.zeros(M)
    u1 = SB.fst(uj, u1)
    
    c = A.matvec(u1)
    
    #from IPython import embed; embed()
    assert np.allclose(c, f_hat, 1e-6, 1e-6)
    
    # Multidimensional
    c2 = (c.repeat(16).reshape((M, 4, 4))+1j*c.repeat(16).reshape((M, 4, 4)))
    u1 = (u1.repeat(16).reshape((M, 4, 4))+1j*u1.repeat(16).reshape((M, 4, 4)))
    
    c = A.matvec(u1)
    
    assert np.allclose(c, c2)

#test_SBBmat(ShenBiharmonicBasis("GC"))

def test_ABBmat(SB):
    M = 6*N
    u = sin(6*pi*x)**2
    f = u.diff(x, 2)
    
    points, weights = SB.points_and_weights(M)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    fj = np.array([f.subs(x, h) for h in points], dtype=np.float)
    
    A = ABBmat(np.arange(M).astype(np.float))
    
    f_hat = np.zeros(M)
    f_hat = SB.fastShenScalar(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[:-4] = la.spsolve(A.diags(), f_hat[:-4])
    
    u0 = np.zeros(M)
    u0 = SB.ifst(u_hat, u0)
        
    assert np.allclose(u0, uj)
    
    u1 = np.zeros(M)
    u1 = SB.fst(uj, u1)
    c = A.matvec(u1)
    
    assert np.allclose(c, f_hat, 1e-6, 1e-6)
    
    # Multidimensional
    f_hat = (f_hat.repeat(16).reshape((M, 4, 4))+1j*f_hat.repeat(16).reshape((M, 4, 4)))
    u1 = (u1.repeat(16).reshape((M, 4, 4))+1j*u1.repeat(16).reshape((M, 4, 4)))
    
    c = A.matvec(u1)
    
    assert np.allclose(c, f_hat, 1e-6, 1e-6)
    
    B = BBBmat(np.arange(M).astype(np.float), SB.quad)
    u0 = np.random.randn(M)
    u0_hat = np.zeros(M)
    u0_hat = SB.fst(u0, u0_hat)
    u0 = SB.ifst(u0_hat, u0)
    b = np.zeros(M)
    k = 2.
    b = A.matvec(u0_hat) - k**2*B.matvec(u0_hat)
    AA = A.diags().toarray() - k**2*B.diags().toarray()
    z0_hat = np.zeros(M)
    z0_hat[:-4] = solve(AA, b[:-4])
    z0 = np.zeros(M)
    z0 = SB.ifst(z0_hat, z0)
    assert np.allclose(z0, u0)
    
    
    k = np.ones(M)*2
    k = k.repeat(16).reshape((M, 4, 4))
    k2 = k**2
    u0_hat = u0_hat.repeat(16).reshape((M, 4, 4)) + 1j*u0_hat.repeat(16).reshape((M, 4, 4))
    u0 = u0.repeat(16).reshape((M, 4, 4))
    b = A.matvec(u0_hat) - k**2*B.matvec(u0_hat)
    alfa = np.ones((M, 4, 4))
    
    BH = Biharmonic(M, 0, alfa[0], -k2[0], SB.quad, "cython")
    z0_hat = np.zeros((M, 4, 4), dtype=np.complex)
    z0_hat = BH(z0_hat, b)    
    z0 = np.zeros((M, 4, 4))
    z0 = SB.ifst(z0_hat.real, z0)
    #from IPython import embed; embed()
    assert np.allclose(z0, u0)
    

#test_ABBmat(ShenBiharmonicBasis("GC"))

#@profile
def test_Helmholtz(ST2):
    M = 4*N
    kx = 12
    
    points, weights = ST2.points_and_weights(M)
    
    fj = np.random.randn(M)
    f_hat = np.zeros(M)
    if not ST2.__class__.__name__ == "ChebyshevTransform":
        f_hat = ST2.fst(fj, f_hat)
        fj = ST2.ifst(f_hat, fj)
    
    if ST2.__class__.__name__ == "ShenDirichletBasis":
        A = ADDmat(np.arange(M).astype(np.float))
        B = BDDmat(np.arange(M).astype(np.float), ST2.quad)
        s = slice(0, M-2)
    elif ST2.__class__.__name__ == "ShenNeumannBasis":
        A = ANNmat(np.arange(M).astype(np.float))
        B = BNNmat(np.arange(M).astype(np.float), ST2.quad)
        s = slice(1, M-2)
        
    f_hat = np.zeros(M)
    f_hat = ST2.fastShenScalar(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[s] = la.spsolve(A.diags()+kx**2*B.diags(), f_hat[s])    
    u1 = np.zeros(M)
    u1 = ST2.ifst(u_hat, u1)
    c = A.matvec(u_hat)+kx**2*B.matvec(u_hat)        
    c2 = np.dot(A.diags().toarray(), u_hat[s]) + kx**2*np.dot(B.diags().toarray(), u_hat[s])
    
    assert np.allclose(c, f_hat)
    assert np.allclose(c[s], c2)
    
    H = Helmholtz(M, kx, ST2.quad, ST2.__class__.__name__ == "ShenNeumannBasis")
    u0_hat = np.zeros(M)
    u0_hat = H(u0_hat, f_hat)
    u0 = np.zeros(M)
    u0 = ST2.ifst(u0_hat, u0)
    
    assert np.linalg.norm(u0 - u1) < 1e-12

    
    # Multidimensional
    f_hat = (f_hat.repeat(16).reshape((M, 4, 4))+1j*f_hat.repeat(16).reshape((M, 4, 4)))
    kx = np.zeros((4, 4))+12
    H = Helmholtz(M, kx, ST2.quad, ST2.__class__.__name__ == "ShenNeumannBasis")
    u0_hat = np.zeros((M, 4, 4), dtype=np.complex)
    u0_hat = H(u0_hat, f_hat)
    u0 = np.zeros((M, 4, 4), dtype=np.complex)
    u0 = ST2.ifst(u0_hat, u0)
    #from IPython import embed; embed()
    
    assert np.linalg.norm(u0[:, 2, 2].real - u1)/(M*16) < 1e-12
    assert np.linalg.norm(u0[:, 2, 2].imag - u1)/(M*16) < 1e-12

#test_Helmholtz(ShenNeumannBasis("GC"))

def test_Helmholtz2(SD):
    M = 2*N
    kx = 11
    points, weights = SD.points_and_weights(M)
    uj = np.random.randn(M)
    u_hat = np.zeros(M)
    u_hat = SD.fst(uj, u_hat)
    uj = SD.ifst(u_hat, uj)
    
    A = ADDmat(np.arange(M).astype(np.float))
    B = BDDmat(np.arange(M).astype(np.float), SD.quad)
    s = slice(0, M-2)

    u1 = np.zeros(M)
    u1 = SD.fst(uj, u1)
    c = A.matvec(u1)+kx**2*B.matvec(u1)

    b = np.zeros(M)
    SFTc.Mult_Helmholtz_1D(M, SD.quad=="GL", 1, kx**2, u1, b)
    assert np.allclose(c, b)
    
    b = np.zeros((M, 4, 4), dtype=np.complex)
    u1 = u1.repeat(16).reshape((M, 4, 4)) +1j*u1.repeat(16).reshape((M, 4, 4))
    kx = np.zeros((4, 4))+kx
    SFTc.Mult_Helmholtz_3D_complex(M, SD.quad=="GL", 1.0, kx**2, u1, b)
    assert np.linalg.norm(b[:, 2, 2].real - c)/(M*16) < 1e-12
    assert np.linalg.norm(b[:, 2, 2].imag - c)/(M*16) < 1e-12
    
def test_Mult_CTD(SD):
    C = CTDmat(np.arange(N).astype(np.float))
    B = BTTmat(np.arange(N).astype(np.float), SD.quad)
    
    uk = np.random.randn((N))+np.random.randn((N))*1j
    vk = np.random.randn((N))+np.random.randn((N))*1j
    wk = np.random.randn((N))+np.random.randn((N))*1j
    
    bv = np.zeros(N, dtype=np.complex)
    bw = np.zeros(N, dtype=np.complex)
    vk0 = np.zeros(N, dtype=np.complex)
    wk0 = np.zeros(N, dtype=np.complex)
    cv = np.zeros(N, dtype=np.complex)
    cw = np.zeros(N, dtype=np.complex)
    
    vk0 = SD.fst(vk, vk0)
    vk  = SD.ifst(vk0, vk)
    vk0 = SD.fst(vk, vk0)
    wk0 = SD.fst(wk, wk0)
    wk  = SD.ifst(wk0, wk)
    wk0 = SD.fst(wk, wk0)

    SFTc.Mult_CTD_1D(N, vk0, wk0, bv, bw)
    
    cv[:] = C.matvec(vk0)
    cw[:] = C.matvec(wk0)
    cv /= B.dd
    cw /= B.dd
    
    #from IPython import embed; embed()
    assert np.allclose(cv, bv)
    assert np.allclose(cw, bw)

#test_Mult_CTD(ShenDirichletBasis("GL"))

def test_Biharmonic(SB):
    M = 128
    x = Symbol("x")
    u = sin(6*pi*x)**2
    a = 1.0
    b = 1.0
    f = -u.diff(x, 4) + a*u.diff(x, 2) + b*u

    points, weights = SB.points_and_weights(M) 

    uj = np.array([u.subs(x, j) for j in points], dtype=float)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

    k = np.arange(M).astype(np.float)
    A = SBBmat(k)
    B = BBBmat(k, SB.quad)
    C = ABBmat(k)

    AA = -A.diags() + C.diags() + B.diags()
    f_hat = np.zeros(M)
    f_hat = SB.fastShenScalar(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[:-4] = la.spsolve(AA, f_hat[:-4])
    u1 = np.zeros(M)
    u1 = SB.ifst(u_hat, u1)
    #from IPython import embed; embed()

    assert np.allclose(u1, uj)

#test_Biharmonic(ShenBiharmonicBasis("GC"))

def test_Helmholtz_matvec(SD):
    M = 2*N
    kx = 11
    points, weights = SD.points_and_weights(M)
    uj = np.random.randn(M)
    u_hat = np.zeros(M)
    u_hat = SD.fst(uj, u_hat)
    uj = SD.ifst(u_hat, uj)
    
    A = ADDmat(np.arange(M).astype(np.float))
    B = BDDmat(np.arange(M).astype(np.float), SD.quad)
    AB = HelmholtzCoeff(np.arange(M).astype(np.float), 1, kx**2, SD.quad)
    s = slice(0, M-2)

    u1 = np.zeros(M)
    u1 = SD.fst(uj, u1)
    c = A.matvec(u1)+kx**2*B.matvec(u1)

    b = np.zeros(M)
    #SFTc.Mult_Helmholtz_1D(M, SD.quad=="GL", 1, kx**2, u1, b)
    b = AB.matvec(u1, b)
    assert np.allclose(c, b)
    
    b = np.zeros((M, 4, 4), dtype=np.complex)
    u1 = u1.repeat(16).reshape((M, 4, 4)) +1j*u1.repeat(16).reshape((M, 4, 4))
    kx = np.zeros((4, 4))+kx
    #SFTc.Mult_Helmholtz_3D_complex(M, SD.quad=="GL", 1.0, kx**2, u1, b)
    AB = HelmholtzCoeff(np.arange(M).astype(np.float), 1, kx**2, SD.quad)
    b = AB.matvec(u1, b)
    
    assert np.linalg.norm(b[:, 2, 2].real - c)/(M*16) < 1e-12
    assert np.linalg.norm(b[:, 2, 2].imag - c)/(M*16) < 1e-12

#test_Helmholtz_matvec(ShenDirichletBasis("GL"))  
  
#test_ADDmat(ShenNeumannBasis("GL")) 
#test_Helmholtz2(ShenDirichletBasis("GL")) 
#test_Mult_CTD(ShenDirichletBasis("GL"))
#test_CDDmat(ShenDirichletBasis("GL"))
