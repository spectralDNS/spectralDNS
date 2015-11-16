import pytest
from cbcdns.shen.shentransform import ShenDirichletBasis, ShenNeumannBasis, ChebyshevTransform
from cbcdns.shen.Helmholtz import TDMA, Helmholtz
from cbcdns.shen.Matrices import BNNmat, BCCmat, BDDmat, CDDmat, CDNmat, BNDmat, CNDmat, BDNmat, ADDmat, ANNmat, CTDmat
from cbcdns.shen import SFTc

from sympy import chebyshevt, Symbol, sin, cos, pi
import numpy as np
import scipy.sparse.linalg as la

N = 12
x = Symbol("x")

@pytest.fixture(params=("NGC", "NGL", "DGC", "DGL", "CGC", "CGL"))
def ST(request):
    if request.param[0] == 'N':
        return ShenNeumannBasis(request.param[1:])
    elif request.param[0] == 'D':
        return ShenDirichletBasis(request.param[1:])
    elif request.param[0] == 'C':
        return ChebyshevTransform(request.param[1:])

@pytest.fixture(params=("NGC", "NGL", "DGC", "DGL"))
def ST2(request):
    if request.param[0] == 'N':
        return ShenNeumannBasis(request.param[1:])
    elif request.param[0] == 'D':
        return ShenDirichletBasis(request.param[1:])

@pytest.fixture(params=("GC", "GL"))
def SD(request):
    return ShenDirichletBasis(request.param)

@pytest.fixture(params=("GCGC", "GCGL", "GLGC", "GLGL"))
def SDSN(request):
    return (ShenDirichletBasis(request.param[:2]), ShenNeumannBasis(request.param[2:]))

@pytest.fixture(params=("NGC", "NGL", "DGC", "DGL"))
def T(request):
    if request.param[0] == 'N':
        return TDMA(request.param[1:], True)
    elif request.param[0] == 'D':
        return TDMA(request.param[1:], False)

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
    
    # Multidimensional version
    f = f.repeat(16).reshape((N, 4, 4))
    f = T(f)
    assert np.allclose(f[s, 2, 2], u)
    
def test_BNNmat(ST):
    points, weights = ST.points_and_weights(N)
    f = (1-x**4)*sin(pi*x)  # A function with f(+-1) = 0 and f'(+-1) = 0
    fj = np.array([f.subs(x, j) for j in points], dtype=float)
    u0 = np.zeros(N)
    f_hat = np.zeros(N)
    if ST.__class__.__name__ == "ShenNeumannBasis":
        B = BNNmat(np.arange(N).astype(np.float), ST.quad)
        u0 = ST.fastShenScalar(fj, u0)
        f_hat = ST.fst(fj, f_hat)
    elif ST.__class__.__name__ == "ShenDirichletBasis":
        B = BDDmat(np.arange(N).astype(np.float), ST.quad)
        u0 = ST.fastShenScalar(fj, u0)
        f_hat = ST.fst(fj, f_hat)
    else:
        B = BCCmat(np.arange(N).astype(np.float), ST.quad)
        u0 = ST.fastChebScalar(fj, u0)
        f_hat = ST.fct(fj, f_hat)

    u2 = B.matvec(f_hat)
    assert np.allclose(u2, u0)
    
    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4))
    u0 = np.zeros((N, 4, 4))
    f_hat = np.zeros((N, 4, 4))
    if ST.__class__.__name__ in ("ShenNeumannBasis", "ShenDirichletBasis"):
        u0 = ST.fastShenScalar(fj, u0)
        f_hat = ST.fst(fj, f_hat)
    else:
        u0 = ST.fastChebScalar(fj, u0)
        f_hat = ST.fct(fj, f_hat)
    u2 = B.matvec(f_hat)
    assert np.allclose(u2, u0)

def test_BDNmat(SDSN):
    SD, SN = SDSN
    
    # Not identical tests, so use finer resolution
    M = 2*N
    pointsD, weightsD = SD.points_and_weights(M)
    pointsN, weightsN = SN.points_and_weights(M)
    
    f = (1-x**6)*sin(pi*x)  # A function with f(+-1) = 0 and f'(+-1) = 0    
    fD = np.array([f.subs(x, j) for j in pointsD], dtype=float)
    fN = np.array([f.subs(x, j) for j in pointsN], dtype=float)
    #fN -= np.dot(fN, weightsN)/weightsN.sum()
    u0 = np.zeros(M)
    f_hat = np.zeros(M)
    B = BDNmat(np.arange(M).astype(np.float), SN.quad)
    
    u0 = SN.fastShenScalar(fN, u0)

    f_hat = SD.fst(fD, f_hat)
    u2 = B.matvec(f_hat)
    
    assert np.allclose(u2, u0)
    
    # Multidimensional version
    fN = fN.repeat(16).reshape((M, 4, 4))
    f_hat = f_hat.repeat(16).reshape((M, 4, 4))
    
    u0 = np.zeros((M, 4, 4))
    u0 = SN.fastShenScalar(fN, u0)    
    u2 = B.matvec(f_hat)
    assert np.linalg.norm(u2-u0) < 1e-8


def test_BNDmat(SDSN):
    SD, SN = SDSN
    
    # Not identical tests, so use finer resolution
    M = 2*N
    pointsD, weightsD = SD.points_and_weights(M)
    pointsN, weightsN = SN.points_and_weights(M)
    
    f = (1-x**12)*sin(pi*x)  # A function with f(+-1) = 0 and f'(+-1) = 0    
    fD = np.array([f.subs(x, j) for j in pointsD], dtype=float)
    fN = np.array([f.subs(x, j) for j in pointsN], dtype=float)
    fN -= np.dot(fN, weightsN)/weightsN.sum()
    
    u0 = np.zeros(M)
    f_hat = np.zeros(M)
    B = BNDmat(np.arange(M).astype(np.float), SD.quad)
    
    u0 = SD.fastShenScalar(fD, u0)

    f_hat = SN.fst(fN, f_hat)
    u2 = B.matvec(f_hat)
    
    assert np.allclose(u2, u0)
    
    # Multidimensional version
    fD = fD.repeat(16).reshape((M, 4, 4))
    f_hat = f_hat.repeat(16).reshape((M, 4, 4))
    
    u0 = np.zeros((M, 4, 4))
    u0 = SD.fastShenScalar(fD, u0)    
    u2 = B.matvec(f_hat)
    assert np.linalg.norm(u2-u0) < 1e-8


def test_transforms(ST):
    points, weights = ST.points_and_weights(N)
    f = cos(pi*x)*(1-x**4)  # A function with f(+-1) = 0 and f'(+-1) = 0
    fj = np.array([f.subs(x, j) for j in points], dtype=float)
    
    # Project function to space first
    if ST.__class__.__name__ in ("ShenNeumannBasis", "ShenDirichletBasis"):
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
    assert np.allclose(fj, u1)
    
    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4))
    u0 = np.zeros((N, 4, 4))    
    u1 = np.zeros((N, 4, 4))
    if ST.__class__.__name__ == "ChebyshevTransform":
        u0 = ST.fct(fj, u0)
        u1 = ST.ifct(u0, u1)
    else:
        u0 = ST.fst(fj, u0)
        u1 = ST.ifst(u0, u1)
    assert np.allclose(fj, u1)
    
    
def test_CDDmat(SD):
    M = 512
    #u = (1-x**2)*sin(np.pi*6*x)
    #dudx = u.diff(x, 1)
    points, weights = SD.points_and_weights(M)
    #dudx_j = np.array([dudx.subs(x, h) for h in points], dtype=np.float)
    #uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    
    from OrrSommerfeld_eig import OrrSommerfeld
    OS = OrrSommerfeld(Re=8000., N=80)
    uj = np.zeros(M)
    for i, y in enumerate(points):
        OS.interp(y)
        uj[i] = -np.dot(OS.f, np.real(1j*OS.phi*np.exp(1j*(5.89048-OS.eigval*0.01))))
        
    dudx_j = np.zeros(M)
    u_hat = np.zeros(M)
    u_hat = SD.fst(uj, u_hat)
    uj = SD.ifst(u_hat, uj)
    u_hat = SD.fst(uj, u_hat)
    
    uc_hat = np.zeros(M)
    uc_hat = SD.fct(uj, uc_hat)
    du_hat = np.zeros(M)
    dudx_j = SD.fastChebDerivative(uj, dudx_j, uc_hat, du_hat)
    
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

    assert np.linalg.norm(du-dudx_j) < 1e-6
    
    # Multidimensional version
    u3_hat = u_hat.repeat(4*4).reshape((M, 4, 4))    
    cs = Cm.matvec(u3_hat)
    cs2 = np.zeros((M, 4, 4))
    du3 = dudx_j.repeat(4*4).reshape((M, 4, 4))
    cs2 = SD.fastShenScalar(du3, cs2)
    
    assert np.allclose(cs, cs2, 1e-6)
    
    cs = TDMASolver(cs)
    d3 = np.zeros((M, 4, 4))
    d3 = SD.ifst(cs, d3)

    assert np.linalg.norm(du3-d3) < 1e-6
        
def test_CDNmat(SDSN):
    SD, SN = SDSN
    M = 2*N
    u = (1-x**2)*sin(np.pi*x)
    dudx = u.diff(x, 1)
    pointsD, weightsD = SD.points_and_weights(M)
    pointsN, weightsN = SN.points_and_weights(M)
    
    uj = np.array([u.subs(x, h) for h in pointsN], dtype=np.float)
    uj -= np.dot(uj, weightsN)/weightsN.sum()
    # project to Neumann space
    u_hat = np.zeros(M)
    u_hat = SN.fst(uj, u_hat)
    
    Cm = CDNmat(np.arange(M).astype(np.float))    
    cs = Cm.matvec(u_hat)
    
    # Should equal (but not exact so use extra resolution)
    cs2 = np.zeros(M)
    dudx_j = np.array([dudx.subs(x, h) for h in pointsD], dtype=np.float)    
    cs2 = SD.fastShenScalar(dudx_j, cs2)
    assert np.allclose(cs2, cs)
        
    TDMASolver = TDMA(SD.quad)
    b = cs.copy()
    b = TDMASolver(b)
    du = np.zeros(M)
    du = SD.ifst(b, du)

    assert np.linalg.norm(du-dudx_j) < 1e-6

    # Multidimensional version
    u3_hat = u_hat.repeat(4*4).reshape((M, 4, 4))    
    cs = Cm.matvec(u3_hat)
    cs2 = np.zeros((M, 4, 4))
    du3 = dudx_j.repeat(4*4).reshape((M, 4, 4))
    cs2 = SD.fastShenScalar(du3, cs2)
    
    assert np.allclose(cs, cs2, 1e-6)
    
    cs = TDMASolver(cs)
    d3 = np.zeros((M, 4, 4))
    d3 = SD.ifst(cs, d3)

    assert np.linalg.norm(du3-d3) < 1e-6

def test_CNDmat(SDSN):
    SD, SN = SDSN
    M = 2*N
    u = (1-x**2)*sin(np.pi*x)
    dudx = u.diff(x, 1)
    pointsD, weightsD = SD.points_and_weights(M)
    pointsN, weightsN = SN.points_and_weights(M)
    
    uj = np.array([u.subs(x, h) for h in pointsD], dtype=np.float)
    # project to Dirichlet space
    u_hat = np.zeros(M)
    u_hat = SD.fst(uj, u_hat)
    
    Cm = CNDmat(np.arange(M).astype(np.float))    
    cs = Cm.matvec(u_hat)
    
    # Should equal (but not exact so use extra resolution)
    cs2 = np.zeros(M)
    dudx_j = np.array([dudx.subs(x, h) for h in pointsN], dtype=np.float)    
    # project to Neumann space
    du_hat = np.zeros(M)
    du_hat = SN.fst(dudx_j, du_hat)    
    dudx_j = SN.ifst(du_hat, dudx_j)
    
    cs2 = SN.fastShenScalar(dudx_j, cs2)
    assert np.allclose(cs2, cs)
        
    TDMASolver = TDMA(SN.quad, True)
    b = cs.copy()
    b = TDMASolver(b)
    du = np.zeros(M)
    du = SN.ifst(b, du)

    assert np.linalg.norm(du-dudx_j) < 1e-6


def test_Mult_Div():
    
    SD = ShenDirichletBasis("GL")
    SN = ShenDirichletBasis("GC")
    
    Cm = CNDmat(np.arange(N).astype(np.float))
    Bm = BDNmat(np.arange(N).astype(np.float), "GC")
    
    uk = np.random.randn((N))+np.random.randn((N))*1j
    vk = np.random.randn((N))+np.random.randn((N))*1j
    wk = np.random.randn((N))+np.random.randn((N))*1j
    
    b = np.zeros(N, dtype=np.complex)
    uk0 = np.zeros(N, dtype=np.complex)
    vk0 = np.zeros(N, dtype=np.complex)
    wk0 = np.zeros(N, dtype=np.complex)
    
    uk0 = SD.fst(uk, uk0)
    uk  = SD.ifst(uk0, uk)
    vk0 = SD.fst(vk, vk0)
    vk  = SD.ifst(vk0, vk)
    wk0 = SD.fst(wk, wk0)
    wk  = SD.ifst(wk0, wk)

    SFTc.Mult_Div_1D(N, 2, 3, uk[:N-2], vk[:N-2], wk[:N-2], b[1:N-2])
        
    uu = Cm.matvec(uk)
    uu += 1j*2*Bm.matvec(vk) + 1j*3*Bm.matvec(wk)
    
    assert np.allclose(uu, b)

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
    
def test_Helmholtz(ST2):
    M = 2*N
    u = (1-x**2)*sin(np.pi*x)
    kx = 12
    f = -u.diff(x, 2)+kx**2*u
    
    points, weights = ST2.points_and_weights(M)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    fj = np.array([f.subs(x, h) for h in points], dtype=np.float)
    
    if ST2.__class__.__name__ == "ShenDirichletBasis":
        A = ADDmat(np.arange(M).astype(np.float))
        B = BDDmat(np.arange(M).astype(np.float), ST2.quad)
        s = slice(0, M-2)
    elif ST2.__class__.__name__ == "ShenNeumannBasis":
        A = ANNmat(np.arange(M).astype(np.float))
        B = BNNmat(np.arange(M).astype(np.float), ST2.quad)
        s = slice(1, M-2)
        fj -= np.dot(fj, weights)/weights.sum()
        uj -= np.dot(uj, weights)/weights.sum()
        
    f_hat = np.zeros(M)
    f_hat = ST2.fastShenScalar(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[s] = la.spsolve(A.diags()+kx**2*B.diags(), f_hat[s])
    
    u0 = np.zeros(M)
    u0 = ST2.ifst(u_hat, u0)
        
    assert np.allclose(u0, uj)
    
    u1 = np.zeros(M)
    u1 = ST2.fst(uj, u1)
    c = A.matvec(u1)+kx**2*B.matvec(u1)
    
    assert np.allclose(c, f_hat)
            
    # Multidimensional
    f_hat = (f_hat.repeat(16).reshape((M, 4, 4))+1j*f_hat.repeat(16).reshape((M, 4, 4)))
    kx = np.zeros((4, 4))+12
    H = Helmholtz(M, kx, ST2.quad, ST2.__class__.__name__ == "ShenNeumannBasis")
    u0_hat = np.zeros((M, 4, 4), dtype=np.complex)
    u0_hat = H(u0_hat, f_hat)
    u0 = np.zeros((M, 4, 4), dtype=np.complex)
    u0 = ST2.ifst(u0_hat, u0)
    
    assert np.linalg.norm(u0[:, 2, 2].real - uj) < 1e-12
    assert np.linalg.norm(u0[:, 2, 2].imag - uj) < 1e-12


def test_Helmholtz2(SD):
    M = 2*N
    u = (1-x**2)*sin(np.pi*x)
    kx = 12
    f = -u.diff(x, 2)+kx**2*u
    
    points, weights = SD.points_and_weights(M)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    fj = np.array([f.subs(x, h) for h in points], dtype=np.float)
    
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
    assert np.linalg.norm(b[:, 2, 2].real - c) < 1e-8
    assert np.linalg.norm(b[:, 2, 2].imag - c) < 1e-8
    
def test_Mult_CTD(SD):
    C = CTDmat(np.arange(N).astype(np.float))
    B = BCCmat(np.arange(N).astype(np.float), SD.quad)
    
    uk = np.random.randn((N))+np.random.randn((N))*1j
    vk = np.random.randn((N))+np.random.randn((N))*1j
    wk = np.random.randn((N))+np.random.randn((N))*1j
    
    bv = np.zeros(N, dtype=np.complex)
    bw = np.zeros(N, dtype=np.complex)
    vk0 = np.zeros(N, dtype=np.complex)
    wk0 = np.zeros(N, dtype=np.complex)
    
    vk0 = SD.fst(vk, vk0)
    vk  = SD.ifst(vk0, vk)
    vk0 = SD.fst(vk, vk0)
    wk0 = SD.fst(wk, wk0)
    wk  = SD.ifst(wk0, wk)
    wk0 = SD.fst(wk, wk0)

    SFTc.Mult_CTD_1D(N, vk0, wk0, bv, bw)
    
    cv = C.matvec(vk0)
    cw = C.matvec(wk0)
    cv /= B.dd
    cw /= B.dd
    
    assert np.allclose(cv, bv)
    assert np.allclose(cw, bw)

  
#test_ADDmat(ShenNeumannBasis("GL")) 
#test_Helmholtz(ShenDirichletBasis("GL")) 
#test_Mult_CTD(ShenDirichletBasis("GL"))
test_CDDmat(ShenDirichletBasis("GL"))
