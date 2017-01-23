import pytest
from spectralDNS.shen.shentransform import ShenDirichletBasis, ShenNeumannBasis, \
    ChebyshevTransform, ShenBiharmonicBasis, SlabShen_R2C
from spectralDNS.shen.la import TDMA, Helmholtz, Biharmonic
from spectralDNS.shen.Matrices import BNNmat, BTTmat, BDDmat, CDDmat, CDNmat, \
    BNDmat, CNDmat, BDNmat, ADDmat, ANNmat, CTDmat, BDTmat, CDTmat, BTDmat, \
    BTNmat, BBBmat, ABBmat, SBBmat, CDBmat, CBDmat, ATTmat, BBDmat, HelmholtzCoeff, \
    mass_matrix
from spectralDNS.shen import LUsolve
from scipy.linalg import solve
from mpi4py import MPI
from sympy import chebyshevt, Symbol, sin, cos, pi
import numpy as np
import scipy.sparse.linalg as la

comm = MPI.COMM_WORLD

N = 32
x = Symbol("x")

Basis = (ChebyshevTransform, ShenDirichletBasis, ShenNeumannBasis,
         ShenBiharmonicBasis)
quads = ('GC', 'GL')

@pytest.mark.parametrize('ST', Basis)
@pytest.mark.parametrize('quad', quads)
def test_scalarproduct(ST, quad):
    """Test fast scalar product against Vandermonde computed version"""
    ST = ST(quad=quad)
    points, weights = ST.points_and_weights(N,  ST.quad)
    f = x*x+cos(pi*x)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)
    u0 = np.zeros(N)
    u1 = np.zeros(N)
    ST.fast_transform = True
    u0 = ST.scalar_product(fj, u0)
    ST.fast_transform = False
    u1 = ST.scalar_product(fj, u1)
    assert np.allclose(u1, u0)


@pytest.mark.parametrize('ST', Basis[1:3])
@pytest.mark.parametrize('quad', quads)
def test_TDMA(ST, quad):
    from scipy.linalg import solve
    from numpy import zeros_like

    ST = ST(quad=quad)
    T = TDMA(ST)
    s = ST.slice(N)
    B = mass_matrix[ST.__class__.__name__](np.arange(N).astype(np.float), ST.quad)
    B = B.diags().toarray()[s, s]
    f = np.random.random(N)
    u = solve(B, f[s])

    u0 = f.copy()
    u0 = T(u0)

    assert np.allclose(u0[s], u)

    # Again
    u0 = f.copy()
    u0 = T(u0)
    assert np.allclose(u0[s], u)

    # Multidimensional version
    fc = f.repeat(N*N).reshape((N, N, N))+f.repeat(N*N).reshape((N, N, N))*1j
    u0 = fc.copy()
    u0 = T(u0)
    assert np.allclose(u0[s, 2, 2].real, u)
    assert np.allclose(u0[s, 2, 2].imag, u)

#test_TDMA(ShenNeumannBasis, 'GC')

#@profile
@pytest.mark.parametrize('ST', Basis)
@pytest.mark.parametrize('quad', quads)
def test_BNNmat(ST, quad):
    ST = ST(quad=quad)
    points, weights = ST.points_and_weights(N,  ST.quad)
    f_hat = np.zeros(N)
    fj = np.random.random(N)
    u0 = np.zeros(N)
    B = mass_matrix[ST.__class__.__name__](np.arange(N).astype(np.float), ST.quad)
    f_hat = ST.forward(fj, f_hat)
    fj = ST.backward(f_hat, fj)
    u0 = ST.scalar_product(fj, u0)
    f_hat = ST.forward(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)

    assert np.allclose(u2[:-2], u0[:-2])

    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    f_hat = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = ST.scalar_product(fj, u0)
    f_hat = ST.forward(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    assert np.allclose(u2[:-2], u0[:-2])

#test_BNNmat(ShenNeumannBasis, 'GC')

@pytest.mark.parametrize('quad', quads)
@pytest.mark.parametrize('mat', (BNDmat, BDNmat))
def test_BDNmat(mat, quad):
    B = mat(np.arange(N).astype(np.float), quad)
    S2 = B.trialfunction
    S1 = B.testfunction

    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = S2.fst(fj, f_hat)
    fj = S2.ifst(f_hat, fj)

    f_hat = S2.fst(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    u0 = np.zeros(N)
    u0 = S1.scalar_product(fj, u0)
    #from IPython import embed; embed()
    assert np.allclose(u0, u2)

    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    f_hat = f_hat.repeat(16).reshape((N, 4, 4)) + 1j*f_hat.repeat(16).reshape((N, 4, 4))

    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = S1.scalar_product(fj, u0)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    assert np.linalg.norm(u2-u0)/(N*16) < 1e-12

#test_BDNmat(BNDmat, "GL")

@pytest.mark.parametrize('quad', quads)
def test_BDTmat(quad):
    SD = ShenDirichletBasis(quad=quad)
    ST = ChebyshevTransform(quad=quad)

    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = ST.fct(fj, f_hat)
    fj = ST.ifct(f_hat, fj)

    B = BDTmat(np.arange(N).astype(np.float), SD.quad)

    f_hat = ST.fct(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    u2 = B.matvec(f_hat, u2, 'csr')
    u2 = B.matvec(f_hat, u2, 'csc')
    u2 = B.matvec(f_hat, u2, 'dia')
    u0 = np.zeros(N)
    u0 = SD.scalar_product(fj, u0)

    #from IPython import embed; embed()
    assert np.allclose(u0, u2)

    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    f_hat = f_hat.repeat(16).reshape((N, 4, 4)) + 1j*f_hat.repeat(16).reshape((N, 4, 4))

    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = SD.scalar_product(fj, u0)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    assert np.linalg.norm(u2-u0)/(N*16) < 1e-12

#test_BDTmat((ShenDirichletBasis("GL"), ShenNeumannBasis("GL")))

@pytest.mark.parametrize('quad', quads)
def test_BBDmat(quad):
    SB = ShenBiharmonicBasis(quad=quad)
    SD = ShenDirichletBasis(quad=quad)

    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = SD.fst(fj, f_hat)
    fj = SD.ifst(f_hat, fj)

    B = BBDmat(np.arange(N).astype(np.float), SB.quad)

    f_hat = SD.fst(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    u0 = np.zeros(N)
    u0 = SB.scalar_product(fj, u0)

    assert np.allclose(u0, u2)

    # Multidimensional version
    fj = fj.repeat(N*N).reshape((N, N, N)) + 1j*fj.repeat(N*N).reshape((N, N, N))
    f_hat = f_hat.repeat(N*N).reshape((N, N, N)) + 1j*f_hat.repeat(N*N).reshape((N, N, N))

    u0 = np.zeros((N, N, N), dtype=np.complex)
    u0 = SB.scalar_product(fj, u0)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    assert np.linalg.norm(u2-u0)/(N*N*N) < 1e-12

    FST = SlabShen_R2C(np.array([N, N, N]), np.array([2*pi, 2*pi, 2*pi]), MPI.COMM_SELF)
    f_hat = np.zeros(FST.complex_shape(), dtype=np.complex)
    fj = np.random.random((N, N, N))
    f_hat = FST.fst(fj, f_hat, SD)
    fj = FST.ifst(f_hat, fj, SD)
    f_hat = FST.fst(fj, f_hat, SD)

    z0 = np.zeros_like(f_hat)
    z0 = B.matvec(f_hat, z0)
    z1 = z0.copy()*0
    z1 = FST.fss(fj, z1, SB)
    assert np.linalg.norm(z1-z0)/(N*N*N) < 1e-12

#test_BBDmat((ShenBiharmonicBasis("GL"), ShenDirichletBasis("GL")))

@pytest.mark.parametrize('mat', (BTDmat, BTNmat))
@pytest.mark.parametrize('quad', quads)
def test_BTXmat(mat, quad):
    B = mat(np.arange(N).astype(np.float), quad)
    SX = B.trialfunction
    ST = B.testfunction

    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = SX.fst(fj, f_hat)
    fj = SX.ifst(f_hat, fj)

    f_hat = SX.fst(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    u0 = np.zeros(N)
    u0 = ST.scalar_product(fj, u0)

    #from IPython import embed; embed()
    assert np.allclose(u0, u2)

    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    f_hat = f_hat.repeat(16).reshape((N, 4, 4)) + 1j*f_hat.repeat(16).reshape((N, 4, 4))

    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = ST.scalar_product(fj, u0)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    assert np.linalg.norm(u2-u0)/(N*16) < 1e-12

#test_BTXmat(BTDmat, 'GC')

@pytest.mark.parametrize('ST', Basis)
@pytest.mark.parametrize('quad', quads)
def test_transforms(ST, quad):
    ST = ST(quad=quad)
    points, weights = ST.points_and_weights(N,  ST.quad)
    fj = np.random.random(N)

    # Project function to space first
    f_hat = np.zeros(N)
    f_hat = ST.forward(fj, f_hat)
    fj = ST.backward(f_hat, fj)

    # Then check if transformations work as they should
    u0 = np.zeros(N)
    u1 = np.zeros(N)
    u0 = ST.forward(fj, u0)
    u1 = ST.backward(u0, u1)

    assert np.allclose(fj, u1)

    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    u1 = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = ST.forward(fj, u0)
    u1 = ST.backward(u0, u1)

    assert np.allclose(fj, u1)

#test_transforms(ShenBiharmonicBasis("GC"))

@pytest.mark.parametrize('ST', Basis)
@pytest.mark.parametrize('quad', quads)
def test_FST(ST, quad):
    ST = ST(quad=quad)
    FST = SlabShen_R2C(np.array([N, N, N]), np.array([2*pi, 2*pi, 2*pi]), comm)

    if FST.rank == 0:

        FST_SELF = SlabShen_R2C(np.array([N, N, N]), np.array([2*pi, 2*pi, 2*pi]),
                                MPI.COMM_SELF)

        A = np.random.random((N, N, N)).astype(FST.float)
        B2 = np.zeros(FST_SELF.complex_shape(), dtype=FST.complex)

        B2 = FST_SELF.forward(A, B2, ST.forward)
        A = FST_SELF.backward(B2, A, ST.backward)
        B2 = FST_SELF.forward(A, B2, ST.forward)

    else:
        A = np.zeros((N, N, N), dtype=FST.float)
        B2 = np.zeros((N, N, N/2+1), dtype=FST.complex)

    atol, rtol = (1e-10, 1e-8) if FST.float is np.float64 else (5e-7, 1e-4)
    FST.comm.Bcast(A, root=0)
    FST.comm.Bcast(B2, root=0)

    a = np.zeros(FST.real_shape(), dtype=FST.float)
    c = np.zeros(FST.complex_shape(), dtype=FST.complex)
    a[:] = A[FST.real_local_slice()]
    c = FST.forward(a, c, ST.forward)

    assert np.all(abs((c - B2[FST.complex_local_slice()])/c.max()) < rtol)

    a = FST.backward(c, a, ST.backward)

    assert np.all(abs((a - A[FST.real_local_slice()])/a.max()) < rtol)

#test_FST(ShenDirichletBasis, 'GC')

@pytest.mark.parametrize('ST', Basis)
@pytest.mark.parametrize('quad', quads)
def test_FST_padded(ST, quad):
    ST = ST(quad=quad)
    M = np.array([N, 2*N, 4*N])
    FST = SlabShen_R2C(M, np.array([2*pi, 2*pi, 2*pi]), comm,
                       communication='Alltoall')
    FST_SELF = SlabShen_R2C(M, np.array([2*pi, 2*pi, 2*pi]),
                            MPI.COMM_SELF)

    if FST.rank == 0:
        A = np.random.random(M).astype(FST.float)
        A_hat = np.zeros(FST_SELF.complex_shape(), dtype=FST.complex)

        A_hat = FST_SELF.forward(A, A_hat, ST.forward)
        A = FST_SELF.backward(A_hat, A, ST.backward)
        A_hat = FST_SELF.forward(A, A_hat, ST.forward)

        A_hat[:, -M[1]/2] = 0

        A_pad = np.zeros(FST_SELF.real_shape_padded(), dtype=FST.float)
        A_pad = FST_SELF.backward(A_hat, A_pad, ST.backward, dealias='3/2-rule')
        A_hat = FST_SELF.forward(A_pad, A_hat, ST.forward, dealias='3/2-rule')

    else:
        A_pad = np.zeros(FST_SELF.real_shape_padded(), dtype=FST.float)
        A_hat = np.zeros(FST_SELF.complex_shape(), dtype=FST.complex)

    atol, rtol = (1e-10, 1e-8) if FST.float is np.float64 else (5e-7, 1e-4)
    FST.comm.Bcast(A_pad, root=0)
    FST.comm.Bcast(A_hat, root=0)

    a = np.zeros(FST.real_shape_padded(), dtype=FST.float)
    c = np.zeros(FST.complex_shape(), dtype=FST.complex)
    a[:] = A_pad[FST.real_local_slice(padsize=1.5)]
    c = FST.forward(a, c, ST.forward, dealias='3/2-rule')

    assert np.all(abs((c - A_hat[FST.complex_local_slice()])/c.max()) < rtol)

    a = FST.backward(c, a, ST.backward, dealias='3/2-rule')

    #print abs((a - A_pad[FST.real_local_slice(padsize=1.5)])/a.max())
    assert np.all(abs((a - A_pad[FST.real_local_slice(padsize=1.5)])/a.max()) < rtol)


#test_FST_padded(ShenBiharmonicBasis, 'GC')

@pytest.mark.parametrize('quad', quads)
def test_CDDmat(quad):
    SD = ShenDirichletBasis(quad=quad)
    M = 256
    u = (1-x**2)*sin(np.pi*6*x)
    dudx = u.diff(x, 1)
    points, weights = SD.points_and_weights(M,  SD.quad)
    dudx_j = np.array([dudx.subs(x, h) for h in points], dtype=np.float)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)

    dudx_j = np.zeros(M)
    u_hat = np.zeros(M)
    u_hat = SD.fst(uj, u_hat)
    uj = SD.ifst(u_hat, uj)
    u_hat = SD.fst(uj, u_hat)

    uc_hat = np.zeros(M)
    uc_hat = SD.CT.fct(uj, uc_hat)
    du_hat = np.zeros(M)
    dudx_j = SD.CT.fast_cheb_derivative(uj, dudx_j)

    Cm = CDDmat(np.arange(M).astype(np.float))
    TDMASolver = TDMA(SD)

    cs = np.zeros_like(u_hat)
    cs = Cm.matvec(u_hat, cs)

    # Should equal (but not exact so use extra resolution)
    cs2 = np.zeros(M)
    cs2 = SD.scalar_product(dudx_j, cs2)

    assert np.allclose(cs, cs2)

    cs = TDMASolver(cs)
    du = np.zeros(M)
    du = SD.ifst(cs, du)

    assert np.linalg.norm(du-dudx_j)/M < 1e-10

    # Multidimensional version
    u3_hat = u_hat.repeat(4*4).reshape((M, 4, 4)) + 1j*u_hat.repeat(4*4).reshape((M, 4, 4))
    cs = np.zeros_like(u3_hat)
    cs = Cm.matvec(u3_hat, cs)
    cs2 = np.zeros((M, 4, 4), dtype=np.complex)
    du3 = dudx_j.repeat(4*4).reshape((M, 4, 4)) + 1j*dudx_j.repeat(4*4).reshape((M, 4, 4))
    cs2 = SD.scalar_product(du3, cs2)

    assert np.allclose(cs, cs2, 1e-10)

    cs = TDMASolver(cs)
    d3 = np.zeros((M, 4, 4), dtype=np.complex)
    d3 = SD.ifst(cs, d3)

    #from IPython import embed; embed()
    assert np.linalg.norm(du3-d3)/(M*16) < 1e-10

#test_CDDmat('GL')

@pytest.mark.parametrize('mat', (CDDmat, CDNmat, CNDmat))
def test_CXXmat(mat):
    Cm = mat(np.arange(N).astype(np.float))
    S2 = Cm.trialfunction
    S1 = Cm.testfunction

    fj = np.random.randn(N)
    # project to S2
    f_hat = np.zeros(N)
    f_hat = S2.fst(fj, f_hat)
    fj = S2.ifst(f_hat, fj)

    # Check S1.fss(f) equals Cm*S2.fst(f)
    f_hat = S2.fst(fj, f_hat)
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    df = np.zeros(N)
    df = S2.CT.fast_cheb_derivative(fj, df)
    cs2 = np.zeros(N)
    cs2 = S1.scalar_product(df, cs2)

    assert np.allclose(cs, cs2)

    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((N, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((N, 4, 4))
    df = df.repeat(4*4).reshape((N, 4, 4)) + 1j*df.repeat(4*4).reshape((N, 4, 4))
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    cs2 = np.zeros((N, 4, 4), dtype=np.complex)
    cs2 = S1.scalar_product(df, cs2)

    assert np.allclose(cs, cs2)

#test_CXXmat(CNDmat)

@pytest.mark.parametrize('quad', quads)
def test_CDTmat(quad):
    SD = ShenDirichletBasis(quad=quad)
    ST = ChebyshevTransform(quad=quad)

    Cm = CDTmat(np.arange(N).astype(np.float))

    fj = np.random.randn(N)
    # project to ST
    f_hat = np.zeros(N)
    f_hat = ST.fct(fj, f_hat)
    fj = ST.ifct(f_hat, fj)

    # Check SD.fss(f) equals Cm*ST.fst(f)
    f_hat = ST.fct(fj, f_hat)
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    df = np.zeros(N)
    df = ST.fast_cheb_derivative(fj, df)
    cs2 = np.zeros(N)
    cs2 = SD.scalar_product(df, cs2)

    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)

    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((N, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((N, 4, 4))
    df = df.repeat(4*4).reshape((N, 4, 4)) + 1j*df.repeat(4*4).reshape((N, 4, 4))
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    cs2 = np.zeros((N, 4, 4), dtype=np.complex)
    cs2 = SD.scalar_product(df, cs2)

    assert np.allclose(cs, cs2)

#test_CDTmat((ShenDirichletBasis('GL'), ChebyshevTransform('GL')))

@pytest.mark.parametrize('quad', quads)
def test_CTDmat(quad):
    SD = ShenDirichletBasis(quad=quad)
    ST = ChebyshevTransform(quad=quad)

    Cm = CTDmat(np.arange(N).astype(np.float))

    fj = np.random.randn(N)
    # project to SD
    f_hat = np.zeros(N)
    f_hat = SD.fst(fj, f_hat)
    fj = SD.ifst(f_hat, fj)

    # Check if ST.fcs(f') equals Cm*SD.fst(f)
    f_hat = SD.fst(fj, f_hat)
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    df = np.zeros(N)
    df = ST.fast_cheb_derivative(fj, df)
    cs2 = np.zeros(N)
    cs2 = ST.scalar_product(df, cs2)

    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)

    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((N, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((N, 4, 4))
    df = df.repeat(4*4).reshape((N, 4, 4)) + 1j*df.repeat(4*4).reshape((N, 4, 4))
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    cs2 = np.zeros((N, 4, 4), dtype=np.complex)
    cs2 = ST.scalar_product(df, cs2)

    assert np.allclose(cs, cs2)

#test_CTDmat((ShenDirichletBasis('GC'), ChebyshevTransform('GC')))

@pytest.mark.parametrize('quad', quads)
def test_CDBmat(quad):
    SB = ShenBiharmonicBasis(quad=quad)
    SD = ShenDirichletBasis(quad=quad)

    M = 8*N
    Cm = CDBmat(np.arange(M).astype(np.float))

    x = Symbol("x")
    u = sin(2*pi*x)**2
    f = u.diff(x, 1)

    points, weights = SB.points_and_weights(M,  SB.quad)

    uj = np.array([u.subs(x, j) for j in points], dtype=float)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

    # project to SB
    f_hat = np.zeros(M)
    f_hat = SB.fst(uj, f_hat)
    uj = SB.ifst(f_hat, uj)

    # Check if SD.fss(f') equals Cm*SD.fst(f)
    f_hat = SB.fst(uj, f_hat)
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)

    df = np.zeros(M)
    df = SB.CT.fast_cheb_derivative(uj, df)
    cs2 = np.zeros(M)
    cs2 = SD.scalar_product(df, cs2)

    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)

    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((M, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((M, 4, 4))
    df = df.repeat(4*4).reshape((M, 4, 4)) + 1j*df.repeat(4*4).reshape((M, 4, 4))
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    cs2 = np.zeros((M, 4, 4), dtype=np.complex)
    cs2 = SD.scalar_product(df, cs2)

    assert np.allclose(cs, cs2)

#test_CDBmat((ShenBiharmonicBasis("GC"), ShenDirichletBasis("GC")))

@pytest.mark.parametrize('quad', quads)
def test_CBDmat(quad):
    SB = ShenBiharmonicBasis(quad=quad)
    SD = ShenDirichletBasis(quad=quad)

    M = 4*N
    Cm = CBDmat(np.arange(M).astype(np.float))

    x = Symbol("x")
    u = sin(12*pi*x)**2
    f = u.diff(x, 1)

    points, weights = SD.points_and_weights(M,  SD.quad)

    uj = np.array([u.subs(x, j) for j in points], dtype=float)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

    # project to SD
    f_hat = np.zeros(M)
    f_hat = SD.fst(uj, f_hat)
    uj = SD.ifst(f_hat, uj)

    # Check if SB.fss(f') equals Cm*SD.fst(f)
    f_hat = SD.fst(uj, f_hat)
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)

    df = np.zeros(M)
    df = SD.CT.fast_cheb_derivative(uj, df)
    cs2 = np.zeros(M)
    cs2 = SB.scalar_product(df, cs2)

    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)

    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((M, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((M, 4, 4))
    df = df.repeat(4*4).reshape((M, 4, 4)) + 1j*df.repeat(4*4).reshape((M, 4, 4))
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    cs2 = np.zeros((M, 4, 4), dtype=np.complex)
    cs2 = SB.scalar_product(df, cs2)

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

    LUsolve.Mult_Div_1D(N, 7, 7, uk0[:N-2], vk0[:N-2], wk0[:N-2], b[1:N-2])

    uu = np.zeros_like(uk0)
    v0 = np.zeros_like(vk0)
    w0 = np.zeros_like(wk0)
    uu = Cm.matvec(uk0, uu)
    uu += 1j*7*Bm.matvec(vk0, v0) + 1j*7*Bm.matvec(wk0, w0)

    #from IPython import embed; embed()
    assert np.allclose(uu, b)

    uk0 = uk0.repeat(4*4).reshape((N,4,4)) + 1j*uk0.repeat(4*4).reshape((N,4,4))
    vk0 = vk0.repeat(4*4).reshape((N,4,4)) + 1j*vk0.repeat(4*4).reshape((N,4,4))
    wk0 = wk0.repeat(4*4).reshape((N,4,4)) + 1j*wk0.repeat(4*4).reshape((N,4,4))
    b = np.zeros((N,4,4), dtype=np.complex)
    m = np.zeros((4,4))+7
    n = np.zeros((4,4))+7
    LUsolve.Mult_Div_3D(N, m, n, uk0[:N-2], vk0[:N-2], wk0[:N-2], b[1:N-2])

    uu = np.zeros_like(uk0)
    v0 = np.zeros_like(vk0)
    w0 = np.zeros_like(wk0)
    uu = Cm.matvec(uk0, uu)
    uu += 1j*7*Bm.matvec(vk0, v0) + 1j*7*Bm.matvec(wk0, w0)

    assert np.allclose(uu, b)

#test_Mult_Div()

@pytest.mark.parametrize('ST', Basis[1:3])
@pytest.mark.parametrize('quad', quads)
def test_ADDmat(ST, quad):
    ST = ST(quad=quad)
    M = 2*N
    u = (1-x**2)*sin(np.pi*x)
    f = -u.diff(x, 2)

    points, weights = ST.points_and_weights(M,  quad)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    fj = np.array([f.subs(x, h) for h in points], dtype=np.float)
    s = ST.slice(M)

    if ST.__class__.__name__ == "ShenDirichletBasis":
        A = ADDmat(np.arange(M).astype(np.float))
    elif ST.__class__.__name__ == "ShenNeumannBasis":
        A = ANNmat(np.arange(M).astype(np.float))
        fj -= np.dot(fj, weights)/weights.sum()
        uj -= np.dot(uj, weights)/weights.sum()

    f_hat = np.zeros(M)
    f_hat = ST.scalar_product(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[s] = solve(A.diags().toarray()[s,s], f_hat[s])

    u0 = np.zeros(M)
    u0 = ST.ifst(u_hat, u0)

    #from IPython import embed; embed()
    assert np.allclose(u0, uj)

    u1 = np.zeros(M)
    u1 = ST.fst(uj, u1)
    c = np.zeros_like(u1)
    c = A.matvec(u1, c)

    assert np.allclose(c, f_hat)

#test_ADDmat(ShenNeumannBasis, "GL")

@pytest.mark.parametrize('quad', quads)
def test_SBBmat(quad):
    SB = ShenBiharmonicBasis(quad=quad)
    M = 72
    u = sin(4*pi*x)**2
    f = u.diff(x, 4)

    points, weights = SB.points_and_weights(M,  SB.quad)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    fj = np.array([f.subs(x, h) for h in points], dtype=np.float)

    A = SBBmat(np.arange(M).astype(np.float))
    f_hat = np.zeros(M)
    f_hat = SB.scalar_product(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[:-4] = la.spsolve(A.diags(), f_hat[:-4])

    u0 = np.zeros(M)
    u0 = SB.ifst(u_hat, u0)

    assert np.allclose(u0, uj)

    u1 = np.zeros(M)
    u1 = SB.fst(uj, u1)

    c = np.zeros_like(u1)
    c = A.matvec(u1, c)

    #from IPython import embed; embed()
    assert np.all(abs(c-f_hat)/c.max() < 1e-10)

    # Multidimensional
    c2 = (c.repeat(16).reshape((M, 4, 4))+1j*c.repeat(16).reshape((M, 4, 4)))
    u1 = (u1.repeat(16).reshape((M, 4, 4))+1j*u1.repeat(16).reshape((M, 4, 4)))

    c = np.zeros_like(u1)
    c = A.matvec(u1, c)

    assert np.allclose(c, c2)

#test_SBBmat(ShenBiharmonicBasis("GC"))

@pytest.mark.parametrize('quad', quads)
def test_ABBmat(quad):
    SB = ShenBiharmonicBasis(quad=quad)
    M = 6*N
    u = sin(6*pi*x)**2
    f = u.diff(x, 2)

    points, weights = SB.points_and_weights(M,  SB.quad)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    fj = np.array([f.subs(x, h) for h in points], dtype=np.float)

    A = ABBmat(np.arange(M).astype(np.float))

    f_hat = np.zeros(M)
    f_hat = SB.scalar_product(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[:-4] = la.spsolve(A.diags(), f_hat[:-4])

    u0 = np.zeros(M)
    u0 = SB.ifst(u_hat, u0)

    assert np.allclose(u0, uj)

    u1 = np.zeros(M)
    u1 = SB.fst(uj, u1)
    c = np.zeros_like(u1)
    c = A.matvec(u1, c)

    assert np.allclose(c, f_hat, 1e-6, 1e-6)

    # Multidimensional
    f_hat = (f_hat.repeat(16).reshape((M, 4, 4))+1j*f_hat.repeat(16).reshape((M, 4, 4)))
    u1 = (u1.repeat(16).reshape((M, 4, 4))+1j*u1.repeat(16).reshape((M, 4, 4)))

    c = np.zeros_like(u1)
    c = A.matvec(u1, c)

    assert np.allclose(c, f_hat, 1e-6, 1e-6)

    B = BBBmat(np.arange(M).astype(np.float), SB.quad)
    u0 = np.random.randn(M)
    u0_hat = np.zeros(M)
    u0_hat = SB.fst(u0, u0_hat)
    u0 = SB.ifst(u0_hat, u0)
    b = np.zeros(M)
    k = 2.
    c0 = np.zeros_like(u0_hat)
    c1 = np.zeros_like(u0_hat)
    b = A.matvec(u0_hat, c0) - k**2*B.matvec(u0_hat, c1)
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
    c0 = np.zeros_like(u0_hat)
    c1 = np.zeros_like(u0_hat)
    b = A.matvec(u0_hat, c0) - k**2*B.matvec(u0_hat, c1)
    alfa = np.ones((M, 4, 4))

    BH = Biharmonic(M, 0, alfa[0], -k2[0], SB.quad, "cython")
    z0_hat = np.zeros((M, 4, 4), dtype=np.complex)
    z0_hat = BH(z0_hat, b)
    z0 = np.zeros((M, 4, 4))
    z0 = SB.ifst(z0_hat.real, z0)
    #from IPython import embed; embed()
    assert np.allclose(z0, u0)

#test_ABBmat(ShenBiharmonicBasis("GC"))

@pytest.mark.parametrize('ST', Basis[1:3])
@pytest.mark.parametrize('quad', quads)
def test_Helmholtz(ST, quad):
    ST = ST(quad=quad)
    M = 4*N
    kx = 12

    points, weights = ST.points_and_weights(M,  ST.quad)

    fj = np.random.randn(M)
    f_hat = np.zeros(M)
    f_hat = ST.forward(fj, f_hat)
    fj = ST.backward(f_hat, fj)
    s = ST.slice(M)

    if ST.__class__.__name__ == "ShenDirichletBasis":
        A = ADDmat(np.arange(M).astype(np.float))
        B = BDDmat(np.arange(M).astype(np.float), ST.quad)
    elif ST.__class__.__name__ == "ShenNeumannBasis":
        A = ANNmat(np.arange(M).astype(np.float))
        B = BNNmat(np.arange(M).astype(np.float), ST.quad)

    f_hat = np.zeros(M)
    f_hat = ST.scalar_product(fj, f_hat)
    u_hat = np.zeros(M)
    H = A + kx**2*B
    u_hat[s] = solve(H.diags().toarray()[s, s], f_hat[s])
    u1 = np.zeros(M)
    u1 = ST.backward(u_hat, u1)
    c0 = np.zeros_like(u_hat)
    c1 = np.zeros_like(u_hat)
    c = A.matvec(u_hat, c0)+kx**2*B.matvec(u_hat, c1)
    c2 = np.dot(A.diags().toarray()[s, s], u_hat[s]) + kx**2*np.dot(B.diags().toarray()[s, s], u_hat[s])

    #from IPython import embed; embed()
    assert np.allclose(c, f_hat)
    assert np.allclose(c[s], c2)

    H = Helmholtz(M, kx, ST)
    u0_hat = np.zeros(M)
    u0_hat = H(u0_hat, f_hat)
    u0 = np.zeros(M)
    u0 = ST.backward(u0_hat, u0)

    assert np.linalg.norm(u0 - u1) < 1e-12


    # Multidimensional
    f_hat = (f_hat.repeat(16).reshape((M, 4, 4))+1j*f_hat.repeat(16).reshape((M, 4, 4)))
    kx = np.zeros((4, 4))+12
    H = Helmholtz(M, kx, ST)
    u0_hat = np.zeros((M, 4, 4), dtype=np.complex)
    u0_hat = H(u0_hat, f_hat)
    u0 = np.zeros((M, 4, 4), dtype=np.complex)
    u0 = ST.backward(u0_hat, u0)

    assert np.linalg.norm(u0[:, 2, 2].real - u1)/(M*16) < 1e-12
    assert np.linalg.norm(u0[:, 2, 2].imag - u1)/(M*16) < 1e-12

#test_Helmholtz(ShenNeumannBasis, "GC")

@pytest.mark.parametrize('quad', quads)
def test_Helmholtz2(quad):
    SD = ShenDirichletBasis(quad=quad)
    M = 2*N
    kx = 11
    points, weights = SD.points_and_weights(M,  SD.quad)
    uj = np.random.randn(M)
    u_hat = np.zeros(M)
    u_hat = SD.fst(uj, u_hat)
    uj = SD.ifst(u_hat, uj)

    A = ADDmat(np.arange(M).astype(np.float))
    B = BDDmat(np.arange(M).astype(np.float), SD.quad)
    s = SD.slice(M)

    u1 = np.zeros(M)
    u1 = SD.fst(uj, u1)
    c0 = np.zeros_like(u1)
    c1 = np.zeros_like(u1)
    c = A.matvec(u1, c0)+kx**2*B.matvec(u1, c1)

    b = np.zeros(M)
    H = Helmholtz(M, kx, SD)
    b = H.matvec(u1, b)
    #LUsolve.Mult_Helmholtz_1D(M, SD.quad=="GL", 1, kx**2, u1, b)
    assert np.allclose(c, b)

    b = np.zeros((M, 4, 4), dtype=np.complex)
    u1 = u1.repeat(16).reshape((M, 4, 4)) +1j*u1.repeat(16).reshape((M, 4, 4))
    kx = np.zeros((4, 4))+kx
    H = Helmholtz(M, kx, SD)
    b = H.matvec(u1, b)
    #LUsolve.Mult_Helmholtz_3D_complex(M, SD.quad=="GL", 1.0, kx**2, u1, b)
    assert np.linalg.norm(b[:, 2, 2].real - c)/(M*16) < 1e-12
    assert np.linalg.norm(b[:, 2, 2].imag - c)/(M*16) < 1e-12

#test_Helmholtz2('GL')

@pytest.mark.parametrize('quad', quads)
def test_Mult_CTD(quad):
    SD = ShenDirichletBasis(quad=quad)
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

    LUsolve.Mult_CTD_1D(N, vk0, wk0, bv, bw)

    cv = np.zeros_like(vk0)
    cw = np.zeros_like(wk0)
    cv = C.matvec(vk0, cv)
    cw = C.matvec(wk0, cw)
    cv /= B[0]
    cw /= B[0]

    #from IPython import embed; embed()
    assert np.allclose(cv, bv)
    assert np.allclose(cw, bw)

#test_Mult_CTD("GL")

@pytest.mark.parametrize('quad', quads)
def test_Mult_CTD_3D(quad):
    SD = ShenDirichletBasis(quad=quad)
    C = CTDmat(np.arange(N).astype(np.float))
    B = BTTmat(np.arange(N).astype(np.float), SD.quad)

    uk = np.random.random((N,4,4))+np.random.random((N,4,4))*1j
    vk = np.random.random((N,4,4))+np.random.random((N,4,4))*1j
    wk = np.random.random((N,4,4))+np.random.random((N,4,4))*1j

    bv = np.zeros((N,4,4), dtype=np.complex)
    bw = np.zeros((N,4,4), dtype=np.complex)
    vk0 = np.zeros((N,4,4), dtype=np.complex)
    wk0 = np.zeros((N,4,4), dtype=np.complex)
    cv = np.zeros((N,4,4), dtype=np.complex)
    cw = np.zeros((N,4,4), dtype=np.complex)

    vk0 = SD.fst(vk, vk0)
    vk  = SD.ifst(vk0, vk)
    vk0 = SD.fst(vk, vk0)
    wk0 = SD.fst(wk, wk0)
    wk  = SD.ifst(wk0, wk)
    wk0 = SD.fst(wk, wk0)

    #from IPython import embed; embed()
    LUsolve.Mult_CTD_3D_n(N, vk0, wk0, bv, bw)

    cv = np.zeros_like(vk0)
    cw = np.zeros_like(wk0)
    cv = C.matvec(vk0, cv)
    cw = C.matvec(wk0, cw)
    cv /= B[0].repeat(np.array(bv.shape[1:]).prod()).reshape(bv.shape)
    cw /= B[0].repeat(np.array(bv.shape[1:]).prod()).reshape(bv.shape)

    assert np.allclose(cv, bv)
    assert np.allclose(cw, bw)

#test_Mult_CTD_3D("GL")

@pytest.mark.parametrize('quad', quads)
def test_Biharmonic(quad):
    SB = ShenBiharmonicBasis(quad=quad)
    M = 128
    x = Symbol("x")
    u = sin(6*pi*x)**2
    a = 1.0
    b = 1.0
    f = -u.diff(x, 4) + a*u.diff(x, 2) + b*u

    points, weights = SB.points_and_weights(M,  SB.quad)

    uj = np.array([u.subs(x, j) for j in points], dtype=float)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

    k = np.arange(M).astype(np.float)
    A = SBBmat(k)
    B = BBBmat(k, SB.quad)
    C = ABBmat(k)

    AA = -A.diags() + C.diags() + B.diags()
    f_hat = np.zeros(M)
    f_hat = SB.scalar_product(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[:-4] = la.spsolve(AA, f_hat[:-4])
    u1 = np.zeros(M)
    u1 = SB.ifst(u_hat, u1)
    #from IPython import embed; embed()

    assert np.allclose(u1, uj)

#test_Biharmonic("GC")

@pytest.mark.parametrize('quad', quads)
def test_Helmholtz_matvec(quad):
    SD = ShenDirichletBasis(quad=quad)
    M = 2*N
    kx = 11
    points, weights = SD.points_and_weights(M,  SD.quad)
    uj = np.random.randn(M)
    u_hat = np.zeros(M)
    u_hat = SD.fst(uj, u_hat)
    uj = SD.ifst(u_hat, uj)

    A = ADDmat(np.arange(M).astype(np.float))
    B = BDDmat(np.arange(M).astype(np.float), SD.quad)
    AB = HelmholtzCoeff(np.arange(M).astype(np.float), 1, kx**2, SD.quad)
    s = SD.slice(M)

    u1 = np.zeros(M)
    u1 = SD.fst(uj, u1)
    c0 = np.zeros_like(u1)
    c1 = np.zeros_like(u1)
    c = A.matvec(u1, c0)+kx**2*B.matvec(u1, c1)

    b = np.zeros(M)
    #LUsolve.Mult_Helmholtz_1D(M, SD.quad=="GL", 1, kx**2, u1, b)
    b = AB.matvec(u1, b)
    assert np.allclose(c, b)

    b = np.zeros((M, 4, 4), dtype=np.complex)
    u1 = u1.repeat(16).reshape((M, 4, 4)) +1j*u1.repeat(16).reshape((M, 4, 4))
    kx = np.zeros((4, 4))+kx
    #LUsolve.Mult_Helmholtz_3D_complex(M, SD.quad=="GL", 1.0, kx**2, u1, b)
    AB = HelmholtzCoeff(np.arange(M).astype(np.float), 1, kx**2, SD.quad)
    b = AB.matvec(u1, b)

    assert np.linalg.norm(b[:, 2, 2].real - c)/(M*16) < 1e-12
    assert np.linalg.norm(b[:, 2, 2].imag - c)/(M*16) < 1e-12

#test_Helmholtz_matvec("GL")

#test_ADDmat(ShenNeumannBasis("GL"))
#test_Helmholtz2(ShenDirichletBasis("GL"))
#test_Mult_CTD(ShenDirichletBasis("GL"))
#test_CDDmat(ShenDirichletBasis("GL"))
