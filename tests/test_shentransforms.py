import pytest
from spectralDNS.shen.shentransform import SlabShen_R2C
from spectralDNS.shen.la import Helmholtz, Biharmonic
from spectralDNS.shen.Matrices import HelmholtzCoeff
from spectralDNS.shen import LUsolve
from scipy.linalg import solve
from mpi4py import MPI
from sympy import chebyshevt, Symbol, sin, cos, pi
import numpy as np
import scipy.sparse.linalg as la
from shenfun.chebyshev import matrices
from shenfun.chebyshev.bases import ChebyshevBasis, ShenDirichletBasis, \
    ShenNeumannBasis, ShenBiharmonicBasis

comm = MPI.COMM_WORLD

N = 32
x = Symbol("x")

Basis = (ChebyshevBasis, ShenDirichletBasis, ShenNeumannBasis,
         ShenBiharmonicBasis)
quads = ('GC', 'GL')


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

        B2 = FST_SELF.forward(A, B2, ST)
        A = FST_SELF.backward(B2, A, ST)
        B2 = FST_SELF.forward(A, B2, ST)

    else:
        A = np.zeros((N, N, N), dtype=FST.float)
        B2 = np.zeros((N, N, N/2+1), dtype=FST.complex)

    atol, rtol = (1e-10, 1e-8) if FST.float is np.float64 else (5e-7, 1e-4)
    FST.comm.Bcast(A, root=0)
    FST.comm.Bcast(B2, root=0)

    a = np.zeros(FST.real_shape(), dtype=FST.float)
    c = np.zeros(FST.complex_shape(), dtype=FST.complex)
    a[:] = A[FST.real_local_slice()]
    c = FST.forward(a, c, ST)

    assert np.all(abs((c - B2[FST.complex_local_slice()])/c.max()) < rtol)

    a = FST.backward(c, a, ST)

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

        A_hat = FST_SELF.forward(A, A_hat, ST)
        A = FST_SELF.backward(A_hat, A, ST)
        A_hat = FST_SELF.forward(A, A_hat, ST)

        A_hat[:, -M[1]/2] = 0

        A_pad = np.zeros(FST_SELF.real_shape_padded(), dtype=FST.float)
        A_pad = FST_SELF.backward(A_hat, A_pad, ST, dealias='3/2-rule')
        A_hat = FST_SELF.forward(A_pad, A_hat, ST, dealias='3/2-rule')

    else:
        A_pad = np.zeros(FST_SELF.real_shape_padded(), dtype=FST.float)
        A_hat = np.zeros(FST_SELF.complex_shape(), dtype=FST.complex)

    atol, rtol = (1e-10, 1e-8) if FST.float is np.float64 else (5e-7, 1e-4)
    FST.comm.Bcast(A_pad, root=0)
    FST.comm.Bcast(A_hat, root=0)

    a = np.zeros(FST.real_shape_padded(), dtype=FST.float)
    c = np.zeros(FST.complex_shape(), dtype=FST.complex)
    a[:] = A_pad[FST.real_local_slice(padsize=1.5)]
    c = FST.forward(a, c, ST, dealias='3/2-rule')

    assert np.all(abs((c - A_hat[FST.complex_local_slice()])/c.max()) < rtol)

    a = FST.backward(c, a, ST, dealias='3/2-rule')

    #print abs((a - A_pad[FST.real_local_slice(padsize=1.5)])/a.max())
    assert np.all(abs((a - A_pad[FST.real_local_slice(padsize=1.5)])/a.max()) < rtol)


#test_FST_padded(ShenBiharmonicBasis, 'GC')


def test_Mult_Div():

    SD = ShenDirichletBasis("GC")
    SN = ShenDirichletBasis("GC")

    Cm = matrices.CNDmat(np.arange(N).astype(np.float))
    Bm = matrices.BNDmat(np.arange(N).astype(np.float), "GC")

    uk = np.random.randn((N))+np.random.randn((N))*1j
    vk = np.random.randn((N))+np.random.randn((N))*1j
    wk = np.random.randn((N))+np.random.randn((N))*1j

    b = np.zeros(N, dtype=np.complex)
    uk0 = np.zeros(N, dtype=np.complex)
    vk0 = np.zeros(N, dtype=np.complex)
    wk0 = np.zeros(N, dtype=np.complex)

    uk0 = SD.forward(uk, uk0)
    uk  = SD.backward(uk0, uk)
    uk0 = SD.forward(uk, uk0)
    vk0 = SD.forward(vk, vk0)
    vk  = SD.backward(vk0, vk)
    vk0 = SD.forward(vk, vk0)
    wk0 = SD.forward(wk, wk0)
    wk  = SD.backward(wk0, wk)
    wk0 = SD.forward(wk, wk0)

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
        A = matrices.ADDmat(np.arange(M).astype(np.float))
        B = matrices.BDDmat(np.arange(M).astype(np.float), ST.quad)
    elif ST.__class__.__name__ == "ShenNeumannBasis":
        A = matrices.ANNmat(np.arange(M).astype(np.float))
        B = matrices.BNNmat(np.arange(M).astype(np.float), ST.quad)

    f_hat = np.zeros(M)
    f_hat = ST.scalar_product(fj, f_hat)
    u_hat = np.zeros(M)
    H = A + kx**2*B
    u_hat[s] = solve(H.diags().toarray()[s, s], f_hat[s])
    if ST.__class__.__name__ == "ShenNeumannBasis":
        u_hat[0] = 0
    u1 = np.zeros(M)
    u1 = ST.backward(u_hat, u1)
    c0 = np.zeros_like(u_hat)
    c1 = np.zeros_like(u_hat)
    c = A.matvec(u_hat, c0)+kx**2*B.matvec(u_hat, c1)
    c2 = np.dot(A.diags().toarray()[s, s], u_hat[s]) + kx**2*np.dot(B.diags().toarray()[s, s], u_hat[s])
    if ST.__class__.__name__ == "ShenNeumannBasis":
        c2[0] = 0

    assert np.allclose(c[s], f_hat[s])
    assert np.allclose(c[s], c2)

    H = Helmholtz(M, kx, ST)
    u0_hat = np.zeros(M)
    u0_hat = H(u0_hat, f_hat)
    u0 = np.zeros(M)
    u0 = ST.backward(u0_hat, u0)
    #from IPython import embed; embed()
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

#test_Helmholtz(ShenDirichletBasis, "GC")

@pytest.mark.parametrize('quad', quads)
def test_Helmholtz2(quad):
    SD = ShenDirichletBasis(quad=quad)
    M = 2*N
    kx = 11
    points, weights = SD.points_and_weights(M,  SD.quad)
    uj = np.random.randn(M)
    u_hat = np.zeros(M)
    u_hat = SD.forward(uj, u_hat)
    uj = SD.backward(u_hat, uj)

    A = matrices.ADDmat(np.arange(M).astype(np.float))
    B = matrices.BDDmat(np.arange(M).astype(np.float), SD.quad)
    s = SD.slice(M)

    u1 = np.zeros(M)
    u1 = SD.forward(uj, u1)
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
    C = matrices.CTDmat(np.arange(N).astype(np.float))
    B = matrices.BTTmat(np.arange(N).astype(np.float), SD.quad)

    uk = np.random.randn((N))+np.random.randn((N))*1j
    vk = np.random.randn((N))+np.random.randn((N))*1j
    wk = np.random.randn((N))+np.random.randn((N))*1j

    bv = np.zeros(N, dtype=np.complex)
    bw = np.zeros(N, dtype=np.complex)
    vk0 = np.zeros(N, dtype=np.complex)
    wk0 = np.zeros(N, dtype=np.complex)
    cv = np.zeros(N, dtype=np.complex)
    cw = np.zeros(N, dtype=np.complex)

    vk0 = SD.forward(vk, vk0)
    vk  = SD.backward(vk0, vk)
    vk0 = SD.forward(vk, vk0)
    wk0 = SD.forward(wk, wk0)
    wk  = SD.backward(wk0, wk)
    wk0 = SD.forward(wk, wk0)

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
    C = matrices.CTDmat(np.arange(N).astype(np.float))
    B = matrices.BTTmat(np.arange(N).astype(np.float), SD.quad)

    uk = np.random.random((N,4,4))+np.random.random((N,4,4))*1j
    vk = np.random.random((N,4,4))+np.random.random((N,4,4))*1j
    wk = np.random.random((N,4,4))+np.random.random((N,4,4))*1j

    bv = np.zeros((N,4,4), dtype=np.complex)
    bw = np.zeros((N,4,4), dtype=np.complex)
    vk0 = np.zeros((N,4,4), dtype=np.complex)
    wk0 = np.zeros((N,4,4), dtype=np.complex)
    cv = np.zeros((N,4,4), dtype=np.complex)
    cw = np.zeros((N,4,4), dtype=np.complex)

    vk0 = SD.forward(vk, vk0)
    vk  = SD.backward(vk0, vk)
    vk0 = SD.forward(vk, vk0)
    wk0 = SD.forward(wk, wk0)
    wk  = SD.backward(wk0, wk)
    wk0 = SD.forward(wk, wk0)

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
    A = matrices.SBBmat(k)
    B = matrices.BBBmat(k, SB.quad)
    C = matrices.ABBmat(k)

    AA = -A.diags() + C.diags() + B.diags()
    f_hat = np.zeros(M)
    f_hat = SB.scalar_product(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[:-4] = la.spsolve(AA, f_hat[:-4])
    u1 = np.zeros(M)
    u1 = SB.backward(u_hat, u1)
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
    u_hat = SD.forward(uj, u_hat)
    uj = SD.backward(u_hat, uj)

    A = matrices.ADDmat(np.arange(M).astype(np.float))
    B = matrices.BDDmat(np.arange(M).astype(np.float), SD.quad)
    AB = HelmholtzCoeff(np.arange(M).astype(np.float), 1, kx**2, SD.quad)
    s = SD.slice(M)

    u1 = np.zeros(M)
    u1 = SD.forward(uj, u1)
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
