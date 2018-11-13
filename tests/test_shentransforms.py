import pytest
from mpi4py import MPI
from sympy import Symbol, sin, pi, lambdify
import numpy as np
import scipy.sparse.linalg as la
from spectralDNS.shen.Matrices import HelmholtzCoeff
from spectralDNS.shen import LUsolve
from shenfun.spectralbase import inner_product
from shenfun.chebyshev.bases import Basis, ShenDirichletBasis, \
    ShenNeumannBasis, ShenBiharmonicBasis

comm = MPI.COMM_WORLD

N = 32
x = Symbol("x")

Basis = (Basis, ShenDirichletBasis, ShenNeumannBasis,
         ShenBiharmonicBasis)
quads = ('GC', 'GL')


def test_Mult_Div():

    SD = ShenDirichletBasis(N, "GC")
    SN = ShenNeumannBasis(N, "GC")
    SD.plan(N, 0, np.complex, {})
    SN.plan(N, 0, np.complex, {})

    Cm = inner_product((SN, 0), (SD, 1))
    Bm = inner_product((SN, 0), (SD, 0))

    uk = np.random.randn((N))+np.random.randn((N))*1j
    vk = np.random.randn((N))+np.random.randn((N))*1j
    wk = np.random.randn((N))+np.random.randn((N))*1j

    b = np.zeros(N, dtype=np.complex)
    uk0 = np.zeros(N, dtype=np.complex)
    vk0 = np.zeros(N, dtype=np.complex)
    wk0 = np.zeros(N, dtype=np.complex)

    uk0 = SD.forward(uk, uk0)
    uk = SD.backward(uk0, uk)
    uk0 = SD.forward(uk, uk0)
    vk0 = SD.forward(vk, vk0)
    vk = SD.backward(vk0, vk)
    vk0 = SD.forward(vk, vk0)
    wk0 = SD.forward(wk, wk0)
    wk = SD.backward(wk0, wk)
    wk0 = SD.forward(wk, wk0)

    LUsolve.Mult_Div_1D(N, 7, 7, uk0[:N-2], vk0[:N-2], wk0[:N-2], b[1:N-2])

    uu = np.zeros_like(uk0)
    v0 = np.zeros_like(vk0)
    w0 = np.zeros_like(wk0)
    uu = Cm.matvec(uk0, uu)
    uu += 1j*7*Bm.matvec(vk0, v0) + 1j*7*Bm.matvec(wk0, w0)

    #from IPython import embed; embed()
    assert np.allclose(uu, b)

    uk0 = uk0.repeat(4*4).reshape((N, 4, 4)) + 1j*uk0.repeat(4*4).reshape((N, 4, 4))
    vk0 = vk0.repeat(4*4).reshape((N, 4, 4)) + 1j*vk0.repeat(4*4).reshape((N, 4, 4))
    wk0 = wk0.repeat(4*4).reshape((N, 4, 4)) + 1j*wk0.repeat(4*4).reshape((N, 4, 4))
    b = np.zeros((N, 4, 4), dtype=np.complex)
    m = np.zeros((4, 4))+7
    n = np.zeros((4, 4))+7
    LUsolve.Mult_Div_3D(N, m, n, uk0[:N-2], vk0[:N-2], wk0[:N-2], b[1:N-2])

    uu = np.zeros_like(uk0)
    v0 = np.zeros_like(vk0)
    w0 = np.zeros_like(wk0)
    uu = Cm.matvec(uk0, uu)
    uu += 1j*7*Bm.matvec(vk0, v0) + 1j*7*Bm.matvec(wk0, w0)

    assert np.allclose(uu, b)

#test_Mult_Div()


@pytest.mark.parametrize('quad', quads)
def test_Mult_CTD(quad):
    SD = ShenDirichletBasis(N, quad=quad)
    SD.plan(N, 0, np.complex, {})
    C = inner_product((SD.CT, 0), (SD, 1))
    B = inner_product((SD.CT, 0), (SD.CT, 0))

    vk = np.random.randn((N))+np.random.randn((N))*1j
    wk = np.random.randn((N))+np.random.randn((N))*1j

    bv = np.zeros(N, dtype=np.complex)
    bw = np.zeros(N, dtype=np.complex)
    vk0 = np.zeros(N, dtype=np.complex)
    wk0 = np.zeros(N, dtype=np.complex)
    cv = np.zeros(N, dtype=np.complex)
    cw = np.zeros(N, dtype=np.complex)

    vk0 = SD.forward(vk, vk0)
    vk = SD.backward(vk0, vk)
    vk0 = SD.forward(vk, vk0)
    wk0 = SD.forward(wk, wk0)
    wk = SD.backward(wk0, wk)
    wk0 = SD.forward(wk, wk0)

    LUsolve.Mult_CTD_1D(N, vk0, wk0, bv, bw)

    cv = np.zeros_like(vk0)
    cw = np.zeros_like(wk0)
    cv = C.matvec(vk0, cv)
    cw = C.matvec(wk0, cw)
    cv /= B[0]
    cw /= B[0]

    assert np.allclose(cv, bv)
    assert np.allclose(cw, bw)

#test_Mult_CTD("GC")

@pytest.mark.parametrize('quad', quads)
def test_Mult_CTD_3D(quad):
    SD = ShenDirichletBasis(N, quad=quad)
    SD.plan((N, 4, 4), 0, np.complex, {})

    C = inner_product((SD.CT, 0), (SD, 1))
    B = inner_product((SD.CT, 0), (SD.CT, 0))

    vk = np.random.random((N, 4, 4))+np.random.random((N, 4, 4))*1j
    wk = np.random.random((N, 4, 4))+np.random.random((N, 4, 4))*1j

    bv = np.zeros((N, 4, 4), dtype=np.complex)
    bw = np.zeros((N, 4, 4), dtype=np.complex)
    vk0 = np.zeros((N, 4, 4), dtype=np.complex)
    wk0 = np.zeros((N, 4, 4), dtype=np.complex)
    cv = np.zeros((N, 4, 4), dtype=np.complex)
    cw = np.zeros((N, 4, 4), dtype=np.complex)

    vk0 = SD.forward(vk, vk0)
    vk = SD.backward(vk0, vk)
    vk0 = SD.forward(vk, vk0)
    wk0 = SD.forward(wk, wk0)
    wk = SD.backward(wk0, wk)
    wk0 = SD.forward(wk, wk0)

    LUsolve.Mult_CTD_3D_ptr(N, vk0, wk0, bv, bw, 0)

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
    M = 128
    SB = ShenBiharmonicBasis(M, quad=quad)
    u = sin(6*pi*x)**2
    a = 1.0
    b = 1.0
    f = -u.diff(x, 4) + a*u.diff(x, 2) + b*u

    ul = lambdify(x, u, 'numpy')
    fl = lambdify(x, f, 'numpy')
    points, _ = SB.points_and_weights(M)
    uj = ul(points)
    fj = fl(points)

    A = inner_product((SB, 0), (SB, 4))
    B = inner_product((SB, 0), (SB, 0))
    C = inner_product((SB, 0), (SB, 2))

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
    M = 2*N
    SD = ShenDirichletBasis(M, quad=quad)
    kx = 11
    uj = np.random.randn(M)
    u_hat = np.zeros(M)
    u_hat = SD.forward(uj, u_hat)
    uj = SD.backward(u_hat, uj)

    B = inner_product((SD, 0), (SD, 0))
    A = inner_product((SD, 0), (SD, 2))

    AB = HelmholtzCoeff(M, 1, kx**2, SD.quad)

    u1 = np.zeros(M)
    u1 = SD.forward(uj, u1)
    c0 = np.zeros_like(u1)
    c1 = np.zeros_like(u1)
    c = A.matvec(u1, c0)+kx**2*B.matvec(u1, c1)

    b = np.zeros(M)
    #LUsolve.Mult_Helmholtz_1D(M, SD.quad=="GL", 1, kx**2, u1, b)
    b = AB.matvec(u1, b)
    #from IPython import embed; embed()
    assert np.allclose(c, b)

    b = np.zeros((M, 4, 4), dtype=np.complex)
    u1 = u1.repeat(16).reshape((M, 4, 4)) +1j*u1.repeat(16).reshape((M, 4, 4))
    kx = np.zeros((1, 4, 4))+kx
    #LUsolve.Mult_Helmholtz_3D_complex(M, SD.quad=="GL", 1.0, kx**2, u1, b)
    AB = HelmholtzCoeff(M, 1, kx**2, SD.quad)
    b = AB.matvec(u1, b)

    assert np.linalg.norm(b[:, 2, 2].real - c)/(M*16) < 1e-12
    assert np.linalg.norm(b[:, 2, 2].imag - c)/(M*16) < 1e-12

#test_Helmholtz_matvec("GL")
#test_ADDmat(ShenNeumannBasis("GL"))
#test_Helmholtz2(ShenDirichletBasis("GL"))
#test_Mult_CTD(ShenDirichletBasis("GL"))
#test_CDDmat(ShenDirichletBasis("GL"))
