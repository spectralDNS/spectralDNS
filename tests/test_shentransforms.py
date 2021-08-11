import pytest
from mpi4py import MPI
from sympy import Symbol
import numpy as np
from spectralDNS.shen import LUsolve
from shenfun.spectralbase import inner_product
from shenfun import FunctionSpace, TensorProductSpace, Function, Array

comm = MPI.COMM_WORLD

N = 32
x = Symbol("x")

quads = ('GC', 'GL')


def test_Mult_Div():

    #SD = ShenDirichlet(N, "GC")
    #SN = ShenNeumann(N, "GC")
    #SD.plan(N, 0, np.complex, {})
    #SN.plan(N, 0, np.complex, {})
    SD = FunctionSpace(N, 'C', bc=(0, 0), dtype='D')
    SN = FunctionSpace(N, 'C', basis='ShenNeumann', dtype='D')

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

    assert np.allclose(uu[1:], b[1:])

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

    assert np.allclose(uu[1:], b[1:])

#test_Mult_Div()

@pytest.mark.parametrize('quad', quads)
def test_Mult_CTD_3D(quad):
    SD = FunctionSpace(N, 'C', bc=(0, 0))
    F0 = FunctionSpace(4, 'F', dtype='D')
    F1 = FunctionSpace(4, 'F', dtype='d')
    T = TensorProductSpace(comm, (SD, F0, F1))

    TO = T.get_orthogonal()
    CT = TO.bases[0]

    C = inner_product((CT, 0), (SD, 1))
    B = inner_product((CT, 0), (CT, 0))

    vk = Array(T)
    wk = Array(T)
    vk[:] = np.random.random(vk.shape)
    wk[:] = np.random.random(vk.shape)

    bv = Function(T)
    bw = Function(T)

    vk0 = vk.forward()
    vk = vk0.backward()
    wk0 = wk.forward()
    wk = wk0.backward()

    LUsolve.Mult_CTD_3D_ptr(N, vk0, wk0, bv, bw, 0)

    cv = np.zeros_like(vk0)
    cw = np.zeros_like(wk0)
    cv = C.matvec(vk0, cv)
    cw = C.matvec(wk0, cw)
    cv /= B[0].repeat(np.array(bv.shape[1:]).prod()).reshape(bv.shape)
    cw /= B[0].repeat(np.array(bv.shape[1:]).prod()).reshape(bv.shape)

    assert np.allclose(cv, bv)
    assert np.allclose(cw, bw)

test_Mult_Div()
