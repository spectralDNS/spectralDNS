import pytest
import numpy as np
from spectralDNS.shen import Matrices
from copy import deepcopy
import six

mats = filter(lambda f: f.endswith('mat'), vars(Matrices).keys())
formats = ('dia', 'cython', 'python', 'self')

N = 16
k = np.arange(N).astype(float)
a = np.random.random(N)
b = np.random.random((N, N, N))
c = np.zeros(N)
c1= np.zeros(N)
d = np.zeros((N, N, N))
d1 = np.zeros((N, N, N))

@pytest.mark.parametrize('mat', mats)
@pytest.mark.parametrize('quad', ('GC', 'GL'))
def test_mat(mat, quad):
    """Test that matrix equals one that is automatically created"""
    mat = eval(('.').join(('Matrices', mat)))
    m = mat(k, quad)
    m.test()

@pytest.mark.parametrize('mat', mats)
@pytest.mark.parametrize('format', formats)
@pytest.mark.parametrize('quad', ('GC', 'GL'))
def test_matvec(mat, format, quad):
    """Test matrix-vector product"""
    global c, c1, d, d1
    mat = eval(('.').join(('Matrices', mat)))
    m = mat(k, quad)
    c = m.matvec(a, c, format='csr')
    c1 = m.matvec(a, c1, format=format)
    assert np.allclose(c, c1)

    d = m.matvec(b, d, format='csr')
    d1 = m.matvec(b, d1, format=format)

@pytest.mark.parametrize('mat', mats)
@pytest.mark.parametrize('quad', ('GC', 'GL'))
def test_imul(mat, quad):
    mat = eval(('.').join(('Matrices', mat)))
    m = mat(k, quad)
    mc = deepcopy(dict(m))
    m *= 2
    for key, val in six.iteritems(m):
        assert np.allclose(val, mc[key]*2)

#test_imul('BDNmat', 'GL')

@pytest.mark.parametrize('mat', mats)
@pytest.mark.parametrize('quad', ('GC', 'GL'))
def test_mul(mat, quad):
    mat = eval(('.').join(('Matrices', mat)))
    m = mat(k, quad)
    mc = 2.*m
    for key, val in six.iteritems(mc):
        assert np.allclose(val, m[key]*2.)

@pytest.mark.parametrize('mat', mats)
@pytest.mark.parametrize('quad', ('GC', 'GL'))
def test_rmul(mat, quad):
    mat = eval(('.').join(('Matrices', mat)))
    m = mat(k, quad)
    mc = m*2.
    for key, val in six.iteritems(mc):
        assert np.allclose(val, m[key]*2.)

@pytest.mark.parametrize('mat', mats)
@pytest.mark.parametrize('quad', ('GC', 'GL'))
def test_div(mat, quad):
    mat = eval(('.').join(('Matrices', mat)))
    m = mat(k, quad)
    mc = m/2.
    for key, val in six.iteritems(mc):
        assert np.allclose(val, m[key]/2.)

@pytest.mark.parametrize('mat', mats)
@pytest.mark.parametrize('quad', ('GC', 'GL'))
def test_add(mat, quad):
    mat = eval(('.').join(('Matrices', mat)))
    m = mat(k, quad)
    mc = m + m
    for key, val in six.iteritems(mc):
        assert np.allclose(val, m[key]*2)

@pytest.mark.parametrize('mat', mats)
@pytest.mark.parametrize('quad', ('GC', 'GL'))
def test_iadd(mat, quad):
    mat = eval(('.').join(('Matrices', mat)))
    m = mat(k, quad)
    mc = deepcopy(m)
    m += mc
    for key, val in six.iteritems(m):
        assert np.allclose(val, mc[key]*2)

@pytest.mark.parametrize('mat', mats)
@pytest.mark.parametrize('quad', ('GC', 'GL'))
def test_isub(mat, quad):
    mat = eval(('.').join(('Matrices', mat)))
    m = mat(k, quad)
    mc = deepcopy(m)
    m -= mc
    for key, val in six.iteritems(m):
        assert np.allclose(val, 0)

@pytest.mark.parametrize('mat', mats)
@pytest.mark.parametrize('quad', ('GC', 'GL'))
def test_sub(mat, quad):
    mat = eval(('.').join(('Matrices', mat)))
    m = mat(k, quad)
    mc = m - m
    for key, val in six.iteritems(mc):
        assert np.allclose(val, 0)
