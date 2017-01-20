import pytest
import numpy as np
from spectralDNS.shen import Matrices
from itertools import product

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

#test_matvec(Matrices.BBDmat)
