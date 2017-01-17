import pytest
import numpy as np
from spectralDNS.shen import Matrices

mats = filter(lambda f: f.endswith('mat'), vars(Matrices).keys())

@pytest.fixture(params=mats)
def mat(request):
    return eval(('.').join(('Matrices', request.param)))

N = 16
k = np.arange(N).astype(float)
a = np.random.random(N)
b = np.random.random((N, N, N))
c = np.zeros(N)
c1= np.zeros(N)
d = np.zeros((N, N, N))
d1 = np.zeros((N, N, N))

def test_mat(mat):
    """Test that matrix equals one that is automatically created"""
    m = mat(k, 'GC')
    m.test()
    m = mat(k, 'GL')
    m.test()

def test_matvec(mat):
    """Test matrix-vector product"""
    global c, c1, d, d1
    m = mat(k, 'GC')
    c = m.matvec(a, c, format='csr')
    c1 = m.matvec(a, c1, format='python')
    assert np.allclose(c, c1)
    c1 = m.matvec(a, c1)
    assert np.allclose(c, c1)
    d = m.matvec(b, d, format='csr')
    d1 = m.matvec(b, d1, format='python')
    assert np.allclose(d, d1)
    d1 = m.matvec(b, d1)
    assert np.allclose(d, d1)


#test_matvec(Matrices.BBDmat)
