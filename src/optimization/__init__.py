import cython_single, cython_double
import weave_single, weave_double
from cython_double import transpose_Uc

def weaverhs(dU, U_hat, K2, K, P_hat, K_over_K2, nu):
    if dU.dtype == "complex64":
        weave_single.weaverhs(dU, U_hat, K2, K, P_hat, K_over_K2, nu)
    else:
        weave_double.weaverhs(dU, U_hat, K2, K, P_hat, K_over_K2, nu)
    return dU
        
def weavecross(a, b, c):
    if c.dtype == "float32":
        weave_single.weavecross(a, b, c)
    else:
        weave_double.weavecross(a, b, c)
    return c
        
def weavecrossi(a, b, c):
    if c.dtype == "complex64":
        weave_single.weavecrossi(a, b, c)
    else:
        weave_double.weavecrossi(a, b, c)
    return c

def weavedealias(a, b):
    if a.dtype == "complex64":
        weave_single.weavedealias(a, b)
    else:
        weave_double.weavedealias(a, b)
    return a

def cythonrhs(dU, U_hat, K2, K, P_hat, K_over_K2, nu):
    if dU.dtype == "complex64":
        cython_single.cythonrhs(dU, U_hat, K2, K, P_hat, K_over_K2, nu)
    else:
        cython_double.cythonrhs(dU, U_hat, K2, K, P_hat, K_over_K2, nu)
    return dU
        
def cythoncross(a, b, c):
    if c.dtype == "float32":
        cython_single.cythoncross(a, b, c)
    else:
        cython_double.cythoncross(a, b, c)
    return c
        
def cythoncrossi(a, b, c):
    if c.dtype == "complex64":
        cython_single.cythoncrossi(a, b, c)
    else:
        cython_double.cythoncrossi(a, b, c)
    return c

def cythondealias(a, b):
    if a.dtype == "complex64":
        cython_single.cythondealias(a, b)
    else:
        cython_double.cythondealias(a, b)
    return a
