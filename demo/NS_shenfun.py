__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2017-09-26"
__copyright__ = "Copyright (C) 2017 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from time import time
import numpy as np
from mpi4py import MPI
from shenfun import *

nu = 0.000625
end_time = 0.1
dt = 0.01
comm = MPI.COMM_WORLD
N = (2**6, 2**6, 2**6)

K0 = FunctionSpace(N[0], 'F', dtype='D')
K1 = FunctionSpace(N[1], 'F', dtype='D')
K2 = FunctionSpace(N[2], 'F', dtype='d')
T = TensorProductSpace(comm, (K0, K1, K2), **{'planner_effort': 'FFTW_MEASURE'})
TV = VectorTensorProductSpace(T)

U = Array(TV)
U_hat = Function(TV)
P_hat = Function(T)
U_hat0 = Function(TV)
U_hat1 = Function(TV)
dU = Function(TV)
curl = Array(TV)
K = np.array(T.local_wavenumbers(True, True, True))
K2 = np.sum(K*K, 0, dtype=int)
K_over_K2 = K.astype(float) / np.where(K2 == 0, 1, K2).astype(float)
a = [1./6., 1./3., 1./3., 1./6.]
b = [0.5, 0.5, 1.]

def Cross(a, b, c):
    c[0] = T.forward(a[1]*b[2]-a[2]*b[1], c[0])
    c[1] = T.forward(a[2]*b[0]-a[0]*b[2], c[1])
    c[2] = T.forward(a[0]*b[1]-a[1]*b[0], c[2])
    return c

def Curl(a, c):
    c[2] = T.backward(1j*(K[0]*a[1]-K[1]*a[0]), c[2])
    c[1] = T.backward(1j*(K[2]*a[0]-K[0]*a[2]), c[1])
    c[0] = T.backward(1j*(K[1]*a[2]-K[2]*a[1]), c[0])
    return c

def ComputeRHS(dU, rk):
    if rk > 0:
        U[:] = TV.backward(U_hat, U)
    curl[:] = Curl(U_hat, curl)
    dU = Cross(U, curl, dU)
    P_hat[:] = np.sum(dU*K_over_K2, 0, out=P_hat)
    dU -= P_hat*K
    dU -= nu*K2*U_hat
    return dU

X = T.local_mesh(True)
U[0] = np.sin(X[0])*np.cos(X[1])*np.cos(X[2])
U[1] = -np.cos(X[0])*np.sin(X[1])*np.cos(X[2])
U[2] = 0
U_hat = TV.forward(U, U_hat)

t = 0.0
tstep = 0
t0 = time()
while t < end_time-1e-8:
    t += dt
    tstep += 1
    U_hat1[:] = U_hat0[:] = U_hat
    for rk in range(4):
        dU = ComputeRHS(dU, rk)
        if rk < 3:
            U_hat[:] = U_hat0 + b[rk]*dt*dU
        U_hat1 += a[rk]*dt*dU
    U_hat[:] = U_hat1
    U = TV.backward(U_hat, U)

k = comm.reduce(0.5*np.sum(U*U)/np.prod(np.array(N)))
if comm.Get_rank() == 0:
    print("Time = {}".format(time()-t0))
    assert np.round(k - 0.124953117517, 7) == 0
