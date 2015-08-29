from cbcdns import config, get_solver
import matplotlib.pyplot as plt
from numpy import array, pi
from numpy.linalg import norm

def initialize(config, **kw):
    if config.solver == 'NS':
        initialize1(**kw)
    
    else:
        initialize2(**kw)
        
def initialize1(U, U_hat, X, sin, cos, fftn_mpi, **kw):    
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0 
    for i in range(3):
        U_hat[i] = fftn_mpi(U[i], U_hat[i])
        
def initialize2(U, W, W_hat, X, sin, cos, fftn_mpi, ifftn_mpi, F_tmp, 
                cross2, K, **kw):
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0         
    for i in range(3):
        F_tmp[i] = fftn_mpi(U[i], F_tmp[i])

    W_hat[:] = cross2(W_hat, K, F_tmp)
    for i in range(3):
        W[i] = ifftn_mpi(W_hat[i], W[i])        

k = []
w = []
def update(t, tstep, dt, comm, rank, P, P_hat, U, curl, float64, dx, L, sum, 
           hdf5file, ifftn_mpi, **kw):
    global k, w
    if tstep % config.write_result == 0 or tstep % config.write_yz_slice[1] == 0:
        P[:] = ifftn_mpi(P_hat*1j, P)
        hdf5file.write(tstep)

    if tstep % config.compute_energy == 0:
        kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2) # Compute energy with double precision
        if config.solver == 'NS':
            ww = comm.reduce(sum(curl.astype(float64)*curl.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
        elif config.solver == 'VV':
            ww = comm.reduce(sum(kw['W'].astype(float64)*kw['W'].astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
        if rank == 0:
            k.append(kk)
            w.append(ww)
            print t, float(kk), float(ww)

def regression_test(t, tstep, comm, U, curl, float64, dx, L, sum, rank, **kw):
    k = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2) # Compute energy with double precision
    if config.solver == 'NS':
        w = comm.reduce(sum(curl.astype(float64)*curl.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
    elif config.solver == 'VV':
        w = comm.reduce(sum(kw['W'].astype(float64)*kw['W'].astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
    if rank == 0:
        assert round(k - 0.124953117517, 7) == 0
        assert round(w - 0.375249930801, 7) == 0

if __name__ == "__main__":
    config.update(
        {
        'nu': 0.000625,             # Viscosity
        'dt': 0.01,                 # Time step
        'T': 0.1,                   # End time
        'L': [2*pi, 2*pi, 2*pi],
        'M': [5, 5, 5]
        }
    )
    config.parser.add_argument("--compute_energy", type=int, default=2)
    #solver = get_solver(update=update, regression_test=regression_test)    
    solver = get_solver(update=update)
    solver.hdf5file.fname = "NS7.h5"
    solver.hdf5file.components["W0"] = solver.curl[0]
    solver.hdf5file.components["W1"] = solver.curl[1]
    solver.hdf5file.components["W2"] = solver.curl[2]
    initialize(**vars(solver))
    solver.solve()
