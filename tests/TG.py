from spectralDNS import config, get_solver
from numpy import array, pi

def initialize(config, **kw):
    if config.solver == 'NS':
        initialize1(**kw)
    
    else:
        initialize2(**kw)
        
def initialize1(U, U_hat, X, sin, cos, FFT, **kw):    
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0 
    for i in range(3):
        U_hat[i] = FFT.fftn(U[i], U_hat[i])
        
def initialize2(U, W, W_hat, X, sin, cos, FFT, work,
                cross2, K, **kw):
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0         
    F_tmp = work[(W_hat, 0)]
    for i in range(3):
        F_tmp[i] = FFT.fftn(U[i], F_tmp[i])

    W_hat[:] = cross2(W_hat, K, F_tmp)
    for i in range(3):
        W[i] = FFT.ifftn(W_hat[i], W[i])        

def update(t, tstep, P, P_hat, hdf5file, FFT, **kw):
    if hdf5file.check_if_write(tstep):
        P = FFT.ifftn(P_hat*1j, P)
        hdf5file.write(tstep)
    
def regression_test(t, tstep, comm, U_hat, U, curl, float64, dx, L, sum, rank, Curl, **kw):
    if config.solver == 'NS':
        curl[:] = Curl(U_hat, curl)
        w = comm.reduce(sum(curl.astype(float64)*curl.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
    elif config.solver == 'VV':
        U = Curl(kw['W_hat'], U)
        w = comm.reduce(sum(kw['W'].astype(float64)*kw['W'].astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)

    k = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2) # Compute energy with double precision
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
    solver = get_solver(update=update, regression_test=regression_test)
    initialize(**vars(solver))
    solver.solve()
