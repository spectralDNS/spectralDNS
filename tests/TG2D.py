from spectralDNS import config, get_solver
from numpy import pi, sin, cos, exp, zeros

def initialize(U, U_hat, X, sin, cos, FFT, **kw):    
    U[0] = sin(X[0])*cos(X[1])
    U[1] = -sin(X[1])*cos(X[0])
    for i in range(2):
        U_hat[i] = FFT.fft2(U[i], U_hat[i])

def update(FFT, P, P_hat, hdf5file, **kw):
    if hdf5file.check_if_write(config.tstep):
        P = FFT.ifft2(P_hat*1j, P)
        hdf5file.write(config.tstep)
        
def regression_test(comm, U, float64, dx, L, sum, rank, X, **kw):
    k = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)
    U[0] = -sin(X[1])*cos(X[0])*exp(-2*config.nu*config.t)
    U[1] = sin(X[0])*cos(X[1])*exp(-2*config.nu*config.t)    
    ke = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)    
    if rank == 0:
        assert round(k - ke, 7) == 0

if __name__ == '__main__':
    config.update(
    {
      'nu': 0.01,
      'dt': 0.05,
      'T': 10,
      'write_result': 100,
      'M': [6, 6]}, 'doublyperiodic'
    )

    solver = get_solver(update=update, regression_test=regression_test, mesh='doublyperiodic')
    initialize(**vars(solver))
    solver.solve()
