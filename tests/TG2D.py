from spectralDNS import config, get_solver
from numpy import pi, sin, cos, exp, zeros, float64, sum

def initialize(U, U_hat, X, FFT, **kw):    
    U[0] = sin(X[0])*cos(X[1])
    U[1] = -sin(X[1])*cos(X[0])
    for i in range(2):
        U_hat[i] = FFT.fft2(U[i], U_hat[i])

def regression_test(comm, U, rank, X, params, **kw):
    dx, L = params.dx, params.L
    k = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)
    U[0] = -sin(X[1])*cos(X[0])*exp(-2*params.nu*params.t)
    U[1] = sin(X[0])*cos(X[1])*exp(-2*params.nu*params.t)    
    ke = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)    
    if rank == 0:
        assert round(k - ke, 7) == 0

if __name__ == '__main__':
    config.update(
    {
      'nu': 0.01,
      'dt': 0.05,
      'T': 10,
      'M': [6, 6]}, 'doublyperiodic'
    )

    solver = get_solver(regression_test=regression_test, mesh='doublyperiodic')
    initialize(**vars(solver))
    solver.solve()

    config.params.dealias = '3/2-rule'
    initialize(**vars(solver))
    solver.solve()
    
    config.params.dealias = '2/3-rule'
    config.params.optimization = 'cython'
    initialize(**vars(solver))
    solver.solve()    
    
    config.params.write_result = 1
    config.params.checkpoint = 1
    config.dt = 0.01
    config.T = 0.04
    solver.regression_test = lambda **kwargs: None
    solver.solve()    
    
