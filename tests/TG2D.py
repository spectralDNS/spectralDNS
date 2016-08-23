from spectralDNS import config, get_solver, solve
from numpy import pi, sin, cos, exp, zeros, float64, sum

def initialize(U, U_hat, X, FFT, **context):
    U[0] = sin(X[0])*cos(X[1])
    U[1] = -sin(X[1])*cos(X[0])
    for i in range(2):
        U_hat[i] = FFT.fft2(U[i], U_hat[i])
    config.params.t = 0
    config.params.tstep = 0

def regression_test(context):
    cx = context
    params = config.params
    solver = config.solver
    dx, L = params.dx, params.L
    U = solver.get_velocity(**context)
    curl = solver.get_curl(**context)
    k = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)
    U[0] = -sin(cx.X[1])*cos(cx.X[0])*exp(-2*params.nu*params.t)
    U[1] = sin(cx.X[0])*cos(cx.X[1])*exp(-2*params.nu*params.t)    
    ke = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)    
    if solver.rank == 0:
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
    context = solver.setup()
    initialize(**context)
    solve(solver, context)

    config.params.dealias = '3/2-rule'
    initialize(**context)
    solve(solver, context)
    
    config.params.dealias = '2/3-rule'
    config.params.optimization = 'cython'
    initialize(**context)
    solve(solver, context)
    
    config.params.write_result = 1
    config.params.checkpoint = 1
    config.params.dt = 0.01
    config.params.t = 0.0
    config.params.tstep = 0    
    config.params.T = 0.04
    solver.regression_test = lambda c: None
    initialize(**context)
    solve(solver, context)
    
    
