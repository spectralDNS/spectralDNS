from spectralDNS import config, get_solver, solve
from numpy import array, sum, pi, sin, cos, float64
import sys

def initialize(solver, **context):
    if config.params.solver == 'NS':
        initialize1(solver, **context)
    
    else:
        initialize2(solver, **context)
    config.params.t = 0.0
    config.params.tstep = 0
        
def initialize1(solver, U, U_hat, X, FFT, **context):
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0 
    for i in range(3):
        U_hat[i] = FFT.fftn(U[i], U_hat[i])
        
def initialize2(solver, U, W_hat, X, FFT, work, K, **context):
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0         
    F_tmp = work[(W_hat, 0)]
    for i in range(3):
        F_tmp[i] = FFT.fftn(U[i], F_tmp[i])
    W_hat = solver.cross2(W_hat, K, F_tmp)

def regression_test(context):
    params = config.params
    solver = config.solver
    dx, L = params.dx, params.L
    U = solver.get_velocity(**context)
    curl = solver.get_curl(**context)
    vol = dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2
    w = solver.comm.reduce(sum(curl.astype(float64)*curl.astype(float64))*vol)
    k = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))*vol) # Compute energy with double precision
    if solver.rank == 0:
        assert round(k - 0.124953117517, params.ntol) == 0
        assert round(w - 0.375249930801, params.ntol) == 0

if __name__ == "__main__":
    config.update(
        {
        'nu': 0.000625,             # Viscosity
        'dt': 0.01,                 # Time step
        'T': 0.1,                   # End time
        'L': [2*pi, 2*pi, 2*pi],
        'M': [4, 4, 4]
        }
    )
    for sol in ['NS', 'VV']:
        solver = get_solver(regression_test=regression_test, 
                            parse_args=sys.argv[1:]+[sol])
        context = solver.get_context()
        initialize(solver, **context)
        solve(solver, context)

        config.params.make_profile = 1
        config.params.dealias = '3/2-rule'
        initialize(solver, **context)
        solve(solver, context)
        
        config.params.dealias = '2/3-rule'
        config.params.optimization = 'cython'
        initialize(solver, **context)
        solve(solver, context)

        config.params.write_result = 1
        config.params.checkpoint = 1
        config.params.dt = 0.01
        config.params.t = 0.0
        config.params.tstep = 0
        config.params.T = 0.04
        solver.regression_test = lambda c: None
        solve(solver, context)
