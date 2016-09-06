from spectralDNS import config, get_solver, solve
from numpy import array, pi, sin, cos, float64, sum

def initialize(UB_hat, UB, U, B, X, FFT, **context):
    # Taylor-Green initialization
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0 
    B[0] = sin(X[0])*sin(X[1])*cos(X[2])
    B[1] = cos(X[0])*cos(X[1])*cos(X[2])
    B[2] = 0 
    for i in range(6):
        UB_hat[i] = FFT.fftn(UB[i], UB_hat[i])
    config.params.t = 0
    config.params.tstep = 0
        
def regression_test(context):
    params = config.params
    solver = config.solver
    dx, L = params.dx, params.L
    UB = solver.get_UB(**context)
    U, B = UB[:3], UB[3:]
    k = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2) # Compute energy with double precision
    b = solver.comm.reduce(sum(B.astype(float64)*B.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
    if solver.rank == 0:
        assert round(k - 0.124565408177, 7) == 0 
        assert round(b - 0.124637762143, 7) == 0

if __name__=='__main__':
    config.update(
        {
        'nu': 0.000625,             # Viscosity
        'dt': 0.01,                 # Time step
        'T': 0.1,                   # End time
        'eta': 0.01,
        'L': [2*pi, 4*pi, 6*pi],
        'M': [4, 5, 6],
        'convection': 'Divergence'
        }
    )
    solver = get_solver(regression_test=regression_test)
    context = solver.get_context()
    initialize(**context)
    solve(solver, context)
