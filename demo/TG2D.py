from spectralDNS import config, get_solver, solve
from numpy import pi, sin, cos, exp, zeros, sum, sin, cos, float64
import warnings

try:
    import matplotlib.pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")
    plt = None

def initialize(U, U_hat, X, FFT, **context):    
    U[0] = sin(X[0])*cos(X[1])
    U[1] = -sin(X[1])*cos(X[0])
    for i in range(2):
        U_hat[i] = FFT.fft2(U[i], U_hat[i])
    config.params.t = 0.0
    config.params.tstep = 0

im = None
def update(context):
    global im
    params = config.params
    solver = config.solver

    if not plt is None:
        # initialize plot
        if params.tstep == 1:
            im = plt.imshow(zeros((params.N[0], params.N[1])))
            plt.colorbar(im)
            plt.draw()

        if params.tstep % params.plot_result == 0 and params.plot_result > 0:
            curl = solver.get_curl(**context)
            im.set_data(curl[:, :])
            im.autoscale()
            plt.pause(1e-6)
        
def regression_test(context):
    params = config.params
    solver = config.solver
    dx, L = params.dx, params.L
    U = solver.get_velocity(**context)
    k = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)
    U[0] = -sin(context.X[1])*cos(context.X[0])*exp(-2*params.nu*params.t)
    U[1] = sin(context.X[0])*cos(context.X[1])*exp(-2*params.nu*params.t)
    ke = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)
    if solver.rank == 0:
        print("Error {}".format(k-ke))
        assert round(k - ke, params.ntol) == 0

if __name__ == '__main__':
    config.update(
    {
      'nu': 0.01,
      'dt': 0.05,
      'T': 10,
      'write_result': 100,
      'M': [6, 6]}, 'doublyperiodic'
    )

    config.doublyperiodic.add_argument("--plot_result", type=int, default=10) # required to allow overloading through commandline
    sol = get_solver(update=update, regression_test=regression_test, mesh="doublyperiodic")
    context = sol.get_context()
    initialize(**context)
    solve(sol, context)
