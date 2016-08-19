"""
2D test case with three vortices
"""
from spectralDNS import config, get_solver, solve
from numpy import zeros, exp, pi, loadtxt, allclose
import matplotlib.pyplot as plt

def initialize(U, X, U_hat, FFT, K_over_K2, **context):
    w =     exp(-((X[0]-pi)**2+(X[1]-pi+pi/4)**2)/(0.2)) \
       +    exp(-((X[0]-pi)**2+(X[1]-pi-pi/4)**2)/(0.2)) \
       -0.5*exp(-((X[0]-pi-pi/4)**2+(X[1]-pi-pi/4)**2)/(0.4))
    w_hat = U_hat[0].copy()
    w_hat = FFT.fft2(w, w_hat)
    U[0] = FFT.ifft2(1j*K_over_K2[1]*w_hat, U[0])
    U[1] = FFT.ifft2(-1j*K_over_K2[0]*w_hat, U[1])
    U_hat[0] = FFT.fft2(U[0], U_hat[0])
    U_hat[1] = FFT.fft2(U[1], U_hat[1])
    
def regression_test(context):
    if config.solver.num_processes > 1:
        return True
    U_ref = loadtxt('vortices.txt')
    assert allclose(context.U[0], U_ref)

im = None
def update(context):
    global im
    params = config.params
    solver = config.solver
    
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
        
if __name__ == '__main__':
    config.update(
    {
      'nu': 0.001,
      'dt': 0.005,
      'T': 50,
      'write_result': 100,
      'M': [6, 6]}, 'doublyperiodic'
    )

    config.doublyperiodic.add_argument('--plot_result', type=int, default=10) # required to allow overloading through commandline
    solver = get_solver(update=update, regression_test=regression_test, mesh='doublyperiodic')
    context = solver.setup()
    context.hdf5file.components['curl'] = context.curl
    def update_components(**context):
        """Overload default because we want to store the curl as well"""
        U = solver.get_velocity(**context)
        P = solver.get_pressure(**context)
        curl = solver.get_curl(**context)
            
    context.hdf5file.update_components = update_components
    initialize(**context)
    solve(solver, context)
