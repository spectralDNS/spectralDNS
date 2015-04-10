"""
2D test case with three vortices
"""
from pylab import plt, zeros

plot_result = 10

def initialize(U, X, U_hat, exp, pi, irfft2_mpi, rfft2_mpi, K_over_K2, **kwargs):
    w =     exp(-((X[0]-pi)**2+(X[1]-pi+pi/4)**2)/(0.2)) \
       +    exp(-((X[0]-pi)**2+(X[1]-pi-pi/4)**2)/(0.2)) \
       -0.5*exp(-((X[0]-pi-pi/4)**2+(X[1]-pi-pi/4)**2)/(0.4))
    w_hat = U_hat[0].copy()
    w_hat = rfft2_mpi(w, w_hat)
    U[0] = irfft2_mpi(1j*K_over_K2[1]*w_hat, U[0])
    U[1] = irfft2_mpi(-1j*K_over_K2[0]*w_hat, U[1])
    U_hat[0] = rfft2_mpi(U[0], U_hat[0])
    U_hat[1] = rfft2_mpi(U[1], U_hat[1])

def regression_test(U, num_processes, loadtxt, allclose, **kwargs):
    if num_processes > 1:
        return True
    U_ref = loadtxt('vortices.txt')
    assert allclose(U[0], U_ref)

im = None
def update(t, tstep, N, curl, U_hat, irfft2_mpi, K, **kw):
    global im
    # initialize plot
    if tstep == 1:
        im = plt.imshow(zeros((N, N)))
        plt.colorbar(im)
        plt.draw()
        
    if tstep % plot_result == 0 and plot_result > 0:
        curl = irfft2_mpi(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        im.set_data(curl[:, :])
        im.autoscale()
        plt.pause(1e-6)
        
if __name__ == '__main__':
    # Set some (any) problem dependent parameters by overloading default parameters
    import config
    config.update(
    {
      'solver': 'NS2D',
      'nu': 0.001,
      'dt': 0.005,
      'T': 50,
      'debug': False}
    )

    from spectral import solver
    initialize(**vars(solver))
    solver.update = update
    solver.regression_test = regression_test
    solver.solve()

