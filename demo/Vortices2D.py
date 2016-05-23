"""
2D test case with three vortices
"""
from spectralDNS import config, get_solver
from numpy import zeros
import matplotlib.pyplot as plt

def initialize(U, X, U_hat, exp, pi, FFT, K_over_K2, **kwargs):
    w =     exp(-((X[0]-pi)**2+(X[1]-pi+pi/4)**2)/(0.2)) \
       +    exp(-((X[0]-pi)**2+(X[1]-pi-pi/4)**2)/(0.2)) \
       -0.5*exp(-((X[0]-pi-pi/4)**2+(X[1]-pi-pi/4)**2)/(0.4))
    w_hat = U_hat[0].copy()
    w_hat = FFT.fft2(w, w_hat)
    U[0] = FFT.ifft2(1j*K_over_K2[1]*w_hat, U[0])
    U[1] = FFT.ifft2(-1j*K_over_K2[0]*w_hat, U[1])
    U_hat[0] = FFT.fft2(U[0], U_hat[0])
    U_hat[1] = FFT.fft2(U[1], U_hat[1])
    
def regression_test(U, num_processes, loadtxt, allclose, **kwargs):
    if num_processes > 1:
        return True
    U_ref = loadtxt('vortices.txt')
    assert allclose(U[0], U_ref)

im = None
def update(N, curl, U_hat, FFT, K, P, P_hat, hdf5file, **kw):
    global im
    # initialize plot
    if config.tstep == 1:
        im = plt.imshow(zeros((N[0], N[1])))
        plt.colorbar(im)
        plt.draw()
        
    if hdf5file.check_if_write(config.tstep):
        P = FFT.ifft2(P_hat*1j, P)
        curl = FFT.ifft2(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        hdf5file.write(config.tstep)   
        
    if config.tstep % config.plot_result == 0 and config.plot_result > 0:
        curl = FFT.ifft2(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
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
    solver.hdf5file.components['curl'] = solver.curl
    initialize(**vars(solver))
    solver.solve()

