"""
2D test case with three vortices
"""
from numpy import zeros
import matplotlib.pyplot as plt

def initialize(U, X, U_hat, exp, pi, ifft2_mpi, fft2_mpi, K_over_K2, **kwargs):
    w =     exp(-((X[0]-pi)**2+(X[1]-pi+pi/4)**2)/(0.2)) \
       +    exp(-((X[0]-pi)**2+(X[1]-pi-pi/4)**2)/(0.2)) \
       -0.5*exp(-((X[0]-pi-pi/4)**2+(X[1]-pi-pi/4)**2)/(0.4))
    w_hat = U_hat[0].copy()
    w_hat = fft2_mpi(w, w_hat)
    U[0] = ifft2_mpi(1j*K_over_K2[1]*w_hat, U[0])
    U[1] = ifft2_mpi(-1j*K_over_K2[0]*w_hat, U[1])
    U_hat[0] = fft2_mpi(U[0], U_hat[0])
    U_hat[1] = fft2_mpi(U[1], U_hat[1])
    
def regression_test(U, num_processes, loadtxt, allclose, **kwargs):
    if num_processes > 1:
        return True
    U_ref = loadtxt('vortices.txt')
    assert allclose(U[0], U_ref)

im = None
def update(t, tstep, N, curl, U_hat, ifft2_mpi, K, P, P_hat, hdf5file, **kw):
    global im
    # initialize plot
    if tstep == 1:
        im = plt.imshow(zeros((N[0], N[1])))
        plt.colorbar(im)
        plt.draw()
        
    if tstep % config.write_result == 0:
        P = ifft2_mpi(P_hat*1j, P)
        curl = ifft2_mpi(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        hdf5file.write(tstep)   
        
    if tstep % config.plot_result == 0 and config.plot_result > 0:
        curl = ifft2_mpi(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        im.set_data(curl[:, :])
        im.autoscale()
        plt.pause(1e-6)
        
if __name__ == '__main__':
    from cbcdns import config, get_solver
    config.update(
    {
      'nu': 0.001,
      'dt': 0.005,
      'T': 50,
      'write_result': 100,
      'M': [5, 6]}
    )

    config.parser.add_argument("--plot_result", type=int, default=10) # required to allow overloading through commandline
    solver = get_solver()
    solver.hdf5file.components["curl"] = solver.curl
    initialize(**vars(solver))
    solver.update = update
    solver.regression_test = regression_test
    solver.solve()

