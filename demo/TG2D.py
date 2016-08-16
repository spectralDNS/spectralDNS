from spectralDNS import config, get_solver
from numpy import pi, sin, cos, exp, zeros, sum, sin, cos, float64
import matplotlib.pyplot as plt

def initialize(U, U_hat, X, FFT, **kw):    
    U[0] = sin(X[0])*cos(X[1])
    U[1] = -sin(X[1])*cos(X[0])
    for i in range(2):
        U_hat[i] = FFT.fft2(U[i], U_hat[i])

im = None
def update(U_hat, curl, X, FFT, K, P, P_hat, hdf5file, params, **kw):
    global im
    # initialize plot
    if params.tstep == 1:
        im = plt.imshow(zeros((params.N[0], params.N[1])))
        plt.colorbar(im)
        plt.draw()

    if params.tstep % params.plot_result == 0 and params.plot_result > 0:
        curl = FFT.ifft2(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        im.set_data(curl[:, :])
        im.autoscale()
        plt.pause(1e-6)
        
def regression_test(comm, U, rank, X, U_hat, params, **kw):
    dx, L = params.dx, params.L
    k = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)
    U[0] = -sin(X[1])*cos(X[0])*exp(-2*params.nu*params.t)
    U[1] = sin(X[0])*cos(X[1])*exp(-2*params.nu*params.t)    
    ke = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)    
    if rank == 0:
        print "Error", k-ke
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

    config.doublyperiodic.add_argument("--plot_result", type=int, default=10) # required to allow overloading through commandline
    solver = get_solver(update=update, regression_test=regression_test, mesh="doublyperiodic")
    initialize(**vars(solver))
    solver.solve()
