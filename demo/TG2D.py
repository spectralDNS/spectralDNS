from cbcdns import config, get_solver
from numpy import pi, sin, cos, exp, zeros
import matplotlib.pyplot as plt

def initialize(U, U_hat, X, sin, cos, fft2_mpi, **kw):    
    U[0] = sin(X[0])*cos(X[1])
    U[1] = -sin(X[1])*cos(X[0])
    for i in range(2):
        U_hat[i] = fft2_mpi(U[i], U_hat[i])

im = None
def update(t, tstep, N, U_hat, curl, X, nu, ifft2_mpi, K, P, P_hat, hdf5file, **kw):
    global im
    # initialize plot
    if tstep == 1:
        im = plt.imshow(zeros((N[0], N[1])))
        plt.colorbar(im)
        plt.draw()
        
    if tstep % config.write_result == 0:
        P = ifft2_mpi(P_hat*1j, P)
        hdf5file.write(tstep)

    if tstep % config.plot_result == 0 and config.plot_result > 0:
        curl = ifft2_mpi(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        im.set_data(curl[:, :])
        im.autoscale()
        plt.pause(1e-6)
        
def regression_test(t, tstep, comm, U, curl, float64, dx, L, sum, rank, X, nu, **kw):
    k = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)
    U[0] = -sin(X[1])*cos(X[0])*exp(-2*nu*t)
    U[1] = sin(X[0])*cos(X[1])*exp(-2*nu*t)    
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
      'M': [6, 6]}
    )

    config.parser.add_argument("--plot_result", type=int, default=10) # required to allow overloading through commandline
    solver = get_solver(update=update, regression_test=regression_test)
    solver.hdf5file.components["curl"] = solver.curl
    initialize(**vars(solver))
    solver.solve()
