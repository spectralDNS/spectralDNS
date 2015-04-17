from cbcdns import config, get_solver
import matplotlib.pyplot as plt
from numpy import zeros

def initialize(X, U, U_hat, exp, sin, cos, tanh, Np, N, pi, fft2_mpi, **kwargs):
    Um = 0.5*(config.U1 - config.U2)
    U[1] = config.A*sin(X[0])       
    U[0, :, :N/4] = config.U1 - Um*exp((X[1,:, :N/4] - 0.5*pi)/config.delta)
    U[0, :, N/4:N/2] = config.U2 + Um*exp(-1.0*(X[1, :, N/4:N/2] - 0.5*pi)/config.delta)
    U[0, :, N/2:3*N/4] = config.U2 + Um*exp((X[1, :, N/2:3*N/4] - 1.5*pi)/config.delta)
    U[0, :, 3*N/4:] = config.U1 - Um*exp(-1.0*(X[1, :, 3*N/4:] - 1.5*pi)/config.delta)
          
    for i in range(2):
        U_hat[i] = fft2_mpi(U[i], U_hat[i])

im, im2 = None, None
def update(t, tstep, comm, rank, N, L, dx, ifft2_mpi, U_hat, U, sum,
           P_hat, P, hdf5file, float64, K, curl, **kwargs):
    global im, im2
    if tstep % config.write_result == 0 or tstep % config.write_yz_slice[1] == 0:
        P = ifft2_mpi(P_hat*1j, P)
        hdf5file.write(tstep)           

    if tstep % config.compute_energy == 0:
        kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx*dx/L**2/2)
        if rank == 0:
            print tstep, kk
            
    if tstep == 1 and config.plot_result > 0:
        fig, ax = plt.subplots(1,1)
        fig.suptitle('U0', fontsize=20)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        im = ax.imshow(zeros((N, N)),cmap=plt.cm.bwr, extent=[0, L, 0, L])
        plt.colorbar(im)
        plt.draw() 

        fig2, ax2 = plt.subplots(1,1)
        fig2.suptitle('Vorticity', fontsize=20)   
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        im2 = ax2.imshow(zeros((N, N)),cmap=plt.cm.bwr, extent=[0, L, 0, L])
        plt.colorbar(im2)
        plt.draw()
        globals().update(dict(im=im, im2=im2))

    if tstep % config.plot_result == 0 and config.plot_result > 0:
        curl[:] = ifft2_mpi(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        #curl = ifft2_mpi(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        im.set_data(U[0, :, :].T)
        im.autoscale()
        plt.pause(1e-6)
        im2.set_data(curl[:,:].T)
        im2.autoscale()
        plt.pause(1e-6)
        if rank == 0:
            print tstep            

if __name__ == "__main__":
    config.update(
    {
    'nu': 1.0e-05,
    'dt': 0.007,
    'T': 25.0,
    'U1':-0.5,
    'U2':0.5,
    'l0': 0.001,    # Smoothing parameter
    'A': 0.01,      # Amplitude of perturbation
    'delta': 0.1,   # Width of perturbations
    'write_result': 500
    }
    )
    config.parser.add_argument("--plot_result", type=int, default=10) # required to allow overloading through commandline    
    config.parser.add_argument("--compute_energy", type=int, default=10)
    solver = get_solver(update)
    initialize(**vars(solver))
    solver.solve()
