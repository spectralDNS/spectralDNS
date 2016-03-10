from spectralDNS import config, get_solver
import matplotlib.pyplot as plt
from numpy import zeros, exp

def initialize(X, U, Ur, Ur_hat, exp, sin, cos, tanh, rho, N, pi, FFT, float, **kwargs):

    Um = 0.5*(config.U1 - config.U2)
    U[1] = config.A*sin(2*X[0])
    #U[0, :, :N/4] = config.U1 - Um*exp((X[1,:, :N/4] - 0.5*pi)/config.delta) 
    #U[0, :, N/4:N/2] = config.U2 + Um*exp(-1.0*(X[1, :, N/4:N/2] - 0.5*pi)/config.delta) 
    #U[0, :, N/2:3*N/4] = config.U2 + Um*exp((X[1, :, N/2:3*N/4] - 1.5*pi)/config.delta) 
    #U[0, :, 3*N/4:] = config.U1 - Um*exp(-1.0*(X[1, :, 3*N/4:] - 1.5*pi)/config.delta)
    
    #rho[:, :N/2] = tanh((X[1][:, :N/2]-(0.5*pi))/config.delta)
    #rho[:, N/2:] =-tanh((X[1][:, N/2:]-(1.5*pi))/config.delta)
                
    rho0 = 0.5*(config.rho1 + config.rho2)
    U[0, :, :N[1]/2] = tanh((X[1, :, :N[1]/2] -0.5*pi)/config.delta)
    U[0, :, N[1]/2:] = -tanh((X[1, :, N[1]/2:]-1.5*pi)/config.delta)
    rho[:, :N[1]/2] = 2.0 + tanh((X[1, :, :N[1]/2] -0.5*pi)/config.delta)
    rho[:, N[1]/2:] = 2.0 -tanh((X[1, :, N[1]/2:]-1.5*pi)/config.delta) 
    rho -= rho0
    
    for i in range(3):
        Ur_hat[i] = FFT.fft2(Ur[i], Ur_hat[i]) 

im, im2 = None, None
def update(t, tstep, comm, rank, rho, N, L, dx, curl, K, FFT, U_hat, U, sum, 
           P_hat, P, hdf5file, float64, rho_hat, **kwargs):
    global im, im2
        
    if tstep == 1 and config.plot_result > 0:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Density', fontsize=20)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        im = ax.imshow(zeros((N[0], N[1])),cmap=plt.cm.bwr, extent=[0, L[0], 0, L[1]])
        plt.colorbar(im)
        plt.draw() 

        fig2, ax2 = plt.subplots(1,1)
        fig2.suptitle('Vorticity', fontsize=20)   
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        im2 = ax2.imshow(zeros((N[0], N[1])),cmap=plt.cm.bwr, extent=[0, L[0], 0, L[1]])
        plt.colorbar(im2)
        plt.draw()
        globals().update(dict(im=im, im2=im2))

    if tstep % config.plot_result == 0 and config.plot_result > 0:
        curl[:] = FFT.ifft2(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        im.set_data(rho[:, :].T)
        im.autoscale()
        plt.pause(1e-6)
        im2.set_data(curl[:,:].T)
        im2.autoscale()
        plt.pause(1e-6)
        if rank == 0:
            print tstep
            
    if tstep % config.write_result == 0 or tstep % config.write_yz_slice[1] == 0:
        P = FFT.ifft2(P_hat*1j, P)
        curl = FFT.ifft2(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        hdf5file.write(tstep)           

    if tstep % config.compute_energy == 0:
        kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)
        if rank == 0:
            print tstep, kk

if __name__ == "__main__":
    config.update(
        {
            'nu': 1.0e-08,
            'dt': 0.001,
            'T': 1.0,
            'U1':-0.5,
            'U2':0.5,
            'l0': 0.001,    # Smoothing parameter
            'A': 0.01,      # Amplitude of perturbation
            'Ri': 0.167,    # Richardson number
            'Pr': 12.0,     # Prantl number
            'delta': 0.05,   # Width of perturbations
            'bb': 0.8,
            'k0': 2,
            'rho1': 1.0,
            'rho2': 3.0,
        }, 'doublyperiodic'
    )
    config.doublyperiodic.add_argument("--plot_result", type=int, default=10)
    config.doublyperiodic.add_argument("--compute_energy", type=int, default=2)
    solver = get_solver(update, mesh='doublyperiodic')
    solver.hdf5file.components["curl"] = solver.curl
    assert config.solver == 'Bq2D'
    initialize(**vars(solver))
    solver.solve()
