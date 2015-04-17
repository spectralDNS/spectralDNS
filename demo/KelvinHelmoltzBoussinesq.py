from cbcdns import config, get_solver
import matplotlib.pyplot as plt
from numpy import zeros

def initialize(X, U, Ur, Ur_hat, exp, sin, cos, tanh, rho, Np, N, pi, fft2_mpi, Rip, **kwargs):

    Um = 0.5*(config.U1 - config.U2)
    U[1] = config.A*sin(0.5*X[0])
    U[0, :, :N/4] = config.U1 - Um*exp((X[1,:, :N/4] - 0.5*pi)/config.delta) 
    U[0, :, N/4:N/2] = config.U2 + Um*exp(-1.0*(X[1, :, N/4:N/2] - 0.5*pi)/config.delta) 
    U[0, :, N/2:3*N/4] = config.U2 + Um*exp((X[1, :, N/2:3*N/4] - 1.5*pi)/config.delta) 
    U[0, :, 3*N/4:] = config.U1 - Um*exp(-1.0*(X[1, :, 3*N/4:] - 1.5*pi)/config.delta)
    Rip *= config.Ri
    
    #for i in range(Np):
        #for j in range(N):
            #if 0.0 <= X[1][i,j] < 0.5*pi:
                #U[0][i,j] = config.U1 - Um*exp((X[1][i,j] - 0.5*pi)/config.delta) 
            #elif 0.5*pi <= X[1][i,j] < pi:
                #U[0][i,j] = config.U2 + Um*exp(-1.0*(X[1][i,j] - 0.5*pi)/config.delta) 
            #elif pi <= X[1][i,j] < 1.5*pi:
                #U[0][i,j] = config.U2 + Um*exp((X[1][i,j] - 1.5*pi)/config.delta) 
            #elif 1.5*pi <= X[1][i,j] < 2*pi:
                #U[0][i,j] = config.U1 - Um*exp(-1.0*(X[1][i,j] - 1.5*pi)/config.delta)

                
    #for i in range(Np):
    #for j in range(N):
        #if 0.0 <= X[1][i,j] < pi:
            #rho[i, j] = 1.0 + 0.5*exp((X[1][i,j] - pi)/delta)
        #elif pi <= X[1][i,j] < 2*pi:
        #rho[i, j] = 2.0 - 0.5*exp(-1*(X[1][i,j] - pi)/delta)
        #elif 2*pi <= X[1][i,j] < 3*pi:
        #rho[i, j] = 2.0 - 0.5*exp((X[1][i,j] - 3*pi)/delta)
        #elif 3*pi <= X[1][i,j] < 4*pi:
        #rho[i, j] =1.0 + 0.5*exp(-1.0*(X[1][i,j] - 3*pi)/delta)

    for i in range(Np):
        for j in range(N):
            if 0.0 <= X[1][i,j] < pi:
                rho[i, j] = tanh((X[1][i,j]-(0.5*pi))/config.delta)
            elif pi <= X[1][i,j] < 2*pi:
                rho[i, j] = -tanh((X[1][i,j]-(1.5*pi))/config.delta)
          
    for i in range(3):
        Ur_hat[i] = fft2_mpi(Ur[i], Ur_hat[i]) 

im, im2 = None, None
def update(t, tstep, comm, rank, rho, N, L, dx, curl, K, ifft2_mpi, U_hat, U, sum, 
           P_hat, P, hdf5file, float64, **kwargs):
    global im, im2
        
    if tstep == 1 and config.plot_result > 0:
        fig, ax = plt.subplots(1,1)
        fig.suptitle('Density', fontsize=20)
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
        im.set_data(rho[:, :].T)
        im.autoscale()
        plt.pause(1e-6)
        im2.set_data(curl[:,:].T)
        im2.autoscale()
        plt.pause(1e-6)
        if rank == 0:
            print tstep
            
    if tstep % config.write_result == 0 or tstep % config.write_yz_slice[1] == 0:
        P = ifft2_mpi(P_hat*1j, P)
        hdf5file.write(tstep)           

    if tstep % config.compute_energy == 0:
        kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx*dx/L**2/2)
        if rank == 0:
            print tstep, kk

def regression_test(comm, U, X, dx, L, nu, t, sin, cos, sum, float64, exp, 
                    rank, **kwargs):
    print "---------regression_test--------------"
    # Check accuracy
    #kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx*dx/L**2/2)
    #u0 = sin(X[0])*cos(X[1])*exp(-2.*nu*t)
    #u1 =-sin(X[1])*cos(X[0])*exp(-2.*nu*t)
    #k1 = comm.reduce(sum(u0*u0+u1*u1)*dx*dx/L**2/2) # Compute energy with double precision)
    #if rank == 0:
        #print "Energy (exact, numeric, error)  = (%2.6f, %2.6f, %2.4e) " %(k1, kk, k1-kk)
        #assert abs(k1-kk)<1.e-10

if __name__ == "__main__":
    config.update(
    {
    'nu': 1.0e-05,
    'dt': 0.001,
    'T': 1.0,
    'U1':-0.5,
    'U2':0.5,
    'l0': 0.001,    # Smoothing parameter
    'A': 0.01,      # Amplitude of perturbation
    'Ri': 0.167,    # Richardson number
    'Pr': 12.0,     # Prantl number
    'delta': 0.1,   # Width of perturbations
    'bb': 0.8,
    'k0': 2
    }
    )
    config.parser.add_argument("--plot_result", type=int, default=10) # required to allow overloading through commandline    
    config.parser.add_argument("--compute_energy", type=int, default=2)
    solver = get_solver(update)
    initialize(**vars(solver))
    solver.solve()
