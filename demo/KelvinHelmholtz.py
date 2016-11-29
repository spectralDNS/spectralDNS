from spectralDNS import config, get_solver, solve
import matplotlib.pyplot as plt
from numpy import zeros, pi, sum, exp, sin, cos, tanh, float64

def initialize(X, U, U_hat, FFT, **context):
    params = config.params
    Um = 0.5*(params.U1 - params.U2)
    N = params.N    
    U[1] = params.A*sin(2*X[0])       
    U[0, :, :N[1]/4] = params.U1 - Um*exp((X[1,:, :N[1]/4] - 0.5*pi)/params.delta)
    U[0, :, N[1]/4:N[1]/2] = params.U2 + Um*exp(-1.0*(X[1, :, N[1]/4:N[1]/2] - 0.5*pi)/params.delta)
    U[0, :, N[1]/2:3*N[1]/4] = params.U2 + Um*exp((X[1, :, N[1]/2:3*N[1]/4] - 1.5*pi)/params.delta)
    U[0, :, 3*N[1]/4:] = params.U1 - Um*exp(-1.0*(X[1, :, 3*N[1]/4:] - 1.5*pi)/params.delta)
          
    for i in range(2):
        U_hat[i] = FFT.fft2(U[i], U_hat[i])

im, im2 = None, None
count = 0
def update(context):
    global im, im2, count
    c = context
    params = config.params
    solver = config.solver
    dx, L, N = params.dx, params.L, params.N
    if params.tstep % params.plot_result == 0 and params.plot_result > 0:
        P = solver.get_pressure(**context)
        curl = solver.get_curl(**context)

    if params.tstep % params.compute_energy == 0:
        U = solver.get_velocity(**context)
        kk = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]/L[0]/L[1]/2)
        if solver.rank == 0:
            print params.tstep, kk
            
    if params.tstep == 1 and params.plot_result > 0:
        fig, ax = plt.subplots(1,1)
        fig.suptitle('Pressure', fontsize=20)
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

    if params.tstep % params.plot_result == 0 and params.plot_result > 0:
        count += 1
        im.set_data(P[:, :].T)
        im.autoscale()
        plt.pause(1e-6)
        im2.set_data(curl[:,:].T)
        im2.autoscale()
        plt.pause(1e-6)
        plt.savefig("KH_{}.png".format(count))
        if solver.rank == 0:
            print(params.tstep)

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
        }, 'doublyperiodic'
    )
    # Adding new arguments required here to allow overloading through commandline
    config.doublyperiodic.add_argument('--plot_result', type=int, default=50)    
    config.doublyperiodic.add_argument('--compute_energy', type=int, default=50)
    solver = get_solver(update=update, mesh='doublyperiodic')
    assert config.params.solver == 'NS2D'
    context = solver.get_context()
    initialize(**context)
    solve(solver, context)
