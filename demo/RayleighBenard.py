"""Rayleigh Benard flow in channel"""
import warnings
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook
from spectralDNS import config, get_solver, solve
from spectralDNS.utilities import dx

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

comm = MPI.COMM_WORLD

def initialize(solver, context):
    X = context.X
    phi = context.phi

    # Perturb temperature
    phi[:] = 0.5*(1-X[0])+0.01*np.random.randn(*phi.shape)*(1-X[0])*(1+X[0])
    phi_hat = phi.forward(context.phi_hat)
    phi = phi_hat.backward(phi)
    phi_hat = phi.forward(phi_hat)

    if not 'RK3' in config.params.solver:
        context.phi_hat0[:] = phi_hat
        context.phi_hat1[:] = phi_hat

im1, im2 = None, None

def update(context):
    global im1, im3

    c = context
    params = config.params
    solver = config.solver
    X, U, phi = c.X, c.U, c.phi

    if (params.tstep % params.compute_energy == 0 or
            params.tstep % params.plot_result == 0 and params.plot_result > 0):
        U = solver.get_velocity(**c)
        phi = c.phi_hat.backward(c.phi)

    if params.tstep == 1 and solver.rank == 0 and params.plot_result > 0:
        # Initialize figures
        plt.figure(1, figsize=(6,3))
        im1 = plt.quiver(X[1][:, :, 0], X[0][:, :, 0], U[1, :, :, 0], U[0, :, :, 0], pivot='mid', scale=5)
        #im1.set_array(U[0,:,:,0])
        plt.draw()

        plt.figure(2, figsize=(6, 3))
        im3 = plt.contourf(X[1][0, :, 0], X[0][:, 0, 0], phi[:, :, 0], 100)
        plt.colorbar(im3)
        plt.draw()

        plt.pause(1e-6)

    if params.tstep % params.plot_result == 0 and solver.rank == 0 and params.plot_result > 0:
        plt.figure(1)
        im1.set_UVC(U[1,:,:,0], U[0,:,:,0])
        im1.scale=np.linalg.norm(0.25*U[1])

        plt.pause(1e-6)
        plt.figure(2)
        im3.ax.clear()
        im3.ax.contourf(X[1][0, :, 0], X[0][:, 0, 0], phi[:, :, 0], 100)
        im3.autoscale()
        plt.pause(1e-6)

    if params.tstep % params.compute_energy == 0:
        e0 = dx(U[0]*U[0], c.FST)
        e1 = dx(U[1]*U[1], c.FST)
        e2 = dx(U[2]*U[2], c.FST)
        e3 = dx(phi*phi, c.FRB)
        div_u = solver.get_divergence(**c)
        e4 = dx(div_u*div_u, c.FST)
        if solver.rank == 0:
            print("Time %2.5f Energy %2.6e %2.6e %2.6e %2.6e div %2.6e" %(config.params.t, e0, e1, e2, e3, e4))

if __name__ == "__main__":
    config.update(
        {'dt': 0.01,               # Time step
         'T': 10.,                  # End time
         'L': [2, 2*np.pi, 2*np.pi],
         'M': [6, 6, 6]
        }, "channel"
    )

    config.channel.add_argument("--compute_energy", type=int, default=10)
    config.channel.add_argument("--plot_result", type=int, default=100)
    config.channel.add_argument("--Ra", type=float, default=20000.0)
    config.channel.add_argument("--Pr", type=float, default=0.7)
    solver = get_solver(update=update, mesh="channel")
    config.params.nu = np.sqrt(config.params.Pr/config.params.Ra)
    config.params.kappa = 1./np.sqrt(config.params.Pr*config.params.Ra)
    context = solver.get_context()
    initialize(solver, context)
    context.hdf5file.fname = "KMM_RB_666g.h5"
    solve(solver, context)
