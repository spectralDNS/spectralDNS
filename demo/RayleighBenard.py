"""Turbulent channel"""
import warnings
from mpi4py import MPI
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cbook
from mpiFFT4py import dct
from scipy.special import erf
from spectralDNS import config, get_solver, solve
from shenfun import Basis, TensorProductSpace, inner, Array, Function, Dx, \
    TrialFunction, TestFunction, MixedTensorProductSpace, div, grad, project
from shenfun.spectralbase import inner_product
from shenfun.chebyshev.la import Helmholtz
from spectralDNS.shen.Matrices import HelmholtzCoeff
from spectralDNS.solvers.spectralinit import HDF5Writer
from MKM import dx

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

def update(context):
    global im1, im3

    c = context
    params = config.params
    solver = config.solver
    X, U, U_hat, phi = c.X, c.U, c.U_hat, c.phi

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
        if hasattr(c.FST, 'complex_shape'):
            e0 = c.FST.dx(U[0]*U[0], c.ST.quad)
            e1 = c.FST.dx(U[1]*U[1], c.ST.quad)
            e2 = c.FST.dx(U[2]*U[2], c.ST.quad)
            q = c.FST.dx(U[1], c.ST.quad)
        else:
            e0 = dx(U[0]*U[0], c.FST)
            e1 = dx(U[1]*U[1], c.FST)
            e2 = dx(U[2]*U[2], c.FST)
            e3 = dx(phi*phi, c.FRB)
            q = dx(U[1], c.FST)
            div_u = solver.get_divergence(**c)
            e4 = dx(div_u*div_u, c.FST)
        if solver.rank == 0:
            print("Time %2.5f Energy %2.6e %2.6e %2.6e %2.6e div %2.6e" %(config.params.t, e0, e1, e2, e3, e4))

if __name__ == "__main__":
    config.update(
        {'Ra': 8e3,
         'Pr': 0.7,
         'dt': 0.01,               # Time step
         'T': 10.,                  # End time
         'L': [2, 2*np.pi, 2*np.pi],
         'M': [6, 6, 6]
        }, "channel"
    )

    config.channel.add_argument("--compute_energy", type=int, default=10)
    config.channel.add_argument("--plot_result", type=int, default=100)
    solver = get_solver(update=update, mesh="channel")
    config.params.nu = np.sqrt(config.params.Pr/config.params.Ra)
    config.params.kappa = 1./np.sqrt(config.params.Pr*config.params.Ra)
    context = solver.get_context()
    initialize(solver, context)
    context.hdf5file.fname = "KMM_RB_666g.h5"
    solve(solver, context)
