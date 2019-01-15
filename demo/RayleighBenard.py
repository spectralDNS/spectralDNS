"""Rayleigh Benard flow in channel"""
import warnings
import sys
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
beta = np.zeros(1)
def update(context):
    global im1, im3

    c = context
    params = config.params
    solver = config.solver
    X, U, phi = c.X, c.U, c.phi

    if (params.tstep % params.compute_energy == 0 or params.tstep % params.sample_stats or
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
        im1.set_UVC(U[1, :, :, 0], U[0, :, :, 0])
        im1.scale = np.linalg.norm(0.25*U[1])

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
        beta[0] = e0+e1+e2+e3+e4
        comm.Bcast(beta)
        if abs(beta[0]) > 1e8 or np.isnan(beta[0]):
            print('Diverging! Stopping...', params.tstep, beta[0])
            sys.exit(1)
        if solver.rank == 0:
            print("Time %2.5f Energy %2.6e %2.6e %2.6e %2.6e div %2.6e" %(config.params.t, e0, e1, e2, e3, e4))

    if params.tstep % params.sample_stats == 0:
        solver.stats(U, phi)

class Stats(object):

    def __init__(self, T, axis=0, fromstats="", filename=""):
        self.T = T
        self.axis = axis
        N = config.params.N
        assert np.all(np.array(T.shape(False)[1:]) == np.array(N))
        M = self.T.local_shape(False)[self.axis+1]
        self.Umean = np.zeros((3, M))
        self.phim = np.zeros(M)
        self.UU = np.zeros((6, M))
        self.pp = np.zeros(M)
        self.num_samples = 0
        self.rank = comm.Get_rank()
        self.fname = filename
        self.comm = comm
        self.f0 = None
        if fromstats:
            self.fromfile(filename=fromstats)

    def create_statsfile(self):
        import h5py
        self.f0 = h5py.File(self.fname+".h5", "w", driver="mpio", comm=comm)
        self.f0.create_group("Average")
        self.f0.create_group("Reynolds Stress")
        for i in ("U", "V", "W", "phi"):
            self.f0["Average"].create_dataset(i, shape=(config.params.N[self.axis],), dtype=float)

        for i in ("UU", "VV", "WW", "UV", "UW", "VW", "pp"):
            self.f0["Reynolds Stress"].create_dataset(i, shape=(config.params.N[self.axis],), dtype=float)

    def __call__(self, U, phi):
        sx = list(range(3))
        sx.pop(self.axis)
        sx = np.array(sx)
        self.num_samples += 1
        self.Umean += np.sum(U, axis=tuple(sx+1))
        sx = tuple(sx)
        self.phim += np.sum(phi, axis=sx)
        self.UU[0] += np.sum(U[0]*U[0], axis=sx)
        self.UU[1] += np.sum(U[1]*U[1], axis=sx)
        self.UU[2] += np.sum(U[2]*U[2], axis=sx)
        self.UU[3] += np.sum(U[0]*U[1], axis=sx)
        self.UU[4] += np.sum(U[0]*U[2], axis=sx)
        self.UU[5] += np.sum(U[1]*U[2], axis=sx)
        self.pp += np.sum(phi*phi, axis=sx)
        self.get_stats()

    def get_stats(self, tofile=True):
        import h5py
        sx = list(range(3))
        sx.pop(self.axis)
        N = config.params.N
        s = self.T.local_slice(False)[self.axis+1]
        Nd = self.num_samples*np.prod(np.take(N, sx))
        self.comm.barrier()
        if tofile:
            if self.f0 is None:
                self.create_statsfile()
            else:
                self.f0 = h5py.File(self.fname+".h5", "a", driver="mpio", comm=comm)

            for i, name in enumerate(("U", "V", "W")):
                self.f0["Average/"+name][s] = self.Umean[i]/Nd
            self.f0["Average/phi"][s] = self.phim/Nd
            for i, name in enumerate(("UU", "VV", "WW", "UV", "UW", "VW")):
                self.f0["Reynolds Stress/"+name][s] = self.UU[i]/Nd
            self.f0["Reynolds Stress/pp"][s] = self.pp/Nd
            self.f0.attrs.create("num_samples", self.num_samples)
            self.f0.close()

        if self.comm.Get_size() == 1:
            return self.Umean/Nd, self.phim/Nd, self.UU/Nd, self.pp/Nd
        return 0

    def reset_stats(self):
        self.num_samples = 0
        self.Umean[:] = 0
        self.phim[:] = 0
        self.UU[:] = 0
        self.pp[:] = 0

    def fromfile(self, filename="stats"):
        import h5py
        sx = list(range(3))
        sx.pop(self.axis)
        self.fname = filename
        self.f0 = h5py.File(filename+".h5", "a", driver="mpio", comm=comm)
        N = config.params.N
        self.num_samples = self.f0.attrs["num_samples"]
        Nd = self.num_samples*np.prod(np.take(N, sx))
        s = self.T.local_slice(False)[self.axis+1]
        for i, name in enumerate(("U", "V", "W")):
            self.Umean[i, :] = self.f0["Average/"+name][s]*Nd
        self.phim[:] = self.f0["Average/phi"][s]*Nd
        for i, name in enumerate(("UU", "VV", "WW", "UV", "UW", "VW")):
            self.UU[i, :] = self.f0["Reynolds Stress/"+name][s]*Nd
        self.pp[:] = self.f0["Reynolds Stress/pp"][s]*Nd
        self.f0.close()

def init_from_file(filename, solver, context):
    import h5py
    f = h5py.File(filename, 'r+', driver="mpio", comm=solver.comm)
    assert "0" in f["U/Vector/3D"]
    U_hat = context.U_hat
    phi_hat = context.phi_hat
    TV = context.U.function_space()
    su = TV.local_slice(True)
    T = context.phi.function_space()
    sp = T.local_slice(True)

    # previous timestep
    if not 'RK3' in config.params.solver:
        assert "1" in f["U/Vector/3D"]
        U_hat[:] = f["U/Vector/3D/1"][su]

        # Set g, which is used in computing convection
        context.g[:] = 1j*context.K[1]*U_hat[2] - 1j*context.K[2]*U_hat[1]
        context.U_hat0[:] = U_hat
        context.H_hat1[:] = solver.get_convection(**context)
        context.phi_hat0[:] = f["phi/3D/1"][sp]

    # current timestep
    U_hat[:] = f["U/Vector/3D/0"][su]
    phi_hat[:] = f["phi/3D/0"][sp]
    context.g[:] = 1j*context.K[1]*U_hat[2] - 1j*context.K[2]*U_hat[1]
    context.hdf5file.filename = filename
    if 'tstep' in f.attrs:
        config.params.tstep = f.attrs['tstep']
    if 't' in f.attrs:
        config.params.tstep = f.attrs['t']
    f.close()

if __name__ == "__main__":
    config.update(
        {'dt': 0.01,               # Time step
         'T': 1000.,                  # End time
         'L': [2, 2*np.pi, 2*np.pi],
         'M': [6, 7, 7]
        }, "channel"
    )
    config.channel.add_argument("--compute_energy", type=int, default=10)
    config.channel.add_argument("--plot_result", type=int, default=100)
    config.channel.add_argument("--Ra", type=float, default=10000.0)
    config.channel.add_argument("--Pr", type=float, default=0.7)
    config.channel.add_argument("--sample_stats", type=int, default=10)
    solver = get_solver(update=update, mesh="channel")
    config.params.nu = np.sqrt(config.params.Pr/config.params.Ra)
    config.params.kappa = 1./np.sqrt(config.params.Pr*config.params.Ra)
    context = solver.get_context()
    #initialize(solver, context)
    init_from_file("KMMRK3_RB_677g_c.h5", solver, context)
    config.params.tstep = 20
    config.params.t = 0.2
    #context.hdf5file.filename = "KMMRK3_RB_677g"

    # Just store slices
    context.hdf5file.results['space'] = context.FST
    context.hdf5file.results['data'] = {'U0': [(context.U[0], [slice(None), slice(None), 0]),
                                               (context.U[0], [slice(None), 0, slice(None)])],
                                        'U1': [(context.U[1], [slice(None), slice(None), 0]),
                                               (context.U[1], [slice(None), 0, slice(None)])],
                                        'U2': [(context.U[2], [slice(None), slice(None), 0]),
                                               (context.U[2], [slice(None), 0, slice(None)])],
                                        'phi': [(context.phi, [slice(None), slice(None), 0]),
                                                (context.phi, [slice(None), 0, slice(None)])]
                                       }
    solver.stats = Stats(context.VFS, filename="KMMRK3_RB_stats")
    solve(solver, context)
