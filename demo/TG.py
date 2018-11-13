from __future__ import print_function
import warnings
from numpy import pi, zeros, sum, float64, sin, cos, prod, asscalar
from spectralDNS import config, get_solver, solve

try:
    import matplotlib.pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")
    plt = None

def initialize(solver, context):
    if 'NS' in config.params.solver:
        initialize1(solver, context)

    else:
        initialize2(solver, context)
    config.params.t = 0.0
    config.params.tstep = 0

def initialize1(solver, context):
    U, X = context.U, context.X
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] = -cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0
    solver.set_velocity(**context)

def initialize2(solver, context):
    U, X = context.U, context.X
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] = -cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0
    solver.set_velocity(**context)
    solver.cross2(context.W_hat, context.K, context.U_hat)

def energy_fourier(comm, a):
    result = 2*sum(abs(a[..., 1:-1])**2) + sum(abs(a[..., 0])**2) + sum(abs(a[..., -1])**2)
    result = comm.allreduce(result)
    return result

k = []
w = []
im1 = None
kold = zeros(1)
def update(context):
    global k, w, im1
    c = context
    params = config.params
    solver = config.solver

    if (params.tstep % params.compute_energy == 0 or
            params.tstep % params.plot_step == 0 and params.plot_step > 0):
        U = solver.get_velocity(**c)
        curl = solver.get_curl(**c)
        if 'NS' in params.solver:
            solver.get_pressure(**c)

    if plt is not None:
        if params.tstep % params.plot_step == 0 and solver.rank == 0 and params.plot_step > 0:
            if im1 is None:
                plt.figure()
                im1 = plt.contourf(c.X[1][:, :, 0], c.X[0][:, :, 0], U[0, :, :, 10], 100)
                plt.colorbar(im1)
                plt.draw()
                globals().update(im1=im1)
            else:
                im1.ax.clear()
                im1.ax.contourf(c.X[1][:, :, 0], c.X[0][:, :, 0], U[0, :, :, 10], 100)
                im1.autoscale()
            plt.pause(1e-6)

    #if params.tstep == 1:
    #    from spectralDNS.utilities import reset_profile
    #    print("Reset profile")
    #    reset_profile(profile)

    if params.tstep % params.compute_energy == 0:
        #dx, L = params.dx, params.L
        #if 'NS' in params.solver:
            #ww = comm.reduce(sum(curl*curl)/prod(params.N)/2)

            #duidxj = work[(((3,3)+FFT.real_shape()), FFT.float, 0)]
            #for i in range(3):
                #for j in range(3):
                    #duidxj[i,j] = FFT.ifftn(1j*K[j]*U_hat[i], duidxj[i,j])
            #ww2 = comm.reduce(sum(duidxj*duidxj)/prod(params.N)/2)

            #ddU = work[(((3,)+FFT.real_shape()), FFT.float, 0)]
            #dU = ComputeRHS(dU, U_hat)
            #for i in range(3):
                #ddU[i] = FFT.ifftn(dU[i], ddU[i])
            #ww3 = comm.reduce(sum(ddU*U)/prod(params.N)/2)

            #if rank == 0:
                #print ww, params.nu*ww2, ww3, ww-ww2

        ww = solver.comm.reduce(sum(curl.astype(float64)*curl.astype(float64))/prod(params.N)/2)
        kk = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))/prod(params.N)/2) # Compute energy with double precision
        ww2 = energy_fourier(solver.comm, c.U_hat)/prod(params.N)**2/2

        kold[0] = kk
        if solver.rank == 0:
            k.append(kk)
            w.append(ww)
            print("%2.2f %2.8f %2.8f %2.8f" %(params.t, float(kk), float(ww), float(ww2)))
    #if params.tstep % params.compute_energy == 1:
        #if 'NS' in params.solver:
            #kk2 = comm.reduce(sum(U.astype(float64)*U.astype(float64))/prod(params.N)/2)
            #if rank == 0:
                #print 0.5*(kk2-kold[0])/params.dt

def regression_test(context):
    params = config.params
    solver = config.solver
    U = solver.get_velocity(**context)
    curl = solver.get_curl(**context)
    w = solver.comm.reduce(sum(curl.astype(float64)*curl.astype(float64))/prod(params.N)/2)
    k = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))/prod(params.N)/2) # Compute energy with double precision
    config.solver.MemoryUsage('End')
    if solver.rank == 0:
        assert round(asscalar(w) - 0.375249930801, params.ntol) == 0, w
        assert round(asscalar(k) - 0.124953117517, params.ntol) == 0, k

if __name__ == "__main__":
    config.update(
        {'nu': 0.000625,             # Viscosity
         'dt': 0.01,                 # Time step
         'T': 0.1,                   # End time
         'L': [2*pi, 2.*pi, 2*pi],
         'M': [5, 5, 5],
         'planner_effort': {'fft': 'FFTW_ESTIMATE',
                            'rfftn': 'FFTW_ESTIMATE',
                            'irfftn': 'FFTW_ESTIMATE'},
        }, "triplyperiodic")
    config.triplyperiodic.add_argument("--compute_energy", type=int, default=2)
    config.triplyperiodic.add_argument("--plot_step", type=int, default=2)
    sol = get_solver(update=update, regression_test=regression_test,
                     mesh="triplyperiodic")

    context = sol.get_context()

    # Add curl to the stored results. For this we need to update the update_components
    # method used by the HDF5File class to compute the real fields that are stored
    context.hdf5file.filename = "NS9"
    context.hdf5file.results['data'].update({'curl': [context.curl]})
    def update_components(**c):
        """Overload default because we want to store the curl as well"""
        sol.get_velocity(**c)
        sol.get_pressure(**c)
        sol.get_curl(**c)

    context.hdf5file.update_components = update_components

    initialize(sol, context)
    solve(sol, context)
