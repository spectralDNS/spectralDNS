from __future__ import print_function
from spectralDNS import config, get_solver, solve
import numpy as np
from numpy import array, pi, zeros, sum, float64, sin, cos
from numpy.linalg import norm
import warnings

try:
    import matplotlib.pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")
    plt = None

def initialize(solver, context):
    c = context
    c.mask = np.where(c.K2 <= config.params.Kf2, 1, 0)

    if 'rogallo' in config.params.initialize:
        initialize_rogallo(solver, context)

    else:
        initialize1(solver, context)
    config.params.t = 0.0
    config.params.tstep = 0
    c.target_energy = energy_fourier(solver.comm, c.U_hat)

def initialize_rogallo(solver, context):
    c = context
    u0 = np.prod(config.params.N)/np.prod(config.params.L)
    if 'shenfun' in config.params.solver:
        u0 = 1./np.prod(config.params.L)

    np.random.seed(solver.rank)
    kf = config.params.Kf2
    k = np.sqrt(c.K2)
    k = np.where(k==0, 1, k)
    K2 = c.K2
    K2 = np.where(K2==0, 1, K2)

    k1, k2, k3 = c.K[0], c.K[1], c.K[2]
    E0 = np.sqrt(9./11./kf*c.K2/kf**2)*c.mask
    E1 = np.sqrt(9./11./kf*(k/kf)**(-5./3.))*(1-c.mask)
    Ek = E0 + E1
    theta1, theta2, phi = np.random.sample(c.U_hat.shape)*2j*np.pi
    alpha = np.sqrt(Ek/4./np.pi/K2)*np.exp(1j*theta1)*np.cos(phi)
    beta = np.sqrt(Ek/4./np.pi/K2)*np.exp(1j*theta2)*np.sin(phi)
    ksq = np.sqrt(k1**2+k2**2)
    ksq = np.where(ksq==0, 1, ksq)

    c.U_hat[0] = (alpha*k*k2 + beta*k1*k3)/(k*ksq)
    c.U_hat[1] = (beta*k2*k3 - alpha*k*k1)/(k*ksq)
    c.U_hat[2] = beta*ksq/k

    if 'VV' in config.params.solver:
        c.W_hat = solver.cross2(c.W_hat, c.K, c.U_hat)

    U = solver.get_velocity(**c)
    U_hat = solver.set_velocity(**c)
    # project to zero divergence
    U_hat[:] -= (c.K[0]*U_hat[0]+c.K[1]*U_hat[1]+c.K[2]*U_hat[2])*c.K_over_K2
    if solver.rank == 0:
        c.U_hat[:, 0, 0, 0] = 0.0

    # Scale to get correct kinetic energy
    energy = 0.5*energy_fourier(solver.comm, c.U_hat)/np.prod(config.params.N)
    target = config.params.Re_lam*(config.params.nu*config.params.kd)**2/np.sqrt(20./3.)
    print(energy, target)
    c.U_hat *= np.sqrt(target/energy)
    print(0.5*energy_fourier(solver.comm, c.U_hat)/np.prod(config.params.N))

    if 'VV' in config.params.solver:
        c.W_hat = solver.cross2(c.W_hat, c.K, c.U_hat)


def initialize1(solver, context):
    c = context
    u0 = np.prod(config.params.N)/np.prod(config.params.L)
    if 'shenfun' in config.params.solver:
        u0 /= np.prod(config.params.N)

    np.random.seed(solver.rank)
    c.U_hat[:] = np.random.sample(c.U_hat.shape)*2j*np.pi
    c.U_hat[:] = u0/(2*np.pi)*c.K2*np.exp(-c.K2/config.params.a0**2)*np.exp(c.U_hat)
    if solver.rank == 0:
        c.U_hat[:, 0, 0, 0] = 0.0

    if 'VV' in config.params.solver:
        c.W_hat = solver.cross2(c.W_hat, c.K, c.U_hat)

    U = solver.get_velocity(**c)
    U_hat = solver.set_velocity(**c)
    # project to zero divergence
    U_hat[:] -= (c.K[0]*U_hat[0]+c.K[1]*U_hat[1]+c.K[2]*U_hat[2])*c.K_over_K2

    if 'VV' in config.params.solver:
        c.W_hat = solver.cross2(c.W_hat, c.K, c.U_hat)

def energy_fourier(comm, a):
    N = config.params.N
    result = 2*np.sum(np.abs(a[..., 1:-1])**2) + np.sum(np.abs(a[..., 0])**2) + np.sum(np.abs(a[..., -1])**2)
    result =  comm.allreduce(result)
    if 'shenfun' in config.params.solver:
        return result*np.prod(N)
    else:
        return result/np.prod(N)

def spectrum(solver, context):
    c = context
    uiui = np.zeros(c.U_hat[0].shape)
    uiui[..., 1:-1] = 2*np.sum((c.U_hat[...,1:-1]*np.conj(c.U_hat[..., 1:-1])).real, axis=0)
    uiui[..., 0] = np.sum((c.U_hat[..., 0]*np.conj(c.U_hat[..., 0])).real, axis=0)
    uiui[..., -1] = np.sum((c.U_hat[..., -1]*np.conj(c.U_hat[..., -1])).real, axis=0)
    if 'shenfun' in config.params.solver:
        uiui *= (2*np.pi*c.K2)
    else:
        uiui *= (2*np.pi*c.K2/np.prod(config.params.N)**2)

    # Create bins for Ek
    Nb = int(np.sqrt(sum((config.params.N/2)**2)))
    bins = range(0, Nb)
    z = np.digitize(np.sqrt(context.K2), bins, right=True)
    #bins = np.unique(np.sqrt(context.K2))
    #z = np.digitize(np.sqrt(context.K2), bins, right=True)
    #Nb = len(bins)

    # Sample
    Ek = np.zeros(Nb)
    for i in range(1, Nb):
        ii = np.where(z == i)
        if len(ii[0]) > 0:
            Ek[i] = np.sum(uiui[ii]) / len(ii[0])

    Ek = solver.comm.allreduce(Ek)

    E0 = uiui.mean(axis=(1, 2))
    E1 = uiui.mean(axis=(0, 2))
    E2 = uiui.mean(axis=(0, 1))

    ## Rij
    #for i in range(3):
        #c.U[i] = c.FFT.ifftn(c.U_hat[i], c.U[i])
    #X = c.FFT.get_local_mesh()
    #R = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    ## Sample
    #Rii = np.zeros_like(c.U)
    #Rii[0] = c.FFT.ifftn(np.conj(c.U_hat[0])*c.U_hat[0], Rii[0])
    #Rii[1] = c.FFT.ifftn(np.conj(c.U_hat[1])*c.U_hat[1], Rii[1])
    #Rii[2] = c.FFT.ifftn(np.conj(c.U_hat[2])*c.U_hat[2], Rii[2])

    #R11 = np.sum(Rii[:, :, 0, 0] + Rii[:, 0, :, 0] + Rii[:, 0, 0, :], axis=0)/3

    #Nr = 20
    #rbins = np.linspace(0, 2*np.pi, Nr)
    #rz = np.digitize(R, rbins, right=True)
    #RR = np.zeros(Nr)
    #for i in range(Nr):
        #ii = np.where(rz == i)
        #RR[i] = np.sum(Rii[0][ii] + Rii[1][ii] + Rii[2][ii]) / len(ii[0])

    #Rxx = np.zeros((3, config.params.N[0]))
    #for i in range(config.params.N[0]):
        #Rxx[0, i] = (c.U[0] * np.roll(c.U[0], -i, axis=0)).mean()
        #Rxx[1, i] = (c.U[0] * np.roll(c.U[0], -i, axis=1)).mean()
        #Rxx[2, i] = (c.U[0] * np.roll(c.U[0], -i, axis=2)).mean()

    return Ek, bins, E0, E1, E2

k = []
w = []
im1 = None
kold = zeros(1)

energy_target = None
energy_new = None
def update(context):
    global k, w, im1, energy_target, energy_new
    c = context
    params = config.params
    solver = config.solver
    curl_hat = c.work[(c.U_hat, 2, True)]

    if solver.rank == 0:
        c.U_hat[:, 0, 0, 0] = 0

    #if energy_target is None:
        #energy_target = energy_fourier(solver.comm, c.U_hat)
    #else:
        #energy_target = energy_new

    if params.solver == 'VV':
        c.U_hat = solver.cross2(c.U_hat, c.K_over_K2, c.W_hat)

    energy_new = energy_fourier(solver.comm, c.U_hat)
    energy_lower = energy_fourier(solver.comm, c.U_hat*c.mask)
    energy_upper = energy_new - energy_lower

    alpha2  = (c.target_energy - energy_upper) /energy_lower
    alpha = np.sqrt(alpha2)

    #du = c.U_hat*c.mask*(alpha)
    #dus = energy_fourier(solver.comm, du*c.U_hat)

    energy_old = energy_new

    #c.dU[:] = alpha*c.mask*c.U_hat
    c.U_hat *= (alpha*c.mask + (1-c.mask))
    #c.U_hat[:] -= (c.K[0]*c.U_hat[0]+c.K[1]*c.U_hat[1]+c.K[2]*c.U_hat[2])*c.K_over_K2

    energy_new = energy_fourier(solver.comm, c.U_hat)

    assert np.sqrt((energy_new-c.target_energy)**2) < 1e-7

    if params.solver == 'VV':
        c.W_hat = solver.cross2(c.W_hat, c.K, c.U_hat)

    if (params.tstep % params.compute_energy == 0 or
          params.tstep % params.plot_step == 0 and params.plot_step > 0):
        U = solver.get_velocity(**c)
        curl = solver.get_curl(**c)
        if params.solver == 'NS':
            P = solver.get_pressure(**c)

    K = c.K
    if plt is not None:
        if params.tstep % params.plot_step == 0 and solver.rank == 0 and params.plot_step > 0:
            #div_u = solver.get_divergence(**c)

            if im1 is None:
                plt.figure()
                #im1 = plt.contourf(c.X[1][:,:,0], c.X[0][:,:,0], div_u[:,:,10], 100)
                im1 = plt.contourf(c.X[1][:,:,0], c.X[0][:,:,0], c.U[0,:,:,10], 100)
                plt.colorbar(im1)
                plt.draw()
                globals().update(im1=im1)
            else:
                im1.ax.clear()
                #im1.ax.contourf(c.X[1][:,:,0], c.X[0][:,:,0], div_u[:,:,10], 100)
                im1.ax.contourf(c.X[1][:,:,0], c.X[0][:,:,0], c.U[0,:,:,10], 100)
                im1.autoscale()
            plt.pause(1e-6)

    if params.tstep % params.compute_spectrum == 0:
        Ek, bins, E0, E1, E2 = spectrum(solver, context)
        context.hdf5file.f = h5py.File(context.hdf5file.fname, driver='mpio', comm=solver.comm)
        context.hdf5file.f['Turbulence/Ek'].create_dataset(str(params.tstep), data=Ek)
        context.hdf5file.f.close()

    if params.tstep % params.compute_energy == 0:
        dx, L = params.dx, params.L
        #ww = solver.comm.reduce(sum(curl*curl)/np.prod(params.N)/2)

        #curl_hat = c.work[(c.U_hat, 2, True)]
        #curl_hat = solver.cross2(curl_hat, K, c.U_hat)
        #ww = energy_fourier(solver.comm, params.N, curl_hat)/np.prod(params.N)/2

        duidxj = c.work[(((3,3)+c.U[0].shape), c.float, 0)]
        for i in range(3):
            for j in range(3):
                if 'shenfun' in config.params.solver:
                    duidxj[i,j] = c.T.backward(1j*K[j]*c.U_hat[i], duidxj[i,j])
                else:
                    duidxj[i,j] = c.FFT.ifftn(1j*K[j]*c.U_hat[i], duidxj[i,j])

        ww2 = solver.comm.reduce(sum(duidxj*duidxj))

        ddU = c.work[(((3,)+c.U[0].shape), c.float, 0)]
        dU = solver.ComputeRHS(c.dU, c.U_hat, solver, **c)
        for i in range(3):
            if 'shenfun' in config.params.solver:
                ddU[i] = c.T.backward(dU[i], ddU[i])
            else:
                ddU[i] = c.FFT.ifftn(dU[i], ddU[i])

        ww3 = solver.comm.reduce(sum(ddU*U))

        ##if solver.rank == 0:
            ##print('W ', params.nu*ww, params.nu*ww2, ww3, ww-ww2)
        curl_hat = solver.cross2(curl_hat, K, c.U_hat)
        dissipation = energy_fourier(solver.comm, curl_hat)
        div_u = solver.get_divergence(**c)
        div_u = np.sum(div_u**2)
        div_u2 = energy_fourier(solver.comm, 1j*(K[0]*c.U_hat[0]+K[1]*c.U_hat[1]+K[2]*c.U_hat[2]))

        kk = 0.5*energy_new/np.prod(params.N)
        eps = dissipation*params.nu/np.prod(params.N)
        Re_lam = np.sqrt(20*kk**2/(3*params.nu*eps))
        Re_lam2 = kk*np.sqrt(20./3.)/(params.nu*params.kd)**2

        kold[0] = energy_new
        factor = 1./np.prod(params.N)
        if solver.rank == 0:
            k.append(energy_new)
            w.append(dissipation)
            print(params.t, alpha,  kk, dissipation*params.nu*factor, ww3, div_u, (energy_new-energy_old)/2/params.dt*factor, Re_lam, Re_lam2)

    #if params.tstep % params.compute_energy == 1:
        #if 'NS' in params.solver:
            #kk2 = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            #if rank == 0:
                #print 0.5*(kk2-kold[0])/params.dt

def init_from_file(filename, solver, context):
    f = h5py.File(filename, driver="mpio", comm=solver.comm)
    assert "1" in f["3D/checkpoint/U"]
    U = context.U
    N = U.shape[1]
    s = context.T.local_slice(spectral=False)

    U[:] = f["3D/checkpoint/U/1"][:, s[0], s[1], s[2]]
    U_hat = solver.set_velocity(**context)
    U_hat[:] -= (context.K[0]*U_hat[0]+context.K[1]*U_hat[1]+context.K[2]*U_hat[2])*context.K_over_K2
    if solver.rank == 0:
        U_hat[:, 0, 0, 0] = 0.0

    if 'VV' in config.params.solver:
        context.W_hat = solver.cross2(ccontext.W_hat, context.K, context.U_hat)

    context.target_energy = energy_fourier(solver.comm, U_hat)

    f.close()


if __name__ == "__main__":
    import h5py
    config.update(
        {
        'nu': 0.005428,              # Viscosity (not used, see below)
        'dt': 0.002,                 # Time step
        'T': 0.05,                   # End time
        'L': [2.*pi, 2.*pi, 2.*pi],
        'M': [7, 7, 7],
        'checkpoint': 100,
        'write_result': 100,
        #'decomposition': 'pencil',
        #'Pencil_alignment': 'Y',
        #'P1': 2
        },  "triplyperiodic"
    )
    config.triplyperiodic.add_argument("--compute_energy", type=int, default=10)
    config.triplyperiodic.add_argument("--compute_spectrum", type=int, default=10)
    config.triplyperiodic.add_argument("--plot_step", type=int, default=1000)
    config.triplyperiodic.add_argument("--Kf2", type=int, default=3)
    config.triplyperiodic.add_argument("--a0", type=float, default=5.5)
    config.triplyperiodic.add_argument("--kd", type=float, default=50.)
    config.triplyperiodic.add_argument("--Re_lam", type=float, default=84.)
    config.triplyperiodic.add_argument("--initialize", type=str, default='rogallo')
    sol = get_solver(update=update, mesh="triplyperiodic")
    config.params.nu = (1./config.params.kd**(4./3.))

    context = sol.get_context()
    initialize(sol, context)
    #init_from_file("NS_isotropic_{}_{}_{}.h5".format(*config.params.M), sol, context)

    Ek, bins, E0, E1, E2 = spectrum(sol, context)
    context.hdf5file.fname = "NS_isotropic_{}_{}_{}.h5".format(*config.params.M)
    context.hdf5file.f = h5py.File(context.hdf5file.fname, driver='mpio', comm=sol.comm)
    context.hdf5file._init_h5file(config.params, sol)
    context.hdf5file.f.create_group("Turbulence")
    context.hdf5file.f["Turbulence"].create_group("Ek")
    bins = np.array(bins)
    context.hdf5file.f["Turbulence"].create_dataset("bins", data=bins)
    context.hdf5file.f.close()
    solve(sol, context)
