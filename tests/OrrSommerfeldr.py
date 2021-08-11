"""Orr-Sommerfeld"""
import warnings
from numpy import real, pi, exp, zeros, imag, sqrt, log10
from spectralDNS import config, get_solver, solve
from spectralDNS.utilities import dx
#from spectralDNS.utilities import reset_profile
from OrrSommerfeld_shen import OrrSommerfeld

try:
    import matplotlib.pyplot as plt
    import matplotlib.cbook
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

except ImportError:
    warnings.warn("matplotlib not installed")
    plt = None

#params.eps = 1e-9

def initOS(OS, eigvals, eigvectors, U, X, t=0.):
    x = X[2][0, 0].copy()
    eigval, phi, dphidy = OS.interp(x, eigvals, eigvectors, eigval=1, verbose=False)
    OS.eigval = eigval
    for j in range(U.shape[1]):
        y = X[0][j, 0, 0]
        v = (1-x**2) + config.params.eps*real(dphidy*exp(1j*(y-eigval*t)))
        u = -config.params.eps*real(1j*phi*exp(1j*(y-eigval*t)))
        U[2, j, :, :] = u.repeat(U.shape[2]).reshape((len(x), U.shape[2])).T
        U[0, j, :, :] = v.repeat(U.shape[2]).reshape((len(x), U.shape[2])).T
    U[1] = 0

acc = zeros(1)
OS, e0 = None, None
def initialize(solver, context):
    global OS, e0
    params = config.params
    OS = OrrSommerfeld(Re=params.Re, N=128)
    eigvals, eigvectors = OS.solve(False)
    OS.eigvals, OS.eigvectors = eigvals, eigvectors
    U = context.U
    X = context.X
    FST = context.FST
    initOS(OS, eigvals, eigvectors, U, X)

    U_hat = solver.set_velocity(**context)
    U = solver.get_velocity(**context)

    # Compute convection from data in context (i.e., context.U_hat and context.g)
    # This is the convection at t=0
    e0 = 0.5*dx(U[2]**2+(U[0]-(1-X[2]**2))**2, context.FST, axis=2)
    #print(e0)
    acc[0] = 0.0

    if 'KMMRK3' not in params.solver:
        # Initialize at t = dt
        context.H_hat1[:] = solver.get_convection(**context)
        initOS(OS, eigvals, eigvectors, U, X, t=params.dt)
        U_hat = solver.set_velocity(**context)
        U = solver.get_velocity(**context)
        context.U_hat0[:] = U_hat
        params.t = params.dt
        params.tstep = 1
        e1 = 0.5*dx(U[2]**2+(U[0]-(1-X[2]**2))**2, context.FST, axis=2)

        if solver.rank == 0:
            acc[0] += abs(e1/e0 - exp(2*imag(OS.eigval)*params.t))

    else:
        params.t = 0
        params.tstep = 0

    if not "KMM" in params.solver:
        P_hat = solver.compute_pressure(**context)
        FST.backward(P_hat, context.P, context.SN)

    else:
        context.g[:] = 0

def set_Source(Source, Sk, FST, ST, N, **kw):
    Source[:] = 0
    Source[0] = -2./config.params.Re
    Sk[:] = 0
    if hasattr(FST, 'complex_shape'):
        Sk[0] = FST.scalar_product(Source[0], Sk[0], ST)

    else:
        Sk[0] = FST.scalar_product(Source[0], Sk[0])
    Sk[0, 0, 0, -2:] = 0

im1, im2, im3, im4 = (None, )*4
def update(context):

    c = context
    params = config.params
    solver = config.solver

    #if params.tstep == 2: reset_profile(profile)

    if (params.tstep % params.plot_step == 0 or
            params.tstep % params.compute_energy == 0):
        U = solver.get_velocity(**context)

    global im1, im2, im3, OS, e0, acc
    if not plt is None:
        if im1 is None and solver.rank == 0 and params.plot_step > 0:
            plt.figure()
            im1 = plt.contourf(c.X[0][:, 0, :], c.X[2][:, 0, :], c.U[2, :, 0, :], 100)
            plt.colorbar(im1)
            plt.draw()

            plt.figure()
            im2 = plt.contourf(c.X[0][:, 0, :], c.X[2][:, 0, :], c.U[0, :, 0, :] - (1-c.X[2][:, 0, :]**2), 100)
            plt.colorbar(im2)
            plt.draw()

            plt.figure()
            im3 = plt.quiver(c.X[0][:, 0, :], c.X[2][:, 0, :], c.U[0, :, 0, :]-(1-c.X[2][:, 0, :]**2), c.U[2, :, 0, :])
            plt.draw()

            plt.pause(1e-6)

        if params.tstep % params.plot_step == 0 and solver.rank == 0 and params.plot_step > 0:
            im1.ax.clear()
            im1.ax.contourf(c.X[0][:, 0, :], c.X[2][:, 0, :], U[2, :, 0, :], 100)
            im1.autoscale()
            im2.ax.clear()
            im2.ax.contourf(c.X[0][:, 0, :], c.X[2][:, 0, :], U[0, :, 0, :]-(1-c.X[2][:, 0, :]**2), 100)
            im2.autoscale()
            im3.set_UVC(U[0, :, 0, :]-(1-c.X[2][:, 0, :]**2), U[2, :, 0, :])
            plt.pause(1e-6)

    if params.tstep % params.compute_energy == 0:
        e1, e2, exact = compute_error(c)
        div_u = solver.get_divergence(**c)
        e3 = dx(div_u**2, c.FST, axis=2)
        if solver.rank == 0 and not config.params.spatial_refinement_test:
            acc[0] += abs(e1/e0-exact)
            print("Time %2.5f Norms %2.16e %2.16e %2.16e %2.16e %2.16e" %(params.t, e1/e0, exact, e1/e0-exact, sqrt(e2), e3))

def compute_error(context):
    global OS, e0, acc
    c = context
    params = config.params
    solver = config.solver
    U = solver.get_velocity(**c)
    pert = (U[0] - (1-c.X[2]**2))**2 + U[2]**2

    e1 = 0.5*dx(pert, c.FST, axis=2)

    exact = exp(2*imag(OS.eigval)*params.t)
    U0 = c.work[(U, 0, True)]
    initOS(OS, OS.eigvals, OS.eigvectors, U0, c.X, t=params.t)
    pert = (U[0] - U0[0])**2 + (U[2]-U0[2])**2
    #pert = (U[2] - U0[2])**2
    e2 = 0.5*dx(pert, c.FST, axis=2)

    return e1, e2, exact

def regression_test(context):
    _, e2, _ = compute_error(context)
    if config.solver.rank == 0:
        assert sqrt(e2) < 1e-12

def refinement_test(context):
    _, e2, _ = compute_error(context)
    if config.solver.rank == 0:
        print("Computed error = %2.8e %2.8e %2.8e" %(sqrt(e2)/config.params.eps, config.params.dt, config.params.eps))

def eps_refinement_test(context):
    e1, e2, exact = compute_error(context)
    if config.solver.rank == 0:
        print(r" %2d & %2.8e & %2.8e \\\ " %(-int(log10(config.params.eps)), sqrt(e2)/config.params.eps, e1/e0-exact))

def spatial_refinement_test(context):
    _, e2, _ = compute_error(context)
    if config.solver.rank == 0:
        print(r" %2d & %2.8e & %2.8e \\\ " %(2**config.params.M[0], sqrt(e2)/config.params.eps, acc[0]))

if __name__ == "__main__":
    config.update(
        {'Re': 8000.,
         'nu': 1./8000.,              # Viscosity
         'dt': 0.001,                 # Time step
         'T': 0.01,                   # End time
         'L': [2*pi, pi, 2],
         'M': [5, 2, 7],
         'Dquad': 'GC',
         'Bquad': 'GC',
         'dealias': None
        }, "channel"
    )
    config.channel.add_argument("--compute_energy", type=int, default=1)
    config.channel.add_argument("--plot_step", type=int, default=1)
    config.channel.add_argument("--refinement_test", type=bool, default=False)
    config.channel.add_argument("--eps_refinement_test", type=bool, default=False)
    config.channel.add_argument("--spatial_refinement_test", type=bool, default=False)
    config.channel.add_argument("--eps", type=float, default=1e-7)
    #solver = get_solver(update=update, regression_test=regression_test, mesh="channel")
    solver = get_solver(update=update, mesh="channel")

    if config.params.eps_refinement_test:
        print("eps refinement-test")
        solver.update = lambda x: None
        solver.regression_test = eps_refinement_test
        config.params.verbose = False
        context = solver.get_context()

        for eps in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]:
            config.params.eps = eps
            initialize(solver, context)
            set_Source(**context)
            solve(solver, context)

    elif config.params.spatial_refinement_test:
        print("spatial refinement-test")
        def update_(con):
            e1, _, exact = compute_error(con)
            acc[0] += abs(e1/e0-exact)
        solver.update = update_
        solver.regression_test = spatial_refinement_test
        config.params.verbose = False
        for M in [4, 5, 6, 7, 8]:
            config.params.M = [M, 3, 1]
            context = solver.get_context()
            initialize(solver, context)
            set_Source(**context)
            solve(solver, context)

    else:
        if config.params.refinement_test:
            solver.update = lambda x: None
            solver.regression_test = refinement_test
        context = solver.get_context()
        initialize(solver, context)
        set_Source(**context)
        solve(solver, context)
