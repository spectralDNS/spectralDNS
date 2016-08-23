"""Orr-Sommerfeld"""
from spectralDNS import config, get_solver, solve
from OrrSommerfeld_eig import OrrSommerfeld
from numpy import dot, real, pi, exp, sum, zeros, arange, imag, sqrt, array, \
    zeros_like, allclose
from mpiFFT4py import dct

eps = 1e-6
def initOS(OS, U, X, t=0.):
    for i in range(U.shape[1]):
        x = X[0, i, 0, 0]
        OS.interp(x)
        for j in range(U.shape[2]):
            y = X[1, i, j, 0]
            v = (1-x**2) + eps*dot(OS.f, real(OS.dphidy*exp(1j*(y-OS.eigval*t))))
            u = -eps*dot(OS.f, real(1j*OS.phi*exp(1j*(y-OS.eigval*t))))  
            U[0, i, j, :] = u
            U[1, i, j, :] = v
    U[2] = 0

OS, e0 = None, None
def initialize(solver, U, U_hat, U_hat0, H_hat1, FST, SB, ST, X, mat,
               la, work, K, **context):
    global OS, e0
    params = config.params
    OS = OrrSommerfeld(Re=params.Re, N=160)
    initOS(OS, U, X)

    U_hat0 = solver.set_velocity(U_hat0, U, FST, ST, SB)
    U = solver.get_velocity(U, U_hat0, FST, ST, SB)

    conv = solver.ComputeRHS()
    H_hat1 = conv.nonlinear(H_hat1, U_hat0, context['g'], K, FST, SB, ST, work, mat, la)

    e0 = 0.5*FST.dx(U[0]**2+(U[1]-(1-X[0]**2))**2, ST.quad)
    initOS(OS, U, X, t=params.dt)
    U_hat = solver.set_velocity(U_hat, U, FST, ST, SB)
    U = solver.get_velocity(U, U_hat, FST, ST, SB)

    if not params.solver in ("KMM", "KMMRK3"):  
        pass
        #conv2 = zeros_like(H_hat1)
        #conv2 = conv(conv2, 0.5*(U_hat0+U_hat))  
        #for j in range(3):
            #conv2[j] = la.TDMASolverD(conv2[j])
        #conv2 *= -1
        #kw['P_hat'] = solvePressure(kw['P_hat'], conv2)

        #kw['P'] = FST.ifst(kw['P_hat'], kw['P'], kw['SN'])
        #U_hat0[:] = U_hat
        #params.t = params.dt
        #params.tstep = 1
        
    else:
        U_hat0[:] = U_hat
        params.t = params.dt
        params.tstep = 1
        context['g'][:] = 0

def set_Source(Source, Sk, FST, ST, **kw):
    Source[:] = 0
    Source[1] = -2./config.params.Re
    Sk[:] = 0
    Sk[1] = FST.fss(Source[1], Sk[1], ST)
        
def regression_test(context):
    global OS, e0
    params = config.params
    solver = config.solver
    U = solver.get_velocity(**context)
    initOS(OS, context.U0, context.X, t=params.t)
    pert = (U[0] - context.U0[0])**2 + (U[1]-context.U0[1])**2
    e1 = 0.5*context.FST.dx(pert, context.ST.quad)
    if solver.rank == 0:
        assert sqrt(e1) < 1e-12

if __name__ == "__main__":
    config.update(
        {
        'Re': 8000.,
        'nu': 1./8000.,             # Viscosity
        'dt': 0.001,                 # Time step
        'T': 0.01,                   # End time
        'L': [2, 2*pi, 4*pi/3.],
        'M': [7, 6, 2]
        },  "channel"
    )
    config.channel.add_argument("--compute_energy", type=int, default=1)
    config.channel.add_argument("--plot_step", type=int, default=1)
    solver = get_solver(regression_test=regression_test, mesh="channel")    
    context = solver.setup()
    initialize(solver, **context)
    set_Source(**context)
    solve(solver, context)
