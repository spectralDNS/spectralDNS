"""Orr-Sommerfeld"""
from spectralDNS import config, get_solver
from OrrSommerfeld_eig import OrrSommerfeld
from numpy import dot, real, pi, exp, sum, zeros, arange, imag, sqrt, array, zeros_like, allclose
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
def initialize(U, U_hat, U0, U_hat0, solvePressure, H_hat1, FST,
               ST, X, comm, rank, conv, TDMASolverD, params, 
               forward_velocity, backward_velocity, **kw): 
    global OS, e0
    OS = OrrSommerfeld(Re=params.Re, N=100)
    initOS(OS, U0, X)
    U_hat0 = forward_velocity(U_hat0, U, FST)
    U = backward_velocity(U, U_hat0, FST)
    H_hat1 = conv(H_hat1, U_hat0)
    e0 = 0.5*FST.dx(U[0]**2+(U[1]-(1-X[0]**2))**2, ST.quad)
    initOS(OS, U, X, t=params.dt)
    U_hat = forward_velocity(U_hat, U, FST)
    U = backward_velocity(U, U_hat, FST)
    
    if not params.solver in ("KMM", "KMMRK3"):
        conv2 = zeros_like(H_hat1)
        conv2 = conv(conv2, 0.5*(U_hat0+U_hat))  
        for j in range(3):
            conv2[j] = TDMASolverD(conv2[j])
        conv2 *= -1
        kw['P_hat'] = solvePressure(kw['P_hat'], conv2)
        kw['P'] = FST.ifst(kw['P_hat'], kw['P'], kw['SN'])

        U0[:] = U
        U_hat0[:] = U_hat
        params.t = params.dt
        params.tstep = 1

    else:
        U0[:] = U
        U_hat0[:] = U_hat
        params.t = params.dt
        params.tstep = 1
        kw['g'][:] = 0
    
def set_Source(Source, Sk, FST, ST, params, **kw):
    Source[:] = 0
    Source[1] = -2./params.Re
    Sk[:] = 0
    Sk[1] = FST.fss(Source[1], Sk[1], ST)
        
def regression_test(X, rank, FST, U0, U, U_hat, params, 
                    backward_velocity, **kw):
    global OS, e0
    U = backward_velocity(U, U_hat, FST)
    initOS(OS, U0, X, t=params.t)
    pert = (U[0] - U0[0])**2 + (U[1]-U0[1])**2
    e1 = 0.5*FST.dx(pert, kw['ST'].quad)
    if rank == 0:
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
    initialize(**vars(solver))
    set_Source(**vars(solver))	
    solver.solve()
