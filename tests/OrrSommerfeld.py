"""Orr-Sommerfeld"""
from cbcdns import config, get_solver
from OrrSommerfeld_eig import OrrSommerfeld
from numpy import dot, real, pi, exp, sum, zeros, arange, imag, sqrt, array, zeros_like, allclose
from mpiFFT4py import dct

eps = 1e-6
def initOS(OS, U, U_hat, X, t=0.):
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

def energy(u, N, comm, rank, L):
    uu = sum(u, axis=(1,2))
    c = zeros(N[0])
    comm.Gather(uu, c)
    if rank == 0:
        ak = 1./(N[0]-1)*dct(c, 1, axis=0)
        w = arange(0, N[0], 1, dtype=float)
        w[2:] = 2./(1-w[2:]**2)
        w[0] = 1
        w[1::2] = 0
        return sum(ak*w)*L[1]*L[2]/N[1]/N[2]
    else:
        return 0    

def initialize(U, U_hat, U0, U_hat0, P, P_hat, solvePressure, H_hat1, FST, U_tmp,
               ST, X, N, comm, rank, L, conv, TDMASolverD, F_tmp, H, H1, **kw):        
    OS = OrrSommerfeld(Re=config.Re, N=100)
    initOS(OS, U0, U_hat0, X)
    
    if not config.solver in ("KMM", "KMMRK3"):
        for i in range(3):
            U_hat0[i] = FST.fst(U0[i], U_hat0[i], ST)        
        for i in range(3):
            U0[i] = FST.ifst(U_hat0[i], U0[i], ST)
        for i in range(3):
            U_hat0[i] = FST.fst(U0[i], U_hat0[i], ST)        
        H_hat1 = conv(H_hat1, U0, U_hat0)
        H1[:] = H[:]
        e0 = 0.5*energy(U0[0]**2+(U0[1]-(1-X[0]**2))**2, N, comm, rank, L)    

        initOS(OS, U, U_hat, X, t=config.dt)
        for i in range(3):
            U_hat[i] = FST.fst(U[i], U_hat[i], ST)        
        for i in range(3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)
        for i in range(3):
            U_hat[i] = FST.fst(U[i], U_hat[i], ST)        

        conv2 = zeros_like(H_hat1)
        conv2 = conv(conv2, U0, 0.5*(U_hat0+U_hat))  
        for j in range(3):
            conv2[j] = TDMASolverD(conv2[j])
        conv2 *= -1
        P_hat = solvePressure(P_hat, conv2)

        P = FST.ifst(P_hat, P, kw['SN'])
        U0[:] = U
        U_hat0[:] = U_hat
        config.t = config.dt
        config.tstep = 1

    else:
        U_hat0[0] = FST.fst(U0[0], U_hat0[0], kw['SB']) 
        for i in range(1, 3):
            U_hat0[i] = FST.fst(U0[i], U_hat0[i], ST)        
        U0[0] = FST.ifst(U_hat0[0], U0[0], kw['SB'])
        for i in range(1, 3):
            U0[i] = FST.ifst(U_hat0[i], U0[i], ST)
        H_hat1 = conv(H_hat1, U0, U_hat0)
        H1[:] = H[:]
        e0 = 0.5*energy(U0[0]**2+(U0[1]-(1-X[0]**2))**2, N, comm, rank, L)    
        
        initOS(OS, U, U_hat, X, t=config.dt)
        U_hat[0] = FST.fst(U[0], U_hat[0], kw['SB']) 
        for i in range(1, 3):
            U_hat[i] = FST.fst(U[i], U_hat[i], ST)        
        U[0] = FST.ifst(U_hat[0], U[0], kw['SB'])
        for i in range(1, 3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)

        U0[:] = U
        U_hat0[:] = U_hat
        config.t = config.dt
        config.tstep = 1
        kw['g'][:] = 0
    
    return dict(OS=OS, e0=e0)

def set_Source(Source, Sk, FST, ST, **kw):
    Source[:] = 0
    Source[1] = -2./config.Re
    Sk[:] = 0
    Sk[1] = FST.fss(Source[1], Sk[1], ST)
        
def regression_test(X, OS, N, comm, rank, L, e0, FST, U0, U_hat0, U, U_hat, **kw):
    if "KMM" in config.solver:
        U[0] = FST.ifst(U_hat[0], U[0], kw["SB"])
        for i in range(1, 3):
            U[i] = FST.ifst(U_hat[i], U[i], kw["ST"])
    
    initOS(OS, U0, U_hat0, X, t=config.t)
    pert = (U[0] - U0[0])**2 + (U[1]-U0[1])**2
    e1 = 0.5*energy(pert, N, comm, rank, L)
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
    vars(solver).update(initialize(**vars(solver)))
    set_Source(**vars(solver))	
    solver.solve()
    
