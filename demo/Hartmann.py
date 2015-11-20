"""Orr-Sommerfeld"""
from cbcdns import config, get_solver
from numpy import dot, real, pi, exp, sum, zeros, arange, imag, sqrt, cosh, sinh, linalg, inf
from cbcdns.fft.wrappyfftw import dct
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


def init(U, U_hat, X, fst, ifst, ST, SN, t=0.):
    Ha = config.Ha
    B_strength = config.B_strength
    for i in range(U.shape[1]):
        x = X[0, i, 0, 0]
        for j in range(U.shape[2]):
            y = X[1, i, j, 0]
            u = 0. 
            v = (cosh(Ha)-cosh(Ha*x))/(cosh(Ha)-1.0)
            Bx = (sinh(Ha*x)-Ha*x*cosh(Ha))/(Ha**2*cosh(Ha))
            By = B_strength
            U[0, i, j, :] = u
            U[1, i, j, :] = v
            U[3, i, j, :] = Bx
            U[4, i, j, :] = By
    U[2] = 0
    U[5] = 0
    for i in range(6):
	if i<3:
            U_hat[i] = fst(U[i], U_hat[i], ST)
        else:
	    U_hat[i] = fst(U[i], U_hat[i], SN)
        
    for i in range(6):
	if i<3:
	    U[i] = ifst(U_hat[i], U[i], ST)
	else:
	    U[i] = ifst(U_hat[i], U[i], SN)
	    
    for i in range(6):
	if i<3:
            U_hat[i] = fst(U[i], U_hat[i], ST)
        else:
	    U_hat[i] = fst(U[i], U_hat[i], SN)


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

def initialize(U, U_hat, U0, U_hat0, P, P_hat, conv1, fst, 
               ifst, ST, SN, X, N, comm, rank, L, standardConvection, dt, **kw):        


    init(U0, U_hat0, X, fst, ifst, ST, SN)   
    conv1 = standardConvection(conv1)
    init(U, U_hat, X, fst, ifst, ST, SN, t=dt) 
    P[:] = 0
    P_hat = fst(P, P_hat, SN)
    U0[:] = U
    U_hat0[:] = U_hat
    config.t = dt
    config.tstep = 1
    return dict()

def set_Source(Source, Sk, fss, ST, **kw):
    Source[:] = 0
    Source[1, :] = -2./config.Re
    Sk[:] = 0
    for i in range(3):
        Sk[i] = fss(Source[i], Sk[i], ST)

def update(rank, X, U, P, N, comm, L, **kw):
    
    Ha = config.Ha
    if config.tstep % config.compute_energy == 0: 
	u_exact = ( cosh(Ha) - cosh(Ha*X[0,:,0,0]))/(cosh(Ha) - 1.0)
	if rank == 0:
            print "Time %2.5f Error %2.12e" %(config.t, linalg.norm(u_exact-U[1,:,0,0],inf))

def regression_test(U, X, N, comm, rank, L, fst, ifst,ST, U0, U_hat0,**kw):
    Ha = config.Ha
    u_exact = ( cosh(Ha) - cosh(Ha*X[0,:,0,0]))/(cosh(Ha) - 1.0)
    if rank == 0:
        print "Time %2.5f Error %2.12e" %(config.t, linalg.norm(u_exact-U[1,:,0,0],inf))
        plt.plot(X[0,:,0,0], U[1,:,0,0], X[0,:,0,0], u_exact,'*r')
        plt.show()

if __name__ == "__main__":
    config.update(
        {
        'solver': 'IPCS_MHD',
        'Re': 8000.,
        'Rm': 600.,
        'nu': 1./8000.,             # Viscosity
        'eta': 1./600.,             # Resistivity
        'dt': 0.001,                # Time step
        'T': 0.01,                  # End time
        'B_strength': 0.000001,
        'Ha': 0.0043817804600413289,
        'L': [2, 2*pi, 4*pi/3.],
        'M': [7, 6, 1]
        },  "ShenMHD"
    )
	
    config.ShenMHD.add_argument("--compute_energy", type=int, default=1)
    config.ShenMHD.add_argument("--plot_step", type=int, default=10)
    solver = get_solver(update=update, regression_test=regression_test, family="ShenMHD")
    vars(solver).update(initialize(**vars(solver)))
    set_Source(**vars(solver))
    solver.solve()
