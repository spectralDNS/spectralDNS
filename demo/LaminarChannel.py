"""Orr-Sommerfeld"""
from numpy.polynomial import chebyshev as n_cheb
from cbcdns import config, get_solver
from numpy import dot, real, pi, exp, sum, zeros, cos, exp, arange, imag, sqrt, array
from cbcdns.fft.wrappyfftw import dct
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


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

def initialize(U, U_hat, U0, U_hat0, P, P_hat, **kw):  
    U0[:] = 0
    U_hat0[:] = 0
    P[:] = 0
    P_hat[:] = 0

def set_Source(Source, Sk, fss, ST, **kw):
    Source[:] = 0
    Source[1, :] = -2./config.Re
    Sk[:] = 0
    for i in range(3):
        Sk[i] = fss(Source[i], Sk[i], ST)


def exact(x, Re, t, num_terms=400):
    beta = 2./Re
    u = zeros(len(x))
    for i in range(1, 2*num_terms, 2):
        lam_k = (2*i-1)*pi/2. 
        lam_kp = (2*(i+1)-1)*pi/2. 
        u[:] -= cos(lam_k*x)*exp(-config.nu*lam_k**2*t)/lam_k**3
        u[:] += cos(lam_kp*x)*exp(-config.nu*lam_kp**2*t)/lam_kp**3
    u *= (2*beta)/config.nu
    u += beta/2./config.nu*(1-x**2)
    return u

def reference(Re, t, num_terms=200):
    u = 1.0
    c = 1.0
    for n in range(1, 2*num_terms, 2):
        a = 32. / (pi**3*n**3)
        b = (0.25/Re)*pi**2*n**2
        c = -c
        u += a*exp(-b*t)*c
    return u

def update(rank, X, U, P, N, comm, L, points, ST, num_processes, **kw):
    global im1
    if config.tstep == 2 and rank == 0 and config.plot_step > 0:
        plt.figure()
        im1 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[1,:,:,0], 100)
        plt.colorbar(im1)
        plt.draw()

        plt.pause(1e-6)
        globals().update(im1=im1)
    
    if config.tstep % config.plot_step == 0 and config.plot_step > 0:
        im1.ax.clear()
        im1.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[1, :, :, 0], 100) 
        im1.autoscale()
        plt.pause(1e-6)                

    if config.tstep % config.compute_energy == 0: 
        u0 = U[1, :, 0, 0].copy()
        uall = None
        if rank == 0:
            uall = zeros((num_processes, N[0]/num_processes))
        comm.Gather(u0, uall, root=0)
        if rank == 0:
            uall = uall.reshape((N[0],))
            #x = points
            #pc = zeros(len(x))
            #pc = ST.fct(uall, pc)  # Cheb transform of result
            #solution at x = 0
            #u = n_cheb.chebval(0, pc)
            u_exact = exact(points, config.Re, config.t)
            #print u_exact-uall
            #u_exact = reference(config.Re, config.t)
            print "Time %2.5f Error %2.12e " %(config.t, sqrt(sum((u_exact-uall)**2)/N[0]))

def regression_test(U, X, N, comm, rank, L, ST, num_processes, points, **kw):
    u0 = U[1, :, 0, 0].copy()
    uall = None
    if rank == 0:
        uall = zeros((num_processes, N[0]/num_processes))
    comm.Gather(u0, uall, root=0)
    if rank == 0:
        uall = uall.reshape((N[0],))
        #x = points
        #pc = zeros(len(x))
        #pc = ST.fct(uall, pc)  # Cheb transform of result
        #solution at x = 0
        #u = n_cheb.chebval(0, pc)
        #u_exact = reference(config.Re, config.t)
        u_exact = exact(points, config.Re, config.t)
        print "Computed error = %2.8e %2.8e " %(sqrt(sum((uall-u_exact)**2)/N[0]), config.dt)

if __name__ == "__main__":
    config.update(
        {
        'solver': 'IPCS',
        'Re': 800.,
        'nu': 1./800.,             # Viscosity
        'dt': 0.5,                 # Time step
        'T': 50.,                   # End time
        'L': [2, 2*pi, 4*pi/3.],
        'M': [6, 5, 1]
        },  "Shen"
    )
    config.Shen.add_argument("--compute_energy", type=int, default=5)
    config.Shen.add_argument("--plot_step", type=int, default=10)
    solver = get_solver(update=update, regression_test=regression_test, family="Shen")    
    initialize(**vars(solver))
    set_Source(**vars(solver))
    solver.solve()
