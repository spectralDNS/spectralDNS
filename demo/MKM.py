"""Turbulent channel"""
from cbcdns import config, get_solver
from numpy import dot, real, pi, exp, sum, zeros, arange, imag, cos, where, pi, random, exp, sin
from cbcdns.fft.wrappyfftw import dct
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

def initialize(U, U_hat, U0, U_hat0, P, P_hat, fst, ST, SN, X, comm, rank, num_processes, **kw):
    # Initialize with pertubation ala perturbU (https://github.com/wyldckat/perturbU) for openfoam
    Y = where(X[0]<0, 1+X[0], 1-X[0])
    utau = config.nu * config.Re_tau
    Um = 40*utau
    U[:] = 0
    U[1] = Um*(Y-0.5*Y**2)
    Xplus = Y*config.Re_tau
    Yplus = X[1]*config.Re_tau
    Zplus = X[2]*config.Re_tau
    duplus = Um*0.25/utau 
    alfaplus = 2*pi/500.
    betaplus = 2*pi/200.
    sigma = 0.00055
    epsilon = Um/200.
    dev = 1+0.3*random.randn(Y.shape[0], Y.shape[1], Y.shape[2])
    dd = utau*duplus/2.0*Xplus/40.*exp(-sigma*Xplus**2+0.5)*cos(betaplus*Zplus)*dev
    U[1] += dd
    U[2] += epsilon*sin(alfaplus*Yplus)*Xplus*exp(-sigma*Xplus**2)*dev
    if rank == 0:
        U[:, 0] = 0
    if rank == num_processes-1:
        U[:, -1] = 0

    for i in range(3):
        U_hat[i] = fst(U[i], U_hat[i], ST)

    P[:] = 0
    P_hat = fst(P, P_hat, SN)
    U0[:] = U[:]
    U_hat0[:] = U_hat[:]

def set_Source(Source, Sk, fss, ST, **kw):
    utau = config.nu * config.Re_tau
    Source[:] = 0
    Source[1, :] = -utau**2
    Sk[:] = 0
    Sk[1] = fss(Source[1], Sk[1], ST)
    
    
def Q(u, rank, comm, N, **kw):
    L = config.L
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
    
def update(U, P, rank, X, stats, **kw):
    global im1, im2, im3
    if config.tstep == 1 and rank == 0:
        plt.figure()
        im1 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[0,:,:,0], 100)
        plt.colorbar(im1)
        plt.draw()

        plt.figure()
        im2 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[1,:,:,0], 100)
        plt.colorbar(im2)
        plt.draw()

        plt.figure()
        im3 = plt.contourf(X[1,:,:,0], X[0,:,:,0], P[:,:,0], 100)
        plt.colorbar(im3)
        plt.draw()

        plt.pause(1e-6)    
        globals().update(im1=im1, im2=im2, im3=im3)
        
    if config.tstep % config.plot_result == 0 and rank == 0:
        im1.ax.clear()
        im1.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[0, :, :, 0], 100)         
        im1.autoscale()
        im2.ax.clear()
        im2.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[1, :, :, 0], 100) 
        im2.autoscale()
        im3.ax.clear()
        im3.ax.contourf(X[1, :,:,0], X[0, :,:,0], P[:, :, 0], 100) 
        im3.autoscale()
        plt.pause(1e-6)
    
    if config.tstep % config.compute_energy == 0: 
        e0 = Q(U[0]*U[0], rank, **kw)
        e1 = Q(U[1]*U[1], rank, **kw)
        e2 = Q(U[2]*U[2], rank, **kw)
        flux = Q(U[1], rank, **kw)
        if rank == 0:
            print "Time %2.5f Energy %2.12e %2.12e %2.12e Flux %2.12e" %(config.t, e0, e1, e2, flux)

    if config.tstep % config.sample_stats == 0:
        stats(U, P)
        

class Stats(object):
    
    def __init__(self, U, rank):
        self.shape = U.shape[1:]
        self.Umean = zeros(U.shape[:2])
        self.Pmean = zeros(U.shape[1])
        self.UU = zeros((6, U.shape[1]))
        self.num_samples = 0
        self.rank = rank
    
    def __call__(self, U, P):
        self.num_samples += 1
        self.Umean += sum(U, axis=(2,3))
        self.Pmean += sum(P, axis=(1,2))
        self.UU[0] += sum(U[0]*U[0], axis=(1,2))
        self.UU[1] += sum(U[1]*U[1], axis=(1,2))
        self.UU[2] += sum(U[2]*U[2], axis=(1,2))
        self.UU[3] += sum(U[0]*U[1], axis=(1,2))
        self.UU[4] += sum(U[0]*U[2], axis=(1,2))
        self.UU[5] += sum(U[1]*U[2], axis=(1,2))
        
    def get_stats(self):
        N = self.shape[0]
        u = zeros((3, 2**config.M[0]))
        p = zeros(2**config.M[0])
        uv = zeros((6, 2**config.M[0]))
        s = slice(self.rank*N, (self.rank+1)*N, 1)
        u[:, s] = self.Umean/(self.num_samples*self.shape[1]*self.shape[2])
        p[s] = self.Pmean/(self.num_samples*self.shape[1]*self.shape[2])
        uv[:, s] = self.UU/(self.num_samples*self.shape[1]*self.shape[2])
        return u, p, uv
        

if __name__ == "__main__":
    config.update(
        {
        'solver': 'IPCS',
        'nu': 2e-5,                  # Viscosity
        'Re_tau': 180., 
        'dt': 0.2,                  # Time step
        'T': 10000.,                   # End time
        'L': [2, 4*pi, 4*pi/3.],
        'M': [6, 6, 5]
        },  "Shen"
    )
    config.Shen.add_argument("--compute_energy", type=int, default=100)
    config.Shen.add_argument("--plot_result", type=int, default=100)
    config.Shen.add_argument("--sample_stats", type=int, default=100)
    solver = get_solver(update=update, family="Shen")    
    initialize(**vars(solver))
    set_Source(**vars(solver))
    solver.stats = Stats(solver.U, solver.rank)
    solver.solve()
