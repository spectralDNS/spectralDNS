"""Turbulent channel"""
from spectralDNS import config, get_solver
from numpy import dot, real, pi, exp, sum, complex, float, zeros, arange, imag, cos, where, pi, random, exp, sin, log, array, zeros_like
import h5py
from mpiFFT4py import dct
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
from OrrSommerfeld_eig import OrrSommerfeld
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# Use constant flux and adjust pressure gradient dynamically
#flux = array([1645.46])
flux = array([736.43])

def initOS(OS, U, X, t=0.):
    for i in range(U.shape[1]):
        x = X[0, i, 0, 0]
        OS.interp(x)
        for j in range(U.shape[2]):
            y = X[1, i, j, 0]
            v =  dot(OS.f, real(OS.dphidy*exp(1j*(y-OS.eigval*t))))
            u = -dot(OS.f, real(1j*OS.phi*exp(1j*(y-OS.eigval*t))))  
            U[0, i, j, :] = u
            U[1, i, j, :] = v
    U[2] = 0

def initialize(U, U_hat, U0, U_hat0, P, P_hat, FST, ST, X, comm, rank, num_processes, 
               Curl, conv, TDMASolverD, solvePressure, N, H_hat, H_hat1, K, **kw):
    # Initialize with pertubation ala perturbU (https://github.com/wyldckat/perturbU) for openfoam
    Y = where(X[0]<0, 1+X[0], 1-X[0])
    utau = config.nu * config.Re_tau
    #Um = 46.9091*utau
    Um = 56.*utau 
    Xplus = Y*config.Re_tau
    Yplus = X[1]*config.Re_tau
    Zplus = X[2]*config.Re_tau
    duplus = Um*0.2/utau  #Um*0.25/utau 
    alfaplus = config.L[1]/500.
    betaplus = config.L[2]/200.
    sigma = 0.00055 # 0.00055
    epsilon = Um/200.   #Um/200.
    U[:] = 0
    U[1] = Um*(Y-0.5*Y**2)
    dev = 1+0.0005*random.randn(Y.shape[0], Y.shape[1], Y.shape[2])
    dd = utau*duplus/2.0*Xplus/40.*exp(-sigma*Xplus**2+0.5)*cos(betaplus*Zplus)*dev
    U[1] += dd
    U[2] += epsilon*sin(alfaplus*Yplus)*Xplus*exp(-sigma*Xplus**2)*dev    
    #U[0] = 0.00001*random.randn(Y.shape[0], Y.shape[1], Y.shape[2])

    #U[:] = 0
    #U[:] = 0.001*random.randn(*U.shape)
    #for i in range(3):
        #U_hat[i] = FST.fst(U[i], U_hat[i], ST)    
    #U = Curl(U_hat, U, ST)
    #U[1] += Um*(Y-0.5*Y**2)
    #U[1] += utau*duplus/2.0*Xplus/40.*exp(-sigma*Xplus**2+0.5)*cos(betaplus*Zplus)
    #U[2] += epsilon*sin(alfaplus*Yplus)*Xplus*exp(-sigma*Xplus**2)
        
    #U[0] += 0.1*(1-X[0]**2)*sin(4*pi*X[0])*cos(2*pi*X[1])
    #U[1] += 0.1*((1-X[0]**2)*sin(2*pi*X[1])+2*X[0]*sin(4*pi*X[0])*sin(2*pi*X[1])/2./pi)
    
    #OS = OrrSommerfeld(Re=6000, N=80)
    #initOS(OS, U0, X)
    #U[1] += 5e-2*U0[1]
    #U[2] += 5e-2*U0[2]
    
    if "KMM" in config.solver:
        U_hat[0] = FST.fst(U[0], U_hat[0], kw['SB'])
        for i in range(1, 3):
            U_hat[i] = FST.fst(U[i], U_hat[i], ST)
            
        U[0] = FST.ifst(U_hat[0], U[0], kw['SB'])
        for i in range(1, 3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)

        U_hat[0] = FST.fst(U[0], U_hat[0], kw['SB'])
        for i in range(1, 3):
            U_hat[i] = FST.fst(U[i], U_hat[i], ST)

        g = kw['g']
        g[:] = 1j*K[1]*U_hat[2] - 1j*K[2]*U_hat[1]

    else:
        # project to Dirichlet space and back
        for i in range(3):
            U_hat[i] = FST.fst(U[i], U_hat[i], ST)
            
        for i in range(3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)

        for i in range(3):
            U_hat[i] = FST.fst(U[i], U_hat[i], ST)

    # Set the flux
    flux[0] = Q(U[1], rank, comm, N)
    comm.Bcast(flux)
    
    if rank == 0:
        print "Flux", flux[0]
    
    if not config.solver in ("KMM", "KMMRK3"):
        conv2 = zeros_like(U_hat)
        conv2 = conv(conv2, U, U_hat)  
        for j in range(3):
            conv2[j] = TDMASolverD(conv2[j])
        conv2 *= -1
        P_hat = solvePressure(P_hat, conv2)
        P = FST.ifst(P_hat, P, kw['SN'])
        
    U0[:] = U[:]
    U_hat0[:] = U_hat[:]
    H_hat1 = conv(H_hat1, U0, U_hat0)
 
def init_from_file(filename, comm, U0, U_hat0, U, U_hat, P, P_hat, H_hat1, K,
                   rank, conv, FST, ST, **kw):
    f = h5py.File(filename, driver="mpio", comm=comm)
    assert "0" in f["3D/checkpoint/U"]
    N = U0.shape[1]
    s = slice(rank*N, (rank+1)*N, 1)
    U0[:] = f["3D/checkpoint/U/0"][:, s]
    
    if config.solver in ("IPCS", "IPCSR"):
        for i in range(3):
            U_hat0[i] = FST.fst(U0[i], U_hat0[i], ST)
        H_hat1[:] = conv(H_hat1, U0, U_hat0)
        
        U0[:] = f["3D/checkpoint/U/1"][:, s]
        P [:] = f["3D/checkpoint/P/1"][s]

        for i in range(3):
            U_hat0[i] = FST.fst(U0[i], U_hat0[i], ST)
        
        if config.solver == "IPCSR":
            P_hat = FST.fct(P, P_hat, kw['SN'])
        else:
            P_hat = FST.fst(P, P_hat, kw['SN'])

    elif "KMM" in config.solver:
        U_hat0[0] = FST.fst(U0[0], U_hat0[0], kw['SB'])
        for i in range(1, 3):
            U_hat0[i] = FST.fst(U0[i], U_hat0[i], ST)
        H_hat1[:] = conv(H_hat1, U0, U_hat0)
        
        U0[:] = f["3D/checkpoint/U/1"][:, s]
        U_hat0[0] = FST.fst(U0[0], U_hat0[0], kw['SB'])
        for i in range(1,3):
            U_hat0[i] = FST.fst(U0[i], U_hat0[i], ST)
        
        g = kw['g']
        g[:] = 1j*K[1]*U_hat0[2] - 1j*K[2]*U_hat0[1]
        U_hat[:] = U_hat0[:]        
        
    f.close()

#def init_from_file_with_refinement(filename, comm, U0, U_hat0, U, U_hat, L, MPI,
                                   #P, P_hat, H_hat1, K, rank, conv, FST, ST, **kw):
    #f = h5py.File(filename, driver="mpio", comm=comm)
    #assert "0" in f["3D/checkpoint/U"]
    #N = FST.global_real_shape()
    #s = FST.real_local_slice()
    #N0 = f.attrs["N"]
    #x, y, z = False, False, False
    #if N[0] / N0[0] == 2: x = True
    #if N[1] / N0[1] == 2: y = True
    #if N[2] / N0[2] == 2: z = True
    
    #FST1 = FastShenFourierTransform(N0, L, MPI)
    #W0 = FST1.get_workarray(((3,)+FST1.real_shape(), float), 0)
    #W0[:] = f["3D/checkpoint/U/0"][:, s[0], s[1], s[2]]
    #W0_hat = FST1.get_workarray(((3,)+FST1.complex_shape(), complex), 0)    
    #W0_hat[0] = FST2.fst(W0[0], W0_hat[0], SB)
    #for i in range(1, 3):
        #W0_hat[i] = FST1.fst(W0[i], W0_hat[i], ST)
        
    ## Now use padding
    #U_hat0[:]
    
    
    #U0[:] = f["3D/checkpoint/U/1"][:, s]
    #U_hat0[0] = FST.fst(U0[0], U_hat0[0], kw['SB'])
    #for i in range(1,3):
        #U_hat0[i] = FST.fst(U0[i], U_hat0[i], ST)
    
    #g = kw['g']
    #g[:] = 1j*K[1]*U_hat0[2] - 1j*K[2]*U_hat0[1]
    #U_hat[:] = U_hat0[:]        
        
    #f.close()


def set_Source(Source, Sk, ST, FST, **kw):
    utau = config.nu * config.Re_tau
    Source[:] = 0
    Source[1, :] = -utau**2
    Sk[:] = 0
    Sk[1] = FST.fss(Source[1], Sk[1], ST)
    
def Q(u, rank, comm, N):
    """Integrate u over entire computational domain
    """
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

beta = zeros(1)    
def update(U, U_hat, P, U0, P_hat, rank, X, stats, FST, hdf5file, Source, Sk, 
           ST, SB, comm, N, dU, diff0, hv, work, **kw):
    global im1, im2, im3, flux

    #q = Q(U[1], rank, comm, N)
    #beta[0] = (flux[0] - q)/(array(config.L).prod())
    #comm.Bcast(beta)
    #U_tmp[1] = beta[0]    
    #F_tmp[1] = FST.fst(U_tmp[1], F_tmp[1], ST)
    #U_hat[1] += F_tmp[1]
    #U[1] = FST.ifst(U_hat[1], U[1], ST)
    #Source[1] -= beta[0]
    #Sk[1] = FST.fss(Source[1], Sk[1], ST)
    #utau = config.Re_tau * config.nu
    #Source[:] = 0
    #Source[1] = -utau**2
    #Source[:] += 0.05*random.randn(*U.shape)
    #for i in range(3):
        #Sk[i] = FST.fss(Source[i], Sk[i], ST)
        
    if (hdf5file.check_if_write(config.tstep) or (config.tstep % config.plot_result == 0 and config.plot_result > 0)):
        U[0] = FST.ifst(U_hat[0], U[0], SB)
        for i in range(1, 3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)     
        
    if config.tstep % config.print_energy0 == 0 and rank == 0:
        print (U_hat[0].real*U_hat[0].real).mean(axis=(0, 2))
        print (U_hat[0].real*U_hat[0].real).mean(axis=(0, 1))
    
    if hdf5file.check_if_write(config.tstep):
        hdf5file.write(config.tstep)
        
    if config.tstep % config.checkpoint == 0:
        hdf5file.checkpoint(U, P, U0)

    if config.tstep == 1 and rank == 0 and config.plot_result > 0:
        plt.figure()
        im1 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[0,:,:,0], 100)
        plt.colorbar(im1)
        plt.draw()

        plt.figure()
        im2 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[1,:,:,0], 100)
        plt.colorbar(im2)
        plt.draw()

        plt.figure()
        im3 = plt.contourf(X[2,:,0,:], X[0,:,0,:], U[0, :,0 ,:], 100)
        plt.colorbar(im3)
        plt.draw()

        plt.pause(1e-6)    
        globals().update(im1=im1, im2=im2, im3=im3)
        
    if config.tstep % config.plot_result == 0 and rank == 0 and config.plot_result > 0:
        im1.ax.clear()
        im1.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[0, :, :, 0], 100)         
        im1.autoscale()
        im2.ax.clear()
        im2.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[1, :, :, 0], 100) 
        im2.autoscale()
        im3.ax.clear()
        #im3.ax.contourf(X[1, :,:,0], X[0, :,:,0], P[:, :, 0], 100) 
        im3.ax.contourf(X[2,:,0,:], X[0,:,0,:], U[0, :,0 ,:], 100)
        im3.autoscale()
        plt.pause(1e-6)
    
    if config.tstep % config.compute_energy == 0: 
        # Skip dealiasing even though nonlinear
        U[0] = FST.ifst(U_hat[0], U[0], SB)
        for i in range(1, 3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)     

        e0 = Q(U[0]*U[0], rank, comm, N)
        e1 = Q(U[1]*U[1], rank, comm, N)
        e2 = Q(U[2]*U[2], rank, comm, N)
        q = Q(U[1], rank, comm, N)
        if rank == 0:
            print "Time %2.5f Energy %2.8e %2.8e %2.8e Flux %2.8e Q %2.8e %2.8e" %(config.t, e0, e1, e2, q, e0+e1+e2, Source[1].mean())

    if config.tstep % config.sample_stats == 0:
        stats(U, P)            

class Stats(object):
    
    def __init__(self, U, comm, fromstats="", filename=""):
        self.shape = U.shape[1:]
        self.Umean = zeros(U.shape[:2])
        self.Pmean = zeros(U.shape[1])
        self.UU = zeros((6, U.shape[1]))
        self.num_samples = 0
        self.rank = comm.Get_rank()
        self.fname = filename
        self.comm = comm
        self.f0 = None
        if fromstats:
            self.fromfile(filename=fromstats)
        
    def create_statsfile(self):
        self.f0 = h5py.File(self.fname+".h5", "w", driver="mpio", comm=self.comm)
        self.f0.create_group("Average")
        self.f0.create_group("Reynolds Stress")
        for i in ("U", "V", "W", "P"):
            self.f0["Average"].create_dataset(i, shape=(2**config.M[0],), dtype=float)
            
        for i in ("UU", "VV", "WW", "UV", "UW", "VW"):
            self.f0["Reynolds Stress"].create_dataset(i, shape=(2**config.M[0], ), dtype=float)

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
        self.get_stats()
        
    def get_stats(self, tofile=True):
        N = self.shape[0]
        s = slice(self.rank*N, (self.rank+1)*N, 1)
        Nd = self.num_samples*self.shape[1]*self.shape[2]
        self.comm.barrier()
        if tofile:
            if self.f0 is None:
                self.create_statsfile()
            else:
                self.f0 = h5py.File(self.fname+".h5", "a", driver="mpio", comm=self.comm)
                
            for i, name in enumerate(("U", "V", "W")):
                self.f0["Average/"+name][s] = self.Umean[i]/Nd
            self.f0["Average/P"][s] = self.Pmean/Nd
            for i, name in enumerate(("UU", "VV", "WW", "UV", "UW", "VW")):
                self.f0["Reynolds Stress/"+name][s] = self.UU[i]/Nd
            self.f0.attrs.create("num_samples", self.num_samples)
            self.f0.close()
            
        if self.comm.Get_size() == 1:
            return self.Umean/Nd, self.Pmean/Nd, self.UU/Nd
    
    def reset_stats(self):
        self.num_samples = 0
        self.Umean[:] = 0
        self.Pmean[:] = 0
        self.UU[:] = 0
        
    def fromfile(self, filename="stats"):
        self.fname = filename
        self.f0 = h5py.File(filename+".h5", "a", driver="mpio", comm=self.comm)        
        N = self.shape[0]
        self.num_samples = self.f0.attrs["num_samples"]
        Nd = self.num_samples*self.shape[1]*self.shape[2]        
        s = slice(self.rank*N, (self.rank+1)*N, 1)
        for i, name in enumerate(("U", "V", "W")):
            self.Umean[i, :] = self.f0["Average/"+name][s]*Nd
        self.Pmean[:] = self.f0["Average/P"][s]*Nd
        for i, name in enumerate(("UU", "VV", "WW", "UV", "UW", "VW")):
            self.UU[i, :] = self.f0["Reynolds Stress/"+name][s]*Nd
        self.f0.close()

if __name__ == "__main__":
    config.update(
        {
        'nu': 1./590.,                  # Viscosity
        'Re_tau': 590., 
        'dt': 0.001,                  # Time step
        'T': 100.,                    # End time
        'L': [2, 2*pi, pi],
        'M': [6, 6, 5]
        },  "channel"
    )
    config.channel.add_argument("--compute_energy", type=int, default=10)
    config.channel.add_argument("--plot_result", type=int, default=10)
    config.channel.add_argument("--sample_stats", type=int, default=10)
    config.channel.add_argument("--print_energy0", type=int, default=10)
    solver = get_solver(update=update, mesh="channel")    
    initialize(**vars(solver))    
    #init_from_file("KMM665.h5", **vars(solver))
    set_Source(**vars(solver))
    solver.stats = Stats(solver.U, solver.comm, fromstats="KMMstats")
    solver.hdf5file.fname = "KMM665q.h5"
    solver.solve()
    s = solver.stats.get_stats()

    #from numpy import meshgrid, float
    #s = solver
    #Np = s.N / s.num_processes
    #x1 = arange(1.5*s.N[1], dtype=float)*config.L[1]/(1.5*s.N[1])
    #x2 = arange(1.5*s.N[2], dtype=float)*config.L[2]/(1.5*s.N[2])
    #points, weights = s.ST.points_and_weights(s.N[0])
    ## Get grid for velocity points
    #X = array(meshgrid(points[s.rank*Np[0]:(s.rank+1)*Np[0]], x1, x2, indexing='ij'), dtype=float)    
    #s.U_pad2[1] = s.FST.ifst_padded(s.U_hat[1], s.U_pad2[1], s.ST)
    #plt.figure()
    #plt.contourf(X[1,:,:,0], X[0,:,:,0], s.U_pad2[1,:,:,0], 100)
    #plt.colorbar()
    #plt.show()    
    
