"""Orr-Sommerfeld"""
from cbcdns import config, get_solver
from OrrSommerfeld_eig import OrrSommerfeld
from numpy import dot, real, pi, exp, sum, zeros, arange, imag, sqrt, array, zeros_like, allclose
from cbcdns.fft.wrappyfftw import dct
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

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
               ST, SN, X, N, comm, rank, L, conv, TDMASolverD, F_tmp, H, H1, **kw):        
    OS = OrrSommerfeld(Re=config.Re, N=100)
    initOS(OS, U0, U_hat0, X)
    
    if not config.solver in ("ChannelRK4", "KMM", "KMMRK3"):
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

        P = FST.ifst(P_hat, P, SN)
        U0[:] = U
        U_hat0[:] = U_hat
        config.t = config.dt
        config.tstep = 1

    elif config.solver == "ChannelRK4":        
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
        
im1, im2, im3, im4 = (None, )*4        
def update(rank, X, U, P, OS, N, comm, L, e0, U_tmp, F_tmp, FST, ST, U_hat, **kw):
    global im1, im2, im3, im4
    if im1 is None and rank == 0 and config.plot_step > 0:
        plt.figure()
        im1 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[0,:,:,0], 100)
        plt.colorbar(im1)
        plt.draw()

        plt.figure()
        im2 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[1,:,:,0] - (1-X[0,:,:,0]**2), 100)
        plt.colorbar(im2)
        plt.draw()

        plt.figure()
        im3 = plt.contourf(X[1,:,:,0], X[0,:,:,0], P[:,:,0], 100)
        plt.colorbar(im3)
        plt.draw()
        
        plt.figure()
        im4 = plt.quiver(X[1, :,:,0], X[0,:,:,0], U[1,:,:,0]-(1-X[0,:,:,0]**2), U[0,:,:,0])
        plt.draw()
        
        plt.pause(1e-6)
        globals().update(im1=im1, im2=im2, im3=im3, im4=im4)

    if (config.tstep % config.plot_step == 0 or
        config.tstep % config.compute_energy == 0):
        
        U[0] = FST.ifst(U_hat[0], U[0], kw['SB'])
        for i in range(1, 3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)     

    
    if config.tstep % config.plot_step == 0 and rank == 0 and config.plot_step > 0:
        im1.ax.clear()
        im1.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[0, :, :, 0], 100) 
        im1.autoscale()
        im2.ax.clear()
        im2.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[1, :, :, 0]-(1-X[0,:,:,0]**2), 100)         
        im2.autoscale()
        im3.ax.clear()
        im3.ax.contourf(X[1, :,:,0], X[0, :,:,0], P[:, :, 0], 100) 
        im3.autoscale()
        im4.set_UVC(U[1,:,:,0]-(1-X[0,:,:,0]**2), U[0,:,:,0])
        plt.pause(1e-6)
                

    if config.tstep % config.compute_energy == 0: 
        pert = (U[1] - (1-X[0]**2))**2 + U[0]**2
        e1 = 0.5*energy(pert, N, comm, rank, L)
        exact = exp(2*imag(OS.eigval)*(config.t))
        initOS(OS, U_tmp, F_tmp, X, t=config.t)
        pert = (U[0] - U_tmp[0])**2 + (U[1]-U_tmp[1])**2
        e2 = 0.5*energy(pert, N, comm, rank, L)
        if rank == 0:
            print "Time %2.5f Norms %2.16e %2.16e %2.16e %2.16e" %(config.t, e1/e0, exact, e1/e0-exact, sqrt(e2))

def regression_test(U, X, OS, N, comm, rank, L, e0, FST, ST, U0, U_hat0,**kw):
    #pert = (U[1] - (1-X[0]**2))**2 + U[0]**2
    #e1 = 0.5*energy(pert, N, comm, rank, L)
    #exact = exp(2*imag(OS.eigval)*config.t)
    #if rank == 0:
        #print "Computed error = %2.8e %2.8e " %(sqrt(abs(e1/e0-exact)), config.dt)

    initOS(OS, U0, U_hat0, X, t=config.t)
    pert = (U[0] - U0[0])**2 + (U[1]-U0[1])**2
    e1 = 0.5*energy(pert, N, comm, rank, L)
    #exact = exp(2*imag(OS.eigval)*config.t)
    if rank == 0:
        print "Computed error = %2.8e %2.8e " %(sqrt(e1), config.dt)

def initOS_and_project(OS, U0, U_hat0, X, FST, ST, SB, **kw):
    initOS(OS, U0, U_hat0, X, t=config.t)
    assert "KMM" in config.solver 
    U_hat0[0] = FST.fst(U0[0], U_hat0[0], SB)
    for i in range(1,3):
        U_hat0[i] = FST.fst(U0[i], U_hat0[i], ST)
    U0[0] = FST.ifst(U_hat0[0], U0[0], SB)    
    for i in range(1, 3):
        U0[i] = FST.ifst(U_hat0[i], U0[i], ST)
    U_hat0[0] = FST.fst(U0[0], U_hat0[0], SB)
    for i in range(1, 3):
        U_hat0[i] = FST.fst(U0[i], U_hat0[i], ST)        

if __name__ == "__main__":
    config.update(
        {
        'solver': 'IPCS',
        'Re': 8000.,
        'nu': 1./8000.,             # Viscosity
        'dt': 0.01,                 # Time step
        'T': 0.01,                   # End time
        'L': [2, 2*pi, 4*pi/3.],
        'M': [7, 6, 1]
        },  "Shen"
    )
    config.Shen.add_argument("--compute_energy", type=int, default=1)
    config.Shen.add_argument("--plot_step", type=int, default=1)
    solver = get_solver(update=update, regression_test=regression_test, family="Shen")    
    vars(solver).update(initialize(**vars(solver)))
    set_Source(**vars(solver))	
    solver.solve()
    s = solver
    
    #from numpy import meshgrid, float, allclose
    #s = solver
    #Np = s.N / s.num_processes
    #x1 = arange(1.5*s.N[1], dtype=float)*config.L[1]/(1.5*s.N[1])
    #x2 = arange(1.5*s.N[2], dtype=float)*config.L[2]/(1.5*s.N[2])
    ## Get grid for velocity points
    #X = array(meshgrid(s.points[s.rank*Np[0]:(s.rank+1)*Np[0]], x1, x2, indexing='ij'), dtype=float)    
    #s.U_pad2[0] = s.FST.ifst_padded(s.U_hat[0], s.U_pad2[0], s.SB)
    #s.F_tmp[0] = s.FST.fst_padded(s.U_pad2[0], s.F_tmp[0], s.SB)
    
    #assert allclose(s.F_tmp[0], s.U_hat[0])
    #plt.figure()
    #plt.contourf(X[1,:,:,0], X[0,:,:,0], s.U_pad2[0,:,:,0], 100)
    #plt.show()


