__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import time, sys, cProfile

#from numpy import *
#from pylab import *
from numpy import array, meshgrid, linspace, empty, zeros, sin, cos, pi, where, sum, int, float, bool
from pylab import fftfreq, fft2, rfft, ifft, ifft2, irfft
from mpi4py import MPI
from utilities import *
from h5io import *
#from numba import jit, complex128, int64
#import numexpr

comm = MPI.COMM_WORLD

params = {
    'convection': 'Vortex',     # ['Standard', 'Divergence', 'Skewed', 'Vortex']
    'make_profile': 0,          # Enable cProfile profiler
    'M': 5,                     # Mesh size
    'temporal': 'RK4',          # Integrator ("RK4", "ForwardEuler", "AB2")
    'write_result': 1e8,        # Write to HDF5 every..
    'write_yz_slice': [0, 1e8], # Write slice 0 (or higher) in y-z plance every..
    'compute_energy': 2,        # Compute solution energy every..
    'plot_result': 2,           # Show an image every..
    'nu': 0.000625,             # Viscosity
    'dt': 0.01,                 # Time step
    'T': 0.1                    # End time
}

# Parse parameters from the command line 
commandline_kwargs = parse_command_line(sys.argv[1:])
params.update(commandline_kwargs)
assert params['convection'] in ['Standard', 'Divergence', 'Skewed', 'Vortex']
assert params['temporal'] in ['RK4', 'ForwardEuler', 'AB2']
vars().update(params)

N = 2**M
L = 2 * pi
dx = L / N

num_processes = comm.Get_size()
rank = comm.Get_rank()
if not num_processes in [2**i for i in range(M+1)]:
    raise IOError("Number of cpus must be in ", [2**i for i in range(M+1)])

if make_profile: profiler = cProfile.Profile()

# Each cpu gets ownership of Np slices
Np = N / num_processes     

hdf5file = HDF5Writer(comm, dt, N, params)

# Create the physical mesh
x = linspace(0, L, N+1)[:-1]
X = array(meshgrid(x[rank*Np:(rank+1)*Np], x, x, indexing='ij'))

"""
Solution U is real and as such its transform, U_hat = fft(U)(k), 
is such that fft(U)(k) = conj(fft(U)(N-k)) and thus it is sufficient 
to store N/2+1 Fourier coefficients in the first transformed direction (y).
For means of efficient MPI communication, the physical box (N^3) is
shared by processors along the first direction, whereas the Fourier 
coefficients are shared along the third direction. The y-direction
is N/2+1 in Fourier space.
"""

Nf = N/2+1
U     = empty((3, Np, N, N))                  
U_hat = empty((3, N, Nf, Np), dtype="complex")
P     = empty((Np, N, N))
P_hat = empty((N, Nf, Np), dtype="complex")

# Temporal storage arrays (Not required by all temporal integrators)
U_hat0  = empty((3, N, Nf, Np), dtype="complex")
U_hat1  = empty((3, N, Nf, Np), dtype="complex")
dU      = empty((3, N, Nf, Np), dtype="complex")

# work arrays (Not required by all convection methods)
F_tmp   = empty((3, N, Nf, Np), dtype="complex")
U_tmp   = empty((3, Np, N, N))
Uc_hat  = empty((N, Nf, Np), dtype="complex")
Uc_hatT = empty((Np, Nf, N), dtype="complex")
Uc_send = Uc_hat.reshape((num_processes, Np, Nf, Np))
U_mpi   = empty((num_processes, Np, Nf, Np), dtype="complex")
curl    = empty((3, Np, N, N))

# Set wavenumbers in grid
kx = fftfreq(N, 1./N)
ky = kx[:Nf].copy(); ky[-1] *= -1
KX = array(meshgrid(kx, ky, kx[rank*Np:(rank+1)*Np], indexing='ij'), dtype=int)
KK = sum(KX*KX, 0)
KX_over_Ksq = array(KX, dtype=float) / where(KK==0, 1, KK)

# Filter for dealiasing nonlinear convection
kmax = 2./3.*(N/2+1)
dealias = array((abs(KX[0]) < kmax)*(abs(KX[1]) < kmax)*
                (abs(KX[2]) < kmax), dtype=bool)

# RK4 parameters
a = [1./6., 1./3., 1./3., 1./6.]
b = [0.5, 0.5, 1.]

def project(u):
    """Project u onto divergence free space"""
    u[:] -= sum(KX_over_Ksq*u, 0)*KX

#@profile
def ifftn_mpi(fu, u):
    """ifft in three directions using mpi.
    Need to do ifft in reversed order of fft
    """
    if num_processes == 1:
        u[:] = irfft(ifft(ifft(fu, axis=0), axis=2), axis=1)
        return
    
    # Do first owned direction
    Uc_hat[:] = ifft(fu, axis=0)
    
    # Communicate all values
    comm.Alltoall([Uc_hat, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    for i in range(num_processes): 
        Uc_hatT[:, :, i*Np:(i+1)*Np] = U_mpi[i]

    #for i in range(num_processes):
    #    if not i == rank:
    #        comm.Sendrecv_replace([Uc_send[i], MPI.DOUBLE_COMPLEX], i, 0, i, 0)   
    #    Uc_hatT[:, :, i*Np:(i+1)*Np] = Uc_send[i]
           
    # Do last two directions
    u[:] = irfft(ifft(Uc_hatT, axis=2), axis=1)
    
#@profile
def fftn_mpi(u, fu):
    """fft in three directions using mpi
    """
    if num_processes == 1:
        fu[:] = fft(fft(rfft(u, axis=1), axis=2), axis=0)       
        return
    
    # Do 2 ffts in y-z directions on owned data
    #ft = fu.transpose(2,1,0)
    #ft[:] = fft(rfft(u, axis=1), axis=2)
    Uc_hatT[:] = fft(rfft(u, axis=1), axis=2)
    
    ## Communicating intermediate result 
    ##rstack(ft, Uc_hatT, Np, num_processes)       
    #fu_send = fu.reshape((num_processes, Np, Nf, Np))
    #for i in range(num_processes):
        #if not i == rank:
           #comm.Sendrecv_replace([fu_send[i], MPI.DOUBLE_COMPLEX], i, 0, i, 0)   
    #fu_send[:] = fu_send.transpose(0,3,2,1)
      
    # Transform data to align with x-direction  
    for i in range(num_processes): 
        #U_mpi[i] = ft[:, :, i*Np:(i+1)*Np]
        U_mpi[i] = Uc_hatT[:, :, i*Np:(i+1)*Np]
        
    # Communicate all values
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [fu, MPI.DOUBLE_COMPLEX])  
                
    # Do fft for last direction 
    fu[:] = fft(fu, axis=0)

#@jit((complex128[:,:,:,:], complex128[:,:,:], int64, int64))
def rstack(f, u, Np, num_processes):
    for i in range(num_processes): 
        f[i] = u[:, :, i*Np:(i+1)*Np]
    
def standardConvection(c):   
    """c_i = u_j du_i/dx_j"""
    for i in range(3):
        for j in range(3):
            ifftn_mpi(1j*KX[j]*U_hat[i], U_tmp[j])
        fftn_mpi(sum(U*U_tmp, 0), c[i])

def divergenceConvection(c, add=False):
    """c_i = div(u_i u_j)"""
    if not add: c.fill(0)
    for i in range(3):
        fftn_mpi(U[0]*U[i], F_tmp[i])
    c[0] += 1j*sum(KX*F_tmp, 0)
    c[1] += 1j*KX[0]*F_tmp[1]
    c[2] += 1j*KX[0]*F_tmp[2]
    fftn_mpi(U[1]*U[1], F_tmp[0])
    fftn_mpi(U[1]*U[2], F_tmp[1])
    fftn_mpi(U[2]*U[2], F_tmp[2])
    c[1] += (1j*KX[1]*F_tmp[0] + 1j*KX[2]*F_tmp[1])
    c[2] += (1j*KX[1]*F_tmp[1] + 1j*KX[2]*F_tmp[2])

def Cross(a, b, c):
    """c_k = F_k(a x b)"""
    fftn_mpi(a[1]*b[2]-a[2]*b[1], c[0])
    fftn_mpi(a[2]*b[0]-a[0]*b[2], c[1])
    fftn_mpi(a[0]*b[1]-a[1]*b[0], c[2])

def Curl(a, c):
    """c = F_inv(curl(a))"""
    ifftn_mpi(1j*(KX[0]*a[1]-KX[1]*a[0]), c[2])
    ifftn_mpi(1j*(KX[2]*a[0]-KX[0]*a[2]), c[1])
    ifftn_mpi(1j*(KX[1]*a[2]-KX[2]*a[1]), c[0])

def Div(a, c):
    """c = div(a)"""
    ifftn_mpi(1j*(sum(KX*a, 0), c))
    
def ComputeRHS(dU, rk):
    if rk > 0: # For rk=0 the correct values are already in U, V, W
        for i in range(3):
            ifftn_mpi(U_hat[i], U[i])
    
    if convection == "Standard":
        standardConvection(dU)
        
    elif convection == "Divergence":
        divergenceConvection(dU)        

    elif convection == "Skewed":
        standardConvection(dU)
        divergenceConvection(dU, add=True)        
        dU[:] = dU/2
        
    elif convection == "Vortex":
        Curl(U_hat, curl)
        Cross(U, curl, dU)

    # Dealias the nonlinear convection
    dU[:] *= dealias*dt
    
    # Compute pressure (To get actual pressure multiply by 1j/dt)
    P_hat[:] = sum(dU*KX_over_Ksq, 0)
        
    # Add pressure gradient
    dU[:] -= P_hat*KX    

    # Add contribution from diffusion
    dU[:] -= nu*dt*KK*U_hat

# Taylor-Green initialization
U[0] = sin(X[0])*cos(X[1])*cos(X[2])
U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
U[2] = 0 

# Transform initial data
for i in range(3):
   fftn_mpi(U[i], U_hat[i])
   
# Set some timers
t = 0.0
tstep = 0
fastest_time = 1e8
slowest_time = 0.0
# initialize plot and list k for storing energy
if rank == 0:
    #im = plt.imshow(zeros((N, N)))
    #plt.colorbar(im)
    #plt.draw()
    k = []

# Forward equations in time
tic = t0 = time.time()
while t < T-1e-8:
    t += dt; tstep += 1
    if temporal == "RK4":
        U_hat1[:] = U_hat0[:] = U_hat
        for rk in range(4):
            ComputeRHS(dU, rk)
            if rk < 3:
                U_hat[:] = U_hat0 + b[rk]*dU
            U_hat1[:] += a[rk]*dU            
        U_hat[:] = U_hat1[:]
        
    elif temporal == "ForwardEuler" or tstep == 1:  
        ComputeRHS(dU, 0)        
        U_hat[:] += dU
        if temporal == "AB2":
            U_hat0[:] = dU
        
    else:
        ComputeRHS(dU, 0)
        U_hat[:] += 1.5*dU - 0.5*U_hat0
        U_hat0[:] = dU

    for i in range(3):
        ifftn_mpi(U_hat[i], U[i])
        
    # Postprocessing intermediate results
    #if tstep % plot_result == 0:
        #ifftn_mpi(P_hat*1j/dt, P)
        #if rank == 0:
            #im.set_data(P[0])
            #im.autoscale()  
            #plt.pause(1e-6) 
            
    if tstep % params['write_result'] == 0 or tstep % params['write_yz_slice'][1] == 0:
        ifftn_mpi(P_hat*1j/dt, P)
        hdf5file.write(U, P, tstep)

    if tstep % compute_energy == 0:
        kk = comm.reduce(0.5*sum(U*U)*dx*dx*dx/L**3)
        if rank == 0:
            k.append(kk)
            print t, kk
            
    tt = time.time()-t0
    t0 = time.time()
    if tstep > 1:
        fastest_time = min(tt, fastest_time)
        slowest_time = max(tt, slowest_time)
        
    if tstep == 1 and make_profile:
        #Enable profiling after first step is finished
        profiler.enable()

toc = time.time()-tic

fast = comm.reduce(fastest_time, op=MPI.MIN, root=0)
slow = comm.reduce(slowest_time, op=MPI.MAX, root=0)

if rank == 0:
    print "Time = ", toc
    print "Fastest = ", fast
    print "Slowest = ", slow

    #figure()
    #k = array(k)
    #dkdt = (k[1:]-k[:-1])/dt
    #plot(-dkdt)
    #show()
    
if make_profile:
    results = create_profile(**vars())
    
hdf5file.generate_xdmf()    
hdf5file.close()
    