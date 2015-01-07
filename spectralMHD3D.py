__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from MPI_knee import mpi_import, MPI
with mpi_import():
    import time
    t0 = time.time()
    import sys, cProfile
    from h5io import *
    from numpy import *
    from utilities import *

#import time
#t0 = time.time()
#import sys, cProfile
#from h5io import *
#from mpi4py import MPI
#from numpy import *
#from utilities import *

comm = MPI.COMM_WORLD
comm.barrier()
if comm.Get_rank()==0:
    print "Import time ", time.time()-t0

params = {
    'make_profile': 0,          # Enable cProfile profiler
    'mem_profile': False,       # Check memory use
    'M': 5,                     # Mesh size
    'temporal': 'RK4',      
    # Integrator ("RK4", "ForwardEuler", "AB2")
    'write_result': 1e8,        # Write to HDF5 every..
    'write_yz_slice': [0, 1e8], # Write slice 0 (or higher) in y-z plance every..
    'compute_energy': 2,        # Compute solution energy every..
    'nu': 0.01,                 # Viscosity
    'eta': 0.01,                # 
    'dt': 0.01,                 # Time step
    'T': 0.1,                   # End time
    'precision': "double"       # single or double precision
}

# Parse parameters from the command line 
commandline_kwargs = parse_command_line(sys.argv[1:])
params.update(commandline_kwargs)
assert params['temporal'] in ['RK4', 'ForwardEuler', 'AB2']
vars().update(params)

if mem_profile: mem = MemoryUsage("Start (numpy/mpi4py++)", comm)

float, complex, mpitype = {"single": (float32, complex64, MPI.F_FLOAT_COMPLEX),
                           "double": (float64, complex128, MPI.F_DOUBLE_COMPLEX)}[precision]

N = 2**M
L = float(2 * pi)
dx = float(L / N)

num_processes = comm.Get_size()
rank = comm.Get_rank()
if not num_processes in [2**i for i in range(M+1)]:
    raise IOError("Number of cpus must be in ", [2**i for i in range(M+1)])

if make_profile: profiler = cProfile.Profile()

# Each cpu gets ownership of Np slices
Np = N / num_processes     

hdf5file = HDF5Writer(comm, dt, N, params, float, filename="UB.h5")

# Create the physical mesh
x = linspace(0, L, N+1).astype(float)[:-1]
X = array(meshgrid(x[rank*Np:(rank+1)*Np], x, x, indexing='ij'), dtype=float)

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
UB     = empty((6, Np, N, N), dtype=float)  
UB_hat = empty((6, N, Nf, Np), dtype=complex)
P      = empty((Np, N, N), dtype=float)
P_hat  = empty((N, Nf, Np), dtype=complex)
# Create views into large data structures
U     = UB[:3] 
U_hat = UB_hat[:3]
B     = UB[3:]
B_hat = UB_hat[3:]

# Temporal storage arrays (Not required by all temporal integrators)
UB_hat0  = empty((6, N, Nf, Np), dtype=complex)
UB_hat1  = empty((6, N, Nf, Np), dtype=complex)
dU      = empty((6, N, Nf, Np), dtype=complex)

# work arrays
F_tmp   = empty((3, 3, N, Nf, Np), dtype=complex)
Uc_hat  = empty((N, Nf, Np), dtype=complex)
Uc_hatT = empty((Np, Nf, N), dtype=complex)
Uc_send = Uc_hat.reshape((num_processes, Np, Nf, Np))
U_mpi   = empty((num_processes, Np, Nf, Np), dtype=complex)

# Set wavenumbers in grid
kx = fftfreq(N, 1./N).astype(int)
ky = kx[:Nf].copy(); ky[-1] *= -1
KX = array(meshgrid(kx, ky, kx[rank*Np:(rank+1)*Np], indexing='ij'), dtype=int)
KK = sum(KX*KX, 0, dtype=int)
KX_over_Ksq = KX.astype(float) / where(KK==0, 1, KK).astype(float)

# Filter for dealiasing nonlinear convection
kmax = 2./3.*(N/2+1)
dealias = array((abs(KX[0]) < kmax)*(abs(KX[1]) < kmax)*
                (abs(KX[2]) < kmax), dtype=bool)

if mem_profile: mem("Arrays")

# RK4 parameters
a = array([1./6., 1./3., 1./3., 1./6.], dtype=float)
b = array([0.5, 0.5, 1.], dtype=float)

def project(u):
    """Project u onto divergence free space"""
    u[:] -= sum(KX_over_Ksq*u, 0)*KX

def ifftn_mpi(fu, u):
    """ifft in three directions using mpi.
    Need to do ifft in reversed order of fft
    """
    if num_processes == 1:
        u[:] = irfftn(fu, axes=(0,2,1))
        return
    
    # Do first owned direction
    Uc_hat[:] = ifft(fu, axis=0)
    
    ## Communicate all values
    #comm.Alltoall([Uc_hat, mpitype], [U_mpi, mpitype])
    #for i in range(num_processes): 
        #Uc_hatT[:, :, i*Np:(i+1)*Np] = U_mpi[i]

    #Uc_send = fu.reshape((num_processes, Np, Nf, Np))
    for i in range(num_processes):
       if not i == rank:
           comm.Sendrecv_replace([Uc_send[i], mpitype], i, 0, i, 0)   
       Uc_hatT[:, :, i*Np:(i+1)*Np] = Uc_send[i]
           
    # Do last two directions
    u[:] = irfft2(Uc_hatT, axes=(2,1))
    
def fftn_mpi(u, fu):
    """fft in three directions using mpi
    """
    if num_processes == 1:
        fu[:] = rfftn(u, axes=(0,2,1))
        return
    
    # Do 2 ffts in y-z directions on owned data
    Uc_hatT[:] = rfft2(u, axes=(2,1))
    # Transform data to align with x-direction  
    for i in range(num_processes): 
       #U_mpi[i] = ft[:, :, i*Np:(i+1)*Np]
       U_mpi[i] = Uc_hatT[:, :, i*Np:(i+1)*Np]
        
    # Communicate all values
    comm.Alltoall([U_mpi, mpitype], [fu, mpitype])  
    
    ## Communicating intermediate result 
    #ft = fu.transpose(2,1,0)
    #ft[:] = rfft2(u, axes=(2,1))
    #fu_send = fu.reshape((num_processes, Np, Nf, Np))
    #for i in range(num_processes):
        #if not i == rank:
            #comm.Sendrecv_replace([fu_send[i], mpitype], i, 0, i, 0)   
    #fu_send[:] = fu_send.transpose(0,3,2,1)
                      
    # Do fft for last direction 
    fu[:] = fft(fu, axis=0)
            
def divergenceConvection(z0, z1, c):
    """Divergence convection using Elsasser variables
    z0=U+B
    z1=U-B
    """
    for i in range(3):
        for j in range(3):
            fftn_mpi(z0[i]*z1[j], F_tmp[i, j])
            
    c[:3] = -1j*(KX[0]*(F_tmp[:, 0] + F_tmp[0, :])
                +KX[1]*(F_tmp[:, 1] + F_tmp[1, :])
                +KX[2]*(F_tmp[:, 2] + F_tmp[2, :]))/2.0

    c[3:] =  1j*(KX[0]*(F_tmp[0, :] - F_tmp[:, 0])
                +KX[1]*(F_tmp[1, :] - F_tmp[:, 1])
                +KX[2]*(F_tmp[2, :] - F_tmp[:, 2]))/2.0    
    
def Div(a, c):
    """c = div(a)"""
    ifftn_mpi(1j*(sum(KX*a, 0)), c)
    
def project(u):
    """Project u onto divergence free space"""
    u[:] -= sum(KX_over_Ksq*u, 0)*KX    
    
def ComputeRHS(dU, rk):
    if rk > 0: # For rk=0 the correct values are already in U, B
        for i in range(6):
            ifftn_mpi(UB_hat[i], UB[i])
    
    # Compute convective term and place in dU
    divergenceConvection(U+B, U-B, dU)
    
    # Dealias the nonlinear convection
    dU[:] *= dealias*dt
    
    # Compute pressure (To get actual pressure multiply by 1j/dt)
    P_hat[:] = sum(dU[:3]*KX_over_Ksq, 0)
        
    # Add pressure gradient
    dU[:3] -= P_hat*KX    

    # Add contribution from diffusion
    dU[:3] -= nu*dt*KK*U_hat
    dU[3:] -= eta*dt*KK*B_hat

# Taylor-Green initialization
U[0] = sin(X[0])*cos(X[1])*cos(X[2])
U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
U[2] = 0 
B[0] = sin(X[0])*sin(X[1])*cos(X[2])
B[1] = cos(X[0])*cos(X[1])*cos(X[2])
B[2] = 0 

# Transform initial data
for i in range(6):
   fftn_mpi(UB[i], UB_hat[i])

if mem_profile: mem("After first FFT")
   
# Set some timers
t = 0.0
tstep = 0
fastest_time = 1e8
slowest_time = 0.0
# initialize k, bt for storing energy
if rank == 0: 
    k = []
    bt = []

# Forward equations in time
tic = t0 = time.time()
while t < T-1e-8:
    t += dt; tstep += 1
    if temporal == "RK4":
        UB_hat1[:] = UB_hat0[:] = UB_hat
        for rk in range(4):
            ComputeRHS(dU, rk)
            if rk < 3:
                UB_hat[:] = UB_hat0 + b[rk]*dU
            UB_hat1[:] += a[rk]*dU            
        UB_hat[:] = UB_hat1[:]
        
    elif temporal == "ForwardEuler" or tstep == 1:  
        ComputeRHS(dU, 0)        
        UB_hat[:] += dU
        if temporal == "AB2":
            UB_hat0[:] = dU
        
    else:
        ComputeRHS(dU, 0)
        UB_hat[:] += 1.5*dU - 0.5*UB_hat0
        UB_hat0[:] = dU

    for i in range(6):
        ifftn_mpi(UB_hat[i], UB[i])
        
    if tstep % params['write_result'] == 0 or tstep % params['write_yz_slice'][1] == 0:
        ifftn_mpi(P_hat*1j/dt, P)
        hdf5file.write(UB, P, tstep)

    if tstep % compute_energy == 0:
        kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx*dx*dx/L**3/2) # Compute energy with double precision
        bb = comm.reduce(sum(B.astype(float64)*B.astype(float64))*dx*dx*dx/L**3/2)
        if rank == 0:
            k.append(kk)
            bt.append(bb)
            print t, float(kk), float(bb)
            
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

from pylab import figure, plot, show
if rank == 0:
    print "Time = ", toc
    print "Fastest = ", fast
    print "Slowest = ", slow

    figure()
    k = array(k)
    #dkdt = (k[1:]-k[:-1])/dt
    #plot(-dkdt)
    plot(k)
    plot(bt)
    show()
    
if make_profile:
    results = create_profile(**vars())

if mem_profile: mem("End")
    
hdf5file.generate_xdmf()    
hdf5file.close()
