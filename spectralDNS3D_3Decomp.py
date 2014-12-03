__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import time, sys, cProfile

from numpy import *
from pylab import *
#from numpy import array, meshgrid, linspace, empty, zeros, sin, cos, pi, where, sum, int, float, bool
#from pylab import fftfreq, fft2, rfft, ifft, ifft2, irfft
from mpi4py import MPI
from utilities import *
from h5io import *
#from numba import jit, complex128, int64
#import numexpr

comm = MPI.COMM_WORLD

params = {
    'make_profile': 0,
    'M': 5,
    'P1': 1,
    'temporal': 'RK4',
    'write_result': 1e8,        # Write to HDF5 every..
    'write_yz_slice': [0, 1e8], # Write slice 0 (or higher) in y-z plance every..
    'compute_energy': 2,        # Compute solution energy every..
    'plot_result': 2,           # Show an image every..
    'nu': 0.000625,
    'dt': 0.01,
    'T': 0.1
}

commandline_kwargs = parse_command_line(sys.argv[1:])
params.update(commandline_kwargs)
assert params['temporal'] in ['RK4', 'ForwardEuler', 'AB2']
vars().update(params)

N = 2**M
L = 2 * pi
dx = L / N

num_processes = comm.Get_size()
rank = comm.Get_rank()

if make_profile: profiler = cProfile.Profile()

# Each cpu gets ownership of a pencil of size N1*N2*N in real space
# and (N1/2+1)*N2*N in Fourier space
P2 = num_processes / P1
N1 = N/P1
N2 = N/P2

if not (num_processes % 2 == 0 or num_processes == 1):
    raise IOError("Number of cpus must be even")

if not ((P1 == 1 or P1 % 2 == 0) and (P2 == 1 or P2 % 2 == 0)):
    raise IOError("Number of cpus in each direction must be even")

# Create two communicator groups for each rank
# The goups correspond to chunks in the xy-plane and the xz-plane
procxz = arange(num_processes)[rank%P1::P1]
procxy = arange(num_processes)[(rank/P1)*P1:(rank/P1+1)*P1]
group1 = comm.Get_group()
groupxy = MPI.Group.Incl(group1, procxy)
commxy = comm.Create(groupxy)
group2 = comm.Get_group()
groupxz = MPI.Group.Incl(group2, procxz)
commxz = comm.Create(groupxz)

xyrank = commxy.Get_rank() # Local rank in P1
xzrank = commxz.Get_rank() # Local rank in P2

# Create the physical mesh
x = linspace(0, L, N+1)[:-1]
x1 = slice(xyrank * N1, (xyrank+1) * N1, 1)
x2 = slice(xzrank * N2, (xzrank+1) * N2, 1)
X = array(meshgrid(x[x1], x, x[x2], indexing='ij'))

"""
Solution U is real and as such its transform, U_hat = fft(U)(k), 
is such that fft(U)(k) = conv(fft(U)(-k)) and thus it is sufficient 
to store N/2+1 Fourier coefficients in the first transformed direction (y).
"""

Nf = N/2+1 # Total points in y-direction
N1f = N1/2 if (P1>1 and xyrank == P1-1) else N1/2+1

U     = empty((3, N1, N, N2))
U_hat = empty((3, N2, N1f, N), dtype="complex")
P     = empty((N1, N, N2))
P_hat = empty((N2, N1f, N), dtype="complex")

# Temporal storage arrays (Not required by all temporal integrators)
U_hat0  = empty((3, N2, N1f, N), dtype="complex")
U_hat1  = empty((3, N2, N1f, N), dtype="complex")
dU      = empty((3, N2, N1f, N), dtype="complex")

# work arrays
Uc_hat_y  = empty((N1, Nf, N2), dtype="complex")
Uc_hat_x  = empty((N, N1/2, N2), dtype="complex")
Uc_hat_xr = empty((N, N1/2, N2), dtype="complex")
Uc_hat_z  = empty((N2, N1f, N), dtype="complex")
xz_plane   = empty((N, N2), dtype="complex")
xz_planer  = empty((N, N2), dtype="complex")
xz_plane2  = empty((N/2+1, N2), dtype="complex")
xz_recv   = zeros((N1, N2), dtype="complex")

curl    = empty((3, N1, N, N2))

# Set wavenumbers in grid
kx = fftfreq(N, 1./N)
ky = kx[:Nf].copy(); ky[-1] *= -1
k1 = slice(xzrank*N2, (xzrank+1)*N2, 1)
k2 = slice(xyrank*N1/2, xyrank*N1/2 + N1f, 1)

KX = array(meshgrid(kx[k1], ky[k2], kx, indexing='ij'), dtype=int)
KK = sum(KX*KX, 0)
KX_over_Ksq = array(KX, dtype=float) / where(KK==0, 1, KK)

# Filter for dealiasing nonlinear convection
dealias = array((abs(KX[0]) < 2./3.*max(kx))*(abs(KX[1]) < 2./3.*max(ky))*(abs(KX[2]) < 2./3.*max(kx)), dtype=bool)

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
        u[:] = irfft(ifft(ifft(fu, 2), 0), 1)
        return
    
    # Do first owned direction
    Uc_hat_z[:] = ifft(fu, 2)

    # Transpose all but k=N/2
    for i in range(P2): 
        Uc_hat_x[i*N2:(i+1)*N2] = Uc_hat_z[:, :N1/2, i*N2:(i+1)*N2]
    
    # Transpose k=N/2
    if xyrank == P1-1:
        for i in range(P2): 
            xz_plane[i*N2:(i+1)*N2] = Uc_hat_z[:, -1, i*N2:(i+1)*N2]
    
    # Communicate in xz-plane and do fft in x-direction
    commxz.Alltoall([Uc_hat_x, MPI.DOUBLE_COMPLEX], [Uc_hat_xr, MPI.DOUBLE_COMPLEX])
    Uc_hat_x[:] = ifft(Uc_hat_xr, 0)
    if xyrank == P1-1:
        commxz.Alltoall([xz_plane, MPI.DOUBLE_COMPLEX], [xz_planer, MPI.DOUBLE_COMPLEX])
        xz_plane[:] = ifft(xz_planer, 0)    
    
    # Communicate in xy-plane
    commxy.Alltoall([Uc_hat_x, MPI.DOUBLE_COMPLEX], [Uc_hat_xr, MPI.DOUBLE_COMPLEX])
    commxy.Scatter(xz_plane, xz_recv, root=P1-1)    
    Uc_hat_y[:, -1, :] = xz_recv[:]
    for i in range(P1):
        Uc_hat_y[:, i*N1/2:(i+1)*N1/2] = Uc_hat_xr[i*N1:(i+1)*N1]
    
    # Do fft for ydirection
    u[:] = irfft(Uc_hat_y, 1)
        
#@profile
def fftn_mpi(u, fu):
    """fft in three directions using mpi
    """
    if num_processes == 1:
        #fu[:] = fft(fft(rfft(u, 1), 0), 2) 
        f0 = rfft(u, 1)
        f0[:, 0, :] += 1j*f0[:, -1, :]
        ff = f0[:, :N/2, :].copy()
        fx = fft(ff, 0)
        xz_plane[:] = fx[:, 0, :]
        xz_plane2[:] = vstack((xz_plane[0].real, 0.5*(xz_plane[1:N/2]+conj(xz_plane[:N/2:-1])), xz_plane[N/2].real))
        f0[:, :N/2, :] = fx[:]
        f0[:, 0, :] = vstack((xz_plane2, conj(xz_plane2[N/2-1:0:-1])))
        xz_plane[:] = vstack((xz_plane[0].imag, -0.5*1j*(xz_plane[1:N/2]-conj(xz_plane[:N/2:-1])), xz_plane[N/2].imag, conj(xz_plane[(N/2-1):0:-1])))
        f0[:, -1, :] = xz_plane[:]    
        fu[:] = fft(f0, 2)
        return
    
    # Do fft in y direction on owned data
    Uc_hat_y[:] = rfft(u, 1)
    
    # Add last plane to complex part of first plane.
    # Both first and last planes are real. This way we store compactly
    # in an even number of planes (N/2)
    Uc_hat_y[:, 0, :] += 1j*Uc_hat_y[:, -1, :]

    # Transform neglecting k=N/2
    for i in range(P1):
        Uc_hat_x[i*N1:(i+1)*N1] = Uc_hat_y[:, i*N1/2:(i+1)*N1/2]
    
    # Communicate 
    commxy.Alltoall([Uc_hat_x, MPI.DOUBLE_COMPLEX], [Uc_hat_xr, MPI.DOUBLE_COMPLEX])

    # Do fft in x-direction
    Uc_hat_x[:] = fft(Uc_hat_xr, 0)
    
    # Take care of k=N/2 plane
    # Now both k=0 and k=N/2 are contained in 
    if xyrank == 0:
        xz_plane[:] = Uc_hat_x[:, 0, :]
        xz_plane2[:] = vstack((xz_plane[0].real, 0.5*(xz_plane[1:N/2]+conj(xz_plane[:N/2:-1])), xz_plane[N/2].real))
        Uc_hat_x[:, 0, :] = vstack((xz_plane2, conj(xz_plane2[(N/2-1):0:-1])))
        xz_plane[:] = vstack((xz_plane[0].imag, -0.5*1j*(xz_plane[1:N/2]-conj(xz_plane[:N/2:-1])), xz_plane[N/2].imag, conj(xz_plane[(N/2-1):0:-1])))
        commxy.Send([xz_plane, MPI.DOUBLE_COMPLEX], dest=commxy.Get_size()-1, tag=77)
        
    if xyrank == P1-1:
        commxy.Recv([xz_plane, MPI.DOUBLE_COMPLEX], source=0, tag=77)
        for i in range(P2):
            Uc_hat_z[:, -1, i*N2:(i+1)*N2] = xz_plane[i*N2:(i+1)*N2]
     
    # Communicate 
    commxz.Alltoall([Uc_hat_x, MPI.DOUBLE_COMPLEX], [Uc_hat_xr, MPI.DOUBLE_COMPLEX])
    
    for i in range(P2): 
        Uc_hat_z[:, :N1/2, i*N2:(i+1)*N2] = Uc_hat_xr[i*N2:(i+1)*N2]
                                   
    # Do fft for last direction 
    fu[:] = fft(Uc_hat_z, 2)

def Cross(a, b, c):
    """c = U x w"""
    fftn_mpi(a[1]*b[2]-a[2]*b[1], c[0])
    fftn_mpi(a[2]*b[0]-a[0]*b[2], c[1])
    fftn_mpi(a[0]*b[1]-a[1]*b[0], c[2])

def Curl(a, c):
    """c = curl(a)"""
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
    
    Curl(U_hat, curl)
    Cross(U, curl, dU)
    # Compute pressure (except the imaginary 1j)
    P_hat[:] = sum(dU*KX_over_Ksq, 0)
        
    # Dealias the nonlinear convection
    dU[:] *= dealias*dt
    
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
   
# Make it divergence free in case it is not
project(U_hat)

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

# RK4 loop in time
tic = t0 = time.time()
while t < T-1e-8:
    t += dt; tstep += 1
    if temporal == "RK4":
        U_hat1[:] = U_hat0[:] = U_hat
        for rk in range(4):
            ComputeRHS(dU, rk)        
            project(dU)
            if rk < 3:
                U_hat[:] = U_hat0 + b[rk]*dU
            U_hat1[:] += a[rk]*dU            
        U_hat[:] = U_hat1[:]
        
    elif temporal == "ForwardEuler" or tstep == 1:  
        ComputeRHS(dU, 0)        
        project(dU)
        U_hat[:] += dU
        if temporal == "AB2":
            U_hat0[:] = dU
        
    else:
        ComputeRHS(dU, 0)
        project(dU)
        U_hat[:] += 1.5*dU - 0.5*U_hat0
        U_hat0[:] = dU

    for i in range(3):
        ifftn_mpi(U_hat[i], U[i])
        
    # Postprocessing intermediate results
    #if tstep % plot_result == 0:
        #p = pressure()
        #if rank == 0:
            #im.set_data(p[Np/2])
            #im.autoscale()  
            #plt.pause(1e-6) 
            
    #if tstep % params['write_result'] == 0 or tstep % params['write_yz_slice'][1] == 0:
        #ifftn_mpi(P_hat*1j, P)
        #hdf5file.write(U, P, tstep)

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
    
#hdf5file.generate_xdmf()    
#hdf5file.close()
    