__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-01-02"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from MPI_knee import mpi_import, MPI
with mpi_import():
    import time
    import config
    t0 = time.time()
    import sys, cProfile
    from numpy import *
    from src import *

# Parse parameters from the command line and update config
commandline_kwargs = parse_command_line(sys.argv[1:])
config.update(commandline_kwargs)

# Import problem specific methods and solver methods specific to either slab or pencil decomposition
with mpi_import():
    from src.mpi import setup, ifftn_mpi, fftn_mpi
    from src.maths import *
    from problems import *

parameters.update(commandline_kwargs)
vars().update(parameters)
comm = MPI.COMM_WORLD
comm.barrier()
num_processes = comm.Get_size()
rank = comm.Get_rank()
if comm.Get_rank()==0: print "Import time ", time.time()-t0

float, complex, mpitype = {"single": (float32, complex64, MPI.F_FLOAT_COMPLEX),
                           "double": (float64, complex128, MPI.F_DOUBLE_COMPLEX)}[precision]

# Apply correct precision and set mesh size
dt = float(dt)
nu = float(nu)
N = 2**M
L = float(2*pi)
dx = float(L/N)

hdf5file = HDF5Writer(comm, dt, N, parameters, float, filename="vort.h5")
if config.make_profile: profiler = cProfile.Profile()

# Set up solver using wither slab or decomposition
vars().update(setup(**vars()))
W = U.copy()

# Rename variable since we are working with a vorticity formulation
W_hat = U_hat
Source = U_hat.copy()

def Cross(a, b, c):
    """c_k = F_k(a x b)"""
    U_tmp[:] = cross1(U_tmp, a, b)
    c[0] = fftn_mpi(U_tmp[0], c[0])
    c[1] = fftn_mpi(U_tmp[1], c[1])
    c[2] = fftn_mpi(U_tmp[2], c[2])
    return c

def Curl(a, c):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    F_tmp[:] = cross3(F_tmp, K_over_K2, a)
    c[0] = ifftn_mpi(F_tmp[0], c[0])
    c[1] = ifftn_mpi(F_tmp[1], c[1])
    c[2] = ifftn_mpi(F_tmp[2], c[2])    
    return c

def ComputeRHS(dU, rk):
    if rk > 0:
        for i in range(3):
            W[i] = ifftn_mpi(W_hat[i], W[i])
            
    U[:] = Curl(W_hat, U)
    F_tmp[:] = Cross(U, W, F_tmp)
    dU = cross2(dU, K, F_tmp)    
    dU = dealias_rhs(dU, dealias)
    dU -= nu*K2*W_hat    
    dU += Source    
    return dU

# initialize velocity and compute vorticity
U = initialize(**vars())
for i in range(3):
    F_tmp[i] = fftn_mpi(U[i], F_tmp[i])

W_hat[:] = cross2(W_hat, K, F_tmp)
for i in range(3):
    W[i] = ifftn_mpi(W_hat[i], W[i])

Source = set_source(**vars())

# Set up function to perform temporal integration (using config.integrator parameter)
integrate = getintegrator(**vars())

t = 0.0
tstep = 0
fastest_time = 1e8
slowest_time = 0.0
tic = t0 = time.time()
while t < T-1e-8:
    t += dt; tstep += 1
    
    W_hat = integrate(t, tstep, dt)

    for i in range(3):
        W[i] = ifftn_mpi(W_hat[i], W[i])
                    
    update(**vars())
    
    tt = time.time()-t0
    t0 = time.time()
    if tstep > 1:
        fastest_time = min(tt, fastest_time)
        slowest_time = max(tt, slowest_time)
        
    if tstep == 1 and config.make_profile:
        #Enable profiling after first step is finished
        profiler.enable()

toc = time.time()-tic

# Get min/max of fastest and slowest process
fast = (comm.reduce(fastest_time, op=MPI.MIN, root=0),
        comm.reduce(slowest_time, op=MPI.MIN, root=0))
slow = (comm.reduce(fastest_time, op=MPI.MAX, root=0),
        comm.reduce(slowest_time, op=MPI.MAX, root=0))

if rank == 0:
    print "Time = ", toc
    print "Fastest = ", fast
    print "Slowest = ", slow
    
if config.make_profile:
    results = create_profile(**vars())
    
hdf5file.close()
finalize(**vars())
