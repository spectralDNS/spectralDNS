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

comm = MPI.COMM_WORLD
comm.barrier()
if comm.Get_rank()==0:
    print "Import time ", time.time()-t0

# Parse parameters from the command line 
commandline_kwargs = parse_command_line(sys.argv[1:])

# Import parameters and problem specific routines
with mpi_import():
    exec("from problems.ThreeD.{0} import *".format(commandline_kwargs.get('problem', 'TaylorGreen')))

parameters.update(commandline_kwargs)
check_parameters(parameters)
vars().update(parameters)

if mem_profile: mem = MemoryUsage("Start (numpy/mpi4py++)", comm)

float, complex, mpitype = {"single": (float32, complex64, MPI.F_FLOAT_COMPLEX),
                           "double": (float64, complex128, MPI.F_DOUBLE_COMPLEX)}[precision]

# Set mesh size. Uniform size in all three directions (for now)
dt = float(dt)
nu = float(nu)
N = 2**M
L = float(2*pi)
dx = float(L / N)

num_processes = comm.Get_size()
rank = comm.Get_rank()
hdf5file = HDF5Writer(comm, dt, N, parameters, float)
if make_profile: profiler = cProfile.Profile()

# Import decomposed mesh, wavenumber mesh and FFT routines with either slab or pencil decomposition
with mpi_import():
    exec("from mpi.{0} import *".format(decomposition))
vars().update(setup(**vars()))

if mem_profile: mem("Arrays")

# RK4 parameters
a = array([1./6., 1./3., 1./3., 1./6.], dtype=float)*dt
b = array([0.5, 0.5, 1.], dtype=float)*dt

def project(u):
    """Project u onto divergence free space"""
    u -= sum(K_over_K2*u, 0)*K
    return u

def standardConvection(c):
    """c_i = u_j du_i/dx_j"""
    for i in range(3):
        for j in range(3):
            U_tmp[j] = ifftn_mpi(1j*K[j]*U_hat[i], U_tmp[j])
        c[i] = fftn_mpi(sum(U*U_tmp, 0), c[i])
    return c

def divergenceConvection(c, add=False):
    """c_i = div(u_i u_j)"""
    if not add: c.fill(0)
    for i in range(3):
        F_tmp[i] = fftn_mpi(U[0]*U[i], F_tmp[i])
    c[0] += 1j*sum(K*F_tmp, 0)
    c[1] += 1j*K[0]*F_tmp[1]
    c[2] += 1j*K[0]*F_tmp[2]
    F_tmp[0] = fftn_mpi(U[1]*U[1], F_tmp[0])
    F_tmp[1] = fftn_mpi(U[1]*U[2], F_tmp[1])
    F_tmp[2] = fftn_mpi(U[2]*U[2], F_tmp[2])
    c[1] += (1j*K[1]*F_tmp[0] + 1j*K[2]*F_tmp[1])
    c[2] += (1j*K[1]*F_tmp[1] + 1j*K[2]*F_tmp[2])
    return c

def Cross(a, b, c):
    """c_k = F_k(a x b)"""
    if useweave:
        weavecross(a, b, U_tmp, precision)
        c[0] = fftn_mpi(U_tmp[0], c[0])
        c[1] = fftn_mpi(U_tmp[1], c[1])
        c[2] = fftn_mpi(U_tmp[2], c[2])
    else:
        c[0] = fftn_mpi(a[1]*b[2]-a[2]*b[1], c[0])
        c[1] = fftn_mpi(a[2]*b[0]-a[0]*b[2], c[1])
        c[2] = fftn_mpi(a[0]*b[1]-a[1]*b[0], c[2])
    return c

def Curl(a, c):
    """c = F_inv(curl(a))"""
    if useweave:
        weavecrossi(a, K, F_tmp, precision)
        c[2] = ifftn_mpi(F_tmp[2], c[2])
        c[1] = ifftn_mpi(F_tmp[1], c[1])
        c[0] = ifftn_mpi(F_tmp[0], c[0])
    else:
        c[2] = ifftn_mpi(1j*(K[0]*a[1]-K[1]*a[0]), c[2])
        c[1] = ifftn_mpi(1j*(K[2]*a[0]-K[0]*a[2]), c[1])
        c[0] = ifftn_mpi(1j*(K[1]*a[2]-K[2]*a[1]), c[0])
    return c

def Div(a, c):
    """c = F_inv(div(a))"""
    c = ifftn_mpi(1j*(sum(KX*a, 0), c))
    return c
        
def ComputeRHS(dU, rk):
    if rk > 0: # For rk=0 the correct values are already in U
        for i in range(3):
            U[i] = ifftn_mpi(U_hat[i], U[i])
    
    # Compute convective term and place in dU
    if convection == "Standard":
        dU = standardConvection(dU)
        
    elif convection == "Divergence":
        dU = divergenceConvection(dU)        

    elif convection == "Skewed":
        dU = standardConvection(dU)
        dU = divergenceConvection(dU, add=True)        
        dU *= 0.5
        
    elif convection == "Vortex":
        curl[:] = Curl(U_hat, curl)
        dU = Cross(U, curl, dU)
    
    if useweave:
        weaverhs(dU, U_hat, K2, K, P_hat, K_over_K2, dealias, nu, precision)
    
    else:
        # Dealias the nonlinear convection
        dU *= dealias
        
        # Compute pressure (To get actual pressure multiply by 1j)
        P_hat[:] = sum(dU*K_over_K2, 0, out=P_hat)
            
        # Subtract pressure gradient
        dU -= P_hat*K
        
        # Subtract contribution from diffusion
        dU -= nu*K2*U_hat
    
    return dU

U = initialize(**vars())

# Transform initial data
for i in range(3):
   U_hat[i] = fftn_mpi(U[i], U_hat[i])

if mem_profile: mem("After first FFT")
   
# Set some timers
t = 0.0
tstep = 0
fastest_time = 1e8
slowest_time = 0.0
# initialize k for storing energy
if rank == 0: k = []; w = []

# Forward equations in time
tic = t0 = time.time()
while t < T-1e-8:
    t += dt; tstep += 1
    if integrator == "RK4":
        U_hat1[:] = U_hat0[:] = U_hat
        for rk in range(4):
            dU = ComputeRHS(dU, rk)
            if rk < 3:
                U_hat[:] = U_hat0;U_hat += b[rk]*dU
            U_hat1 += a[rk]*dU
        U_hat[:] = U_hat1
        
    elif integrator == "ForwardEuler" or tstep == 1:  
        dU = ComputeRHS(dU, 0)        
        U_hat += dU*dt
        if integrator == "AB2":
            U_hat0[:] = dU; U_hat0 *= dt
        
    else:
        dU = ComputeRHS(dU, 0)
        U_hat[:] += (1.5*dU*dt - 0.5*U_hat0)
        U_hat0[:] = dU; U_hat0 *= dt

    for i in range(3):
        U[i] = ifftn_mpi(U_hat[i], U[i])
                    
    update(**vars())
    
    tt = time.time()-t0
    t0 = time.time()
    if tstep > 1:
        fastest_time = min(tt, fastest_time)
        slowest_time = max(tt, slowest_time)
        
    if tstep == 1 and make_profile:
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
if make_profile:
    results = create_profile(**vars())

if mem_profile: mem("End")
    
hdf5file.generate_xdmf()  
hdf5file.close()
finalize(**vars())
