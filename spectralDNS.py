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
#from mpi4py import MPI
#from numpy import *
#from h5io import *
#from utilities import *

comm = MPI.COMM_WORLD
comm.barrier()
if comm.Get_rank()==0:
    print "Import time ", time.time()-t0

params = {
    'decomposition': 'slab',    # 'slab' or 'pencil'
    'communication': 'alltoall',# 'alltoall' or 'sendrecv_replace' (only for slab)
    'convection': 'Vortex',     # 'Standard', 'Divergence', 'Skewed', 'Vortex'
    'make_profile': 0,          # Enable cProfile profiler
    'mem_profile': False,       # Check memory use
    'M': 5,                     # Mesh size
    'P1': 1,                    # Mesh decomposition in first direction (pencil P1*P2=num_processes)
    'temporal': 'RK4',          # Integrator ('RK4', 'ForwardEuler', 'AB2')
    'write_result': 1e8,        # Write to HDF5 every..
    'write_yz_slice': [0, 1e8], # Write slice 0 (or higher) in y-z plance every..
    'compute_energy': 2,        # Compute solution energy every..
    'nu': 0.000625,             # Viscosity
    'dt': 0.01,                 # Time step
    'T': 0.1,                   # End time
    'precision': "double"       # single or double precision
}

# Parse parameters from the command line 
commandline_kwargs = parse_command_line(sys.argv[1:])
params.update(commandline_kwargs)
assert params['convection'] in ['Standard', 'Divergence', 'Skewed', 'Vortex']
assert params['temporal'] in ['RK4', 'ForwardEuler', 'AB2']
vars().update(params)

if mem_profile: mem = MemoryUsage("Start (numpy/mpi4py++)", comm)

float, complex, mpitype = {"single": (float32, complex64, MPI.F_FLOAT_COMPLEX),
                           "double": (float64, complex128, MPI.F_DOUBLE_COMPLEX)}[precision]

# Set mesh size. Uniform size in all three directions (for now)
N = 2**M
L = 2 * pi
dx = L / N

num_processes = comm.Get_size()
rank = comm.Get_rank()
hdf5file = HDF5Writer(comm, dt, N, params, float)
if make_profile: profiler = cProfile.Profile()

# Import decomposed mesh, wavenumber mesh and FFT routines with either slab or pencil decomposition
with mpi_import():
    exec("from mpi.{} import *".format(decomposition))
vars().update(setup(**vars()))

if mem_profile: mem("Arrays")

# RK4 parameters
a = array([1./6., 1./3., 1./3., 1./6.], dtype=float)
b = array([0.5, 0.5, 1.], dtype=float)

def project(u):
    """Project u onto divergence free space"""
    u[:] -= sum(KX_over_Ksq*u, 0)*KX

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
    """c = F_inv(div(a))"""
    ifftn_mpi(1j*(sum(KX*a, 0), c))
    
def ComputeRHS(dU, rk):
    if rk > 0: # For rk=0 the correct values are already in U
        for i in range(3):
            ifftn_mpi(U_hat[i], U[i])
    
    # Compute convective term and place in dU
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

if mem_profile: mem("After first FFT")
   
# Set some timers
t = 0.0
tstep = 0
fastest_time = 1e8
slowest_time = 0.0
# initialize k for storing energy
if rank == 0: k = []

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
        
    if tstep % params['write_result'] == 0 or tstep % params['write_yz_slice'][1] == 0:
        ifftn_mpi(P_hat*1j/dt, P)
        hdf5file.write(U, P, tstep)

    if tstep % compute_energy == 0:
        kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx*dx*dx/L**3/2) # Compute energy with double precision
        if rank == 0:
            k.append(kk)
            print t, float(kk)
            
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
        comm.reduce(fastest_time, op=MPI.MAX, root=0))
slow = (comm.reduce(slowest_time, op=MPI.MIN, root=0),
        comm.reduce(slowest_time, op=MPI.MAX, root=0))

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

if mem_profile: mem("End")
    
hdf5file.generate_xdmf()  
hdf5file.close()
