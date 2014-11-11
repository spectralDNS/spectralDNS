__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from numpy import *
from pylab import *
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD

try:
    from wrappyfftw import *
    
except:
    print Warning("Install pyfftw, it is much faster than numpy fft")

# Set the size of the triply periodic box N**3
M = 6
N = 2**M

num_processes = comm.Get_size()
rank = comm.Get_rank()
threads = 2

if not num_processes in [2**i for i in range(M+1)]:
    raise IOError("Number of cpus must be in ", [2**i for i in range(M+1)])

# Each cpu gets ownership of Np slices
Np = N / num_processes     

L = 2 * pi
dx = L / N
dt = 0.01
nu = 0.000625
T = 0.1

try:
    from HDF55Writer import HDF5Writer
    hdf5file = HDF5Writer(comm, dt, N)
    
except:
    hdf5file = None

# Set some switches for doing postprocessing
write_result = 2        # Write to HDF5 every..
compute_energy = 2      # Compute solution energy every..
plot_result = 1e8       # Show an image every..

# Choose convection scheme
conv = {0: "Standard",
        1: "Divergence",
        2: "Skewed",
        3: "VortexI",
        4: "VortexII"}.get(eval(sys.argv[-1]), "Standard")

# Create the mesh
x = linspace(0, L, N+1)[:-1]
X = array(meshgrid(x[rank*Np:(rank+1)*Np], x, x, indexing='ij'))

# Solution array and Fourier coefficients
U     = empty((3, Np, N, N))
U_hat = empty((3, Np, N, N), dtype="complex") 

# Arrays for mpi communication
U_sendc = empty((num_processes, Np, N, Np), dtype="complex")
U_recvc = empty((num_processes, Np, N, Np), dtype="complex")
U_send  = empty((num_processes, Np, N, Np))
U_recv  = empty((num_processes, Np, N, Np))

# RK4 arrays
U_hatold= empty((3, Np, N, N), dtype="complex")
U_hatc  = empty((3, Np, N, N), dtype="complex")
dU      = empty((3, Np, N, N), dtype="complex")

# work arrays
U_hat_tmp = empty((3, Np, N, N), dtype="complex")
U_tmp   = empty((3, Np, N, N))
Uc_hat  = empty((Np, N, N), dtype="complex")
Utc     = empty((N, N, Np), dtype="complex")
Ut      = empty((N, N, Np))
curl    = empty((3, Np, N, N))

# Set wavenumbers in grid
kx = (mod(0.5 + arange(0, N, dtype="float")/N, 1) - 0.5)*2*pi/dx
KX = array(meshgrid(kx[rank*Np:(rank+1)*Np], kx, kx, indexing='ij'))
KK = sum(KX*KX, 0)
U_tmp[0] = where(KK==0, 1, KK)
KX_over_Ksq = KX.copy()
for j in range(3):
    KX_over_Ksq[j] /= U_tmp[0]

# Filter for dealiasing nonlinear convection
dealias = array((abs(KX[0]) < 2./3.*max(kx))*(abs(KX[1]) < 2./3.*max(kx))*(abs(KX[2]) < 2./3.*max(kx)), dtype=int)

# RK4 parameters
a = [1./6., 1./3., 1./3., 1./6.]
b = [0.5, 0.5, 1.]

def pressure():
    """Pressure is not really used, but may be recovered from the velocity.
    """
    p = zeros((Np, N, N))
    for i in range(3):
        for j in range(3):
            ifftn_mpi(1j*KX[j]*U_hat[i], U_tmp[j])
        fftn_mpi(sum(U*U_tmp, 0), U_hat_tmp[i])
    p_hat = 1j*sum(KX_over_Ksq*U_hat_tmp, 0)
    ifftn_mpi(p_hat, p)
    return p

def project(xU):
    Uc_hat[:] = sum(KX*xU, 0)
    for i in range(3):
        xU[i] = xU[i] - Uc_hat*KX_over_Ksq[i]
    
def ifftn_mpi(fu, u):
    """ifft in three directions using mpi
    """
    # Do 2D ifft2 in y-z directions on owned data
    Uc_hat[:] = ifft2(fu)    
    
    # Set up for communicating intermediate result 
    for i in range(num_processes):
        U_sendc[i] = Uc_hat[:, :, i*Np:(i+1)*Np]
        
    # Communicate all values
    comm.Alltoall(U_sendc, U_recvc)
    
    # Place received data in chunk Utc
    for i in range(num_processes):
        Utc[i*Np:(i+1)*Np] = U_recvc[i]
        
    # Do ifft for final direction        
    Ut[:] = real(ifft(Utc, axis=0))
    
    # Store values to be sent
    for i in range(num_processes):
        U_send[i] = Ut[i*Np:(i+1)*Np]
    
    # Communicate all values
    comm.Alltoall(U_send, U_recv)

    # Copy to final array
    for i in range(num_processes):
        u[:, :, i*Np:(i+1)*Np] = U_recv[i]

def fftn_mpi(u, fu):
    """fft in three directions using mpi
    """
    # Do 2D fft2 in y-z directions on owned data
    Uc_hat[:] = fft2(u)
    
    # Set up for communicating intermediate result 
    for i in range(num_processes):
        U_sendc[i] = Uc_hat[:, :, i*Np:(i+1)*Np]
        
    # Communicate all values
    comm.Alltoall(U_sendc, U_recvc)
    
    # Place in chunk Utc
    for i in range(num_processes):
        Utc[i*Np:(i+1)*Np] = U_recvc[i]
        
    # Do fft for final direction        
    Utc[:] = fft(Utc, axis=0)
    
    # Store values to be sent
    for i in range(num_processes):
        U_sendc[i] = Utc[i*Np:(i+1)*Np]
    
    # Communicate all values
    comm.Alltoall(U_sendc, U_recvc)

    # Copy to final array 
    for i in range(num_processes):
        fu[:, :, i*Np:(i+1)*Np] = U_recvc[i]
            
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
        fftn_mpi(U[0]*U[i], U_hat_tmp[i])
    c[0] += 1j*sum(KX*U_hat_tmp, 0)
    c[1] += 1j*KX[0]*U_hat_tmp[1]
    c[2] += 1j*KX[0]*U_hat_tmp[2]
    fftn_mpi(U[1]*U[1], U_hat_tmp[0])
    fftn_mpi(U[1]*U[2], U_hat_tmp[1])
    fftn_mpi(U[2]*U[2], U_hat_tmp[2])
    c[1] += (1j*KX[1]*U_hat_tmp[0] + 1j*KX[2]*U_hat_tmp[1])
    c[2] += (1j*KX[1]*U_hat_tmp[1] + 1j*KX[2]*U_hat_tmp[2])
    
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
    ifftn_mpi(1j*(KX[0]*a[0]+KX[1]*a[1]+KX[2]*a[2]), c)
    
def ComputeRHS(dU, rk):
    if rk > 0: # For rk=0 the correct values are already in U, V, W
        for i in range(3):
            ifftn_mpi(U_hat[i], U[i])
    
    if conv == "Standard":
        standardConvection(dU)
        
    elif conv == "Divergence":
        divergenceConvection(dU)        

    elif conv == "Skewed":
        standardConvection(dU)
        divergenceConvection(dU, add=True)        
        dU[:] = dU/2

    elif conv == "VortexI":    
        Curl(U_hat, curl)
        Cross(U, curl, dU)
        fftn_mpi(0.5*sum(U**2, 0), U_hat_tmp[0])        
        for i in range(3):
            dU[i] -= 1j*KX[i]*U_hat_tmp[0]
        
    elif conv == "VortexII":
        Curl(U_hat, curl)
        Cross(U, curl, dU)

    else:
        raise TypeError, "Wrong type of convection"
    
    # Dealias the nonlinear convection
    dU[:] *= dealias*dt
    
    # Add pressure gradient
    Uc_hat[:] = sum(dU*KX_over_Ksq, 0)
    for i in range(3):
        dU[i] -= Uc_hat*KX[i]
                 
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

tic = time.time()
t = 0.0
tstep = 0

# initialize plot and list k for storing energy
if rank == 0:
    im = plt.imshow(zeros((N, N)))
    plt.colorbar(im)
    plt.draw()
    k = []

# RK4 loop in time
while t < T:
    t += dt; tstep += 1
    U_hatold[:] = U_hat
    U_hatc[:] = U_hat
    for rk in range(4):
        ComputeRHS(dU, rk)        
        project(dU)

        if rk < 3:
            U_hat[:] = U_hatold + b[rk]*dU
        U_hatc[:] = U_hatc + a[rk]*dU
        
    U_hat[:] = U_hatc[:]        
    for i in range(3):
        ifftn_mpi(U_hat[i], U[i])
    
    # Postprocessing intermediate results
    if tstep % plot_result == 0:
        p = pressure()
        if rank == 0:
            im.set_data(p[Np/2])
            im.autoscale()  
            plt.pause(1e-6) 
            
    if tstep % write_result == 0 and hdf5file:
        hdf5file.write(U, pressure(), tstep)

    if tstep % compute_energy == 0:
        kk = comm.reduce(0.5*sum(U*U)*dx*dx*dx/L**3)
        if rank == 0:
            k.append(kk)
            print t, kk

if hdf5file: hdf5file.close()

if rank == 0:
    print "Time = ", time.time()-tic
    #figure()
    #k = array(k)
    #dkdt = (k[1:]-k[:-1])/dt
    #plot(-dkdt)
    #show()

