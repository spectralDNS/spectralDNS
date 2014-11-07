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
    import pyfftw
    # Monkey patches for fft
    ifft = pyfftw.interfaces.numpy_fft.ifft
    fft = pyfftw.interfaces.numpy_fft.fft
    fft2 = pyfftw.interfaces.numpy_fft.fft2
    ifft2 = pyfftw.interfaces.numpy_fft.ifft2
    # Keep fft objects in cache for efficiency
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1000)
    def empty(N, dtype="float", bytes=32):
        return pyfftw.n_byte_align_empty(N, bytes, dtype=dtype)

except:
    Warning("Install pyfftw, it is much faster than numpy fft")

# Set the size of the triply periodic box N**3
M = 6
N = 2**M

num_processes = comm.Get_size()
rank = comm.Get_rank()
if not num_processes in [2**i for i in range(M)]:
    raise IOError("Number of cpus must be in ", [2**i for i in range(M)])

# Each cpu gets ownership of Np slices
Np = N / num_processes     

L = 2 * pi
dx = L / N
#dt = eval(sys.argv[-2])
dt = 0.01
nu = 0.000625
plot_result = 1000
T = 0.1
# Choose convection scheme
convection = {0: "Standard",
              1: "Divergence",
              2: "Skewed",
              3: "VortexI",
              4: "VortexII"}

conv = convection.get(eval(sys.argv[-1]), "Standard")
#conv = convection[4]

# Create the mesh
x = linspace(0, L, N+1)[:-1]
X = array(meshgrid(x[rank*Np:(rank+1)*Np], x, x, indexing='ij'))

# Solution array
U = empty((3, Np, N, N))

# Fourier coefficients
U_hat = empty((3, Np, N, N), dtype="complex") 

# Arrays for mpi communication
U_sendc = empty((num_processes, Np, N, Np), dtype="complex")
U_recvc = empty((num_processes, Np, N, Np), dtype="complex")
U_send = empty((num_processes, Np, N, Np))
U_recv = empty((num_processes, Np, N, Np))

# RK4 arrays
U_hatold = empty((3, Np, N, N), dtype="complex")
U_hatc = empty((3, Np, N, N), dtype="complex")
dU = empty((3, Np, N, N), dtype="complex")

# work arrays
U_tmp = empty((3, Np, N, N))
U_hat_tmp = empty((3, Np, N, N), dtype="complex")
Uc_hat = empty((Np, N, N), dtype="complex")
Utc = empty((N, N, Np), dtype="complex")
Ut = empty((N, N, Np))

curl = empty((3, Np, N, N))

kx = (mod(0.5+arange(0, N, dtype="float")/N, 1)-0.5)*2*pi/dx
KX = array(meshgrid(kx[rank*Np:(rank+1)*Np], kx, kx, indexing='ij'))
KK = KX[0]**2 + KX[1]**2 + KX[2]**2
dealias = array((abs(KX[0]) < (2./3.)*max(kx))*(abs(KX[1]) < (2./3.)*max(kx))*(abs(KX[2]) < (2./3.)*max(kx)), dtype=int)
Ksq = where(KK==0, 1, KK)

# RK4 parameters
a=[1./6., 1./3., 1./3., 1./6.]
b=[0.5, 0.5, 1., 1.]

def pressure():
    p = zeros((Np, N, N))
    for i in range(3):
        for j in range(3):
            ifftn_mpi(1j*KX[j]*U_hat[i], U_tmp[j])
        fftn_mpi(U[0]*U_tmp[0] + U[1]*U_tmp[1] + U[2]*U_tmp[2], U_hat_tmp[i])
    p_hat = (1j*KX[0]*U_hat_tmp[0]
            +1j*KX[1]*U_hat_tmp[1]
            +1j*KX[2]*U_hat_tmp[2]) / Ksq    
    ifftn_mpi(p_hat, p)
    return p

def project(xU):
    Uc_hat[:] = (KX[0]*xU[0] + KX[1]*xU[1] + KX[2]*xU[2]) / Ksq
    for i in range(3):
        xU[i] = xU[i] - Uc_hat*KX[i]
    
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
    
    # Return to owner
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
    
    # Return to owner
    for i in range(num_processes):
        U_sendc[i] = Utc[i*Np:(i+1)*Np]
    
    # Communicate all values
    comm.Alltoall(U_sendc, U_recvc)

    # Copy to final array 
    for i in range(num_processes):
        fu[:, :, i*Np:(i+1)*Np] = U_recvc[i]
        
def ComputeRHS(dU, rk):
    # Compute U from U_hat
    if rk > 0: # For rk=0 the correct values are already in U, V, W
        for i in range(3):
            ifftn_mpi(U_hat[i], U[i])
    
    if conv == "Standard":
        for i in range(3):
            for j in range(3):
                ifftn_mpi(1j*KX[j]*U_hat[i], U_tmp[j])
            for j in range(3):
                fftn_mpi(U[j]*U_tmp[j], U_hat_tmp[j])
            dU[i] = -dealias*(sum(U_hat_tmp, axis=0))*dt
        
    elif conv == "Divergence":
        for i in range(3):
            fftn_mpi(U[0]*U[j], U_hat_tmp[j])
        dU[0] = 1j*KX[0]*U_hat_tmp[0] + 1j*KX[1]*U_hat_tmp[1] + 1j*KX[2]*U_hat_tmp[2]
        dU[1] = 1j*KX[0]*U_hat_tmp[1]
        dU[2] = 1j*KX[0]*U_hat_tmp[2]
        fftn_mpi(U[1]*U[1], U_hat_tmp[0])
        fftn_mpi(U[1]*U[2], U_hat_tmp[1])
        fftn_mpi(U[2]*U[2], U_hat_tmp[2])
        dU[1] += (1j*KX[1]*U_hat_tmp[0] + 1j*KX[2]*U_hat_tmp[1])
        dU[2] += (1j*KX[1]*U_hat_tmp[1] + 1j*KX[2]*U_hat_tmp[2])
        dU[:] = -dealias*dU[:]*dt

    elif conv == "Skewed":
        for i in range(3):
            for j in range(3):
                ifftn_mpi(1j*KX[j]*U_hat[i], U_tmp[j])
            for j in range(3):
                fftn_mpi(U[j]*U_tmp[j], U_hat_tmp[j])
            dU[i] = sum(U_hat_tmp, axis=0)
        for i in range(3):
            fftn_mpi(U[0]*U[i], U_hat_tmp[i])
        dU[0] += (1j*KX[0]*U_hat_tmp[0] + 1j*KX[1]*U_hat_tmp[1] + 1j*KX[2]*U_hat_tmp[2])
        dU[1] += 1j*KX[0]*U_hat_tmp[1]
        dU[2] += 1j*KX[0]*U_hat_tmp[2]
        fftn_mpi(U[1]*U[1], U_hat_tmp[0])
        fftn_mpi(U[1]*U[2], U_hat_tmp[1])
        fftn_mpi(U[2]*U[2], U_hat_tmp[2])
        dU[1] += (1j*KX[1]*U_hat_tmp[0] + 1j*KX[2]*U_hat_tmp[1])
        dU[2] += (1j*KX[1]*U_hat_tmp[1] + 1j*KX[2]*U_hat_tmp[2])
        dU[:] = -dealias*dU[:]*dt/2.

    elif conv == "VortexI":    
        ifftn_mpi(1j*KX[0]*U_hat[1], curl[2])
        ifftn_mpi(1j*KX[1]*U_hat[0], U_tmp[0])
        curl[2] -= U_tmp[0]
        ifftn_mpi(1j*KX[2]*U_hat[0], curl[1])
        ifftn_mpi(1j*KX[0]*U_hat[2], U_tmp[0])
        curl[1] -= U_tmp[0]
        ifftn_mpi(1j*KX[1]*U_hat[2], curl[0])
        ifftn_mpi(1j*KX[2]*U_hat[1], U_tmp[0])
        curl[0] -= U_tmp[0]
        fftn_mpi(0.5*(U[0]*U[0]+U[1]*U[1]+U[2]*U[2]), U_hat_tmp[0])        
        fftn_mpi(U[1]*curl[2]-U[2]*curl[1], dU[0])
        fftn_mpi(U[2]*curl[0]-U[0]*curl[2], dU[1])
        fftn_mpi(U[0]*curl[1]-U[1]*curl[0], dU[2])        
        for i in range(3):
            dU[i] = (dU[i]-1j*KX[i]*U_hat_tmp[0])*dealias*dt
        
    elif conv == "VortexII":
        ifftn_mpi(1j*KX[0]*U_hat[1], curl[2])
        ifftn_mpi(1j*KX[1]*U_hat[0], U_tmp[0])
        curl[2] -= U_tmp[0]
        ifftn_mpi(1j*KX[2]*U_hat[0], curl[1])
        ifftn_mpi(1j*KX[0]*U_hat[2], U_tmp[0])
        curl[1] -= U_tmp[0]
        ifftn_mpi(1j*KX[1]*U_hat[2], curl[0])
        ifftn_mpi(1j*KX[2]*U_hat[1], U_tmp[0])
        curl[0] -= U_tmp[0]        
        fftn_mpi(U[1]*curl[2]-U[2]*curl[1], dU[0])
        fftn_mpi(U[2]*curl[0]-U[0]*curl[2], dU[1])
        fftn_mpi(U[0]*curl[1]-U[1]*curl[0], dU[2])
        dU[:] *= dealias*dt

    else:
        raise TypeError, "Wrong type of convection"
                 
    # Add contribution from diffusion
    dU[:] += -nu*dt*KK*U_hat[:]

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

# initialize plot
if rank == 0:
    #im = plt.imshow(zeros((N, N)))
    #plt.colorbar(im)
    #plt.draw()
    k = []
# RK loop in time
while t < T:
    t += dt; tstep += 1
    U_hatold[:] = U_hat
    U_hatc[:] = U_hat
    for rk in range(4):
        ComputeRHS(dU, rk)
        project(dU)

        if rk < 3:
            U_hat[:] = U_hatold+b[rk]*dU
        U_hatc[:] = U_hatc+a[rk]*dU
        
    U_hat[:] = U_hatc[:]
    project(U_hat)
    for i in range(3):
        ifftn_mpi(U_hat[i], U[i])
    
    #if mod(tstep, plot_result) == 0:
        #if rank == 0:
            ##im.set_data(curl[2, 0, :, :])
            #im.set_data(U[1,:,:]*U[1,:,:])
            #im.autoscale()  
            #plt.pause(1e-6)  

    kk = comm.reduce(0.5*sum(U*U)*dx*dx*dx/L**3)
    if rank == 0:
        k.append(kk)
        #print t

if rank == 0:
    print "Time = ", time.time()-tic
    print conv
    figure()
    k = array(k)
    dkdt = (k[1:]-k[:-1])/dt
    plot(-dkdt)
    show()
