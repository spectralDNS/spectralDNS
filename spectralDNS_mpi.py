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
    fftn = pyfftw.interfaces.numpy_fft.fftn
    ifftn = pyfftw.interfaces.numpy_fft.ifftn
    fft2 = pyfftw.interfaces.numpy_fft.fft2
    ifft2 = pyfftw.interfaces.numpy_fft.ifft2
    # Keep fft objects in cache for efficiency
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1000)
    def empty(N, dtype="float", bytes=16):
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

# The x index is split between processes and each process owns Np 2D slices, 
# where Np = N/num_processes
#
# process 0 owns 0:Np
# process 1 owns Np:2*Np
#         .
# process rank owns rank*Np:(rank+1)*Np
#
# The variables are globally U[N, N, N], V[N, N, N], etc.,
# where indices are laid out in x, y, z direction.
# Chunks of the global matrices live on each process
#
#   U[0:Np,    :, :] lives on process 0
#   U[Np:2*Np, :, :] lives on process 1
#   etc.
#
# Convection requires communication.
# Each process does its own fft2 on its 2D slices
#
# Responsibility for performing remaining one-dimensional fft in x-direction
# is also split between processes. This is done by allocating slices to
# processes just like the regular ownership, but in z-direction. Of the global
# matrix the following responsibilities are given
#
#  process 0:  U[:, :, :Np]  
#  process 1:  U[:, :, Np:2*Np]
#  etc
#
# To perform the operations in x-direction the following matrix is needed
#
#  Ut[:, :, Np]
#
# where fft is performed along axis 0 fft(Ut, axis=0)
#        

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

#conv = convection.get(eval(sys.argv[-1]), "Standard")
conv = convection[4]

# Create the mesh
x = linspace(0, L, N+1)[:-1]
[X, Y, Z] = meshgrid(x[rank*Np:(rank+1)*Np], x, x, indexing='ij')

# Solution arrays
U = empty((Np, N, N))
V = empty((Np, N, N))
W = empty((Np, N, N))

# Fourier coefficients
U_hat = empty((Np, N, N), dtype="complex") 
V_hat = empty((Np, N, N), dtype="complex")
W_hat = empty((Np, N, N), dtype="complex")

# Arrays for mpi communication
U_sendc = empty((num_processes, Np, N, Np), dtype="complex")
U_recvc = empty((num_processes, Np, N, Np), dtype="complex")
U_send = empty((num_processes, Np, N, Np))
U_recv = empty((num_processes, Np, N, Np))

# RK4 
U_hatold = empty((Np, N, N), dtype="complex")
V_hatold = empty((Np, N, N), dtype="complex")
W_hatold = empty((Np, N, N), dtype="complex")
U_hatc = empty((Np, N, N), dtype="complex")
V_hatc = empty((Np, N, N), dtype="complex")
W_hatc = empty((Np, N, N), dtype="complex")
dU = empty((Np, N, N), dtype="complex")
dV = empty((Np, N, N), dtype="complex")
dW = empty((Np, N, N), dtype="complex")

# work arrays
U_tmp = empty((Np, N, N))
V_tmp = empty((Np, N, N))
U_hat_tmp = empty((Np, N, N), dtype="complex")
V_hat_tmp = empty((Np, N, N), dtype="complex")
W_hat_tmp = empty((Np, N, N), dtype="complex")
Utc = empty((N, N, Np), dtype="complex")
Ut = empty((N, N, Np))

curl = empty((3, Np, N, N))

kx = (mod(0.5+arange(0, N, dtype="float")/N, 1)-0.5)*2*pi/dx
[KX, KY, KZ] = meshgrid(kx[rank*Np:(rank+1)*Np], kx, kx, indexing='ij')
KK = KX**2 + KY**2 + KZ**2
dealias = array((abs(KX) < (2./3.)*max(kx))*(abs(KY) < (2./3.)*max(kx))*(abs(KZ) < (2./3.)*max(kx)), dtype=int)
Ksq = where(KK==0, 1, KK)

# RK4 parameters
a1= 1./6.; a2 = 1./3.; a3 = 1./3.; a4 = 1./6.
b1=0.5; b2=0.5; b3=1.; b4=1.;
a=[a1, a2, a3, a4]
b=[b1, b2, b3, b4]

# Taylor-Green
U[:] = sin(X)*cos(Y)*cos(Z)
V[:] =-cos(X)*sin(Y)*cos(Z)
W[:] = 0 

#def pressure():
    #U[:] = real(ifftn(U_hat))
    #V[:] = real(ifftn(V_hat))
    #p_hat = (1j*KX*fftn(U*ifftn(1j*KX*U_hat) + V*ifftn(1j*KY*U_hat) + W*ifftn(1j*KZ*U_hat))
            #+1j*KY*fftn((U*ifftn(1j*KX*V_hat)) + V*ifftn(1j*KY*V_hat) + W*ifftn(1j*KZ*V_hat))
            #+1j*KZ*fftn((U*ifftn(1j*KX*W_hat)) + V*ifftn(1j*KY*W_hat) + W*ifftn(1j*KZ*W_hat))) / Ksq
    
    #return real(ifftn(p_hat))

def project(xU, xV, xW):
    U_hat_tmp[:] = (KX*xU + KY*xV + KZ*xW)*KX / Ksq
    V_hat_tmp[:] = (KX*xU + KY*xV + KZ*xW)*KY / Ksq
    W_hat_tmp[:] = (KX*xU + KY*xV + KZ*xW)*KZ / Ksq
    xU[:] = xU - U_hat_tmp
    xV[:] = xV - V_hat_tmp
    xW[:] = xW - W_hat_tmp
    
def ifftn_mpi(fu, u):
    # Do 2D ifft2 in y-z directions on owned data
    U_hat_tmp[:] = ifft2(fu)
    
    # Set up for communicating intermediate result 
    for i in range(num_processes):
        U_sendc[i, :, :, :] = U_hat_tmp[:, :, i*Np:(i+1)*Np]
        
    # Communicate all values
    comm.Alltoall(U_sendc, U_recvc)
    
    # Place received data in chunk Utc
    for i in range(num_processes):
        Utc[i*Np:(i+1)*Np, :, :] = U_recvc[i, :, :, :]
        
    # Do ifft for final direction        
    Ut[:] = real(ifft(Utc, axis=0))
    
    # Return to owner
    for i in range(num_processes):
        U_send[i, :, :, :] = Ut[i*Np:(i+1)*Np, :, :]
    
    # Communicate all values
    comm.Alltoall(U_send, U_recv)

    # Copy to final array
    for i in range(num_processes):
        u[:, :, i*Np:(i+1)*Np] = U_recv[i, :, :, :]

def fftn_mpi(u, fu):
    # Do 2D fft2 in y-z directions on owned data
    U_hat_tmp[:] = fft2(u)
    
    # Set up for communicating intermediate result 
    for i in range(num_processes):
        U_sendc[i, :, :, :] = U_hat_tmp[:, :, i*Np:(i+1)*Np]
        
    # Communicate all values
    comm.Alltoall(U_sendc, U_recvc)
    
    # Place in chunk Utc
    for i in range(num_processes):
        Utc[i*Np:(i+1)*Np, :, :] = U_recvc[i, :, :, :]
        
    # Do fft for final direction        
    Utc[:] = fft(Utc, axis=0)
    
    # Return to owner
    for i in range(num_processes):
        U_sendc[i, :, :, :] = Utc[i*Np:(i+1)*Np, :, :]
    
    # Communicate all values
    comm.Alltoall(U_sendc, U_recvc)

    # Copy to final array 
    for i in range(num_processes):
        fu[:, :, i*Np:(i+1)*Np] = U_recvc[i, :, :, :]
        
def ComputeRHS(dU, dV, dW, rk):
    if rk > 0: # For rk=0 the correct values are already in U, V, W
        ifftn_mpi(U_hat, U)
        ifftn_mpi(V_hat, V)
        ifftn_mpi(W_hat, W)
    
    if conv == "Standard":
        pass
        #dU[:] = -dealias*(fftn(U*ifftn(1j*KX*U_hat)) + fftn(V*ifftn(1j*KY*U_hat)) + fftn(W*ifftn(1j*KZ*U_hat)))*dt
        #dV[:] = -dealias*(fftn(U*ifftn(1j*KX*V_hat)) + fftn(V*ifftn(1j*KY*V_hat)) + fftn(W*ifftn(1j*KZ*V_hat)))*dt
        #dW[:] = -dealias*(fftn(U*ifftn(1j*KX*W_hat)) + fftn(V*ifftn(1j*KY*W_hat)) + fftn(W*ifftn(1j*KZ*W_hat)))*dt
        
    elif conv == "Divergence":
        pass
        #fftUV[:] = fftn(U*V)    
        #dU[:] = -dealias*(1j*KX*fftn(U*U) + 1j*KY*fftUV + 1j*KZ*fftn(U*W))*dt  
        #dV[:] = -dealias*(1j*KX*fftUV + 1j*KY*fftn(V*V) + 1j*KZ*fftn(V*W))*dt
        #dW[:] = -dealias*(1j*KX*fftn(U*W) + 1j*KY*fftn(V*W) + 1j*KZ*fftn(W*W))*dt

    elif conv == "Skewed":
        pass
        #fftUV[:] = fftn(U*V)    
        #dU[:] = -dealias*(1j*KX*fftn(U*U) + 1j*KY*fftUV + 1j*KZ*fftn(U*W) +
                #fftn(U*ifftn(1j*KX*U_hat)) + fftn(V*ifftn(1j*KY*U_hat)) + fftn(W*ifftn(1j*KZ*U_hat)))*dt/2.
                
        #dV[:] = -dealias*(1j*KX*fftUV + 1j*KY*fftn(V*V) + 1j*KZ*fftn(V*W) +
                #fftn(U*ifftn(1j*KX*V_hat)) + fftn(V*ifftn(1j*KY*V_hat)) + fftn(W*ifftn(1j*KZ*V_hat)))*dt/2.

        #dW[:] = -dealias*(1j*KX*fftn(U*W) + 1j*KY*fftn(V*W) + 1j*KZ*fftn(W*W) +
                #fftn(U*ifftn(1j*KX*W_hat)) + fftn(V*ifftn(1j*KY*W_hat)) + fftn(W*ifftn(1j*KZ*W_hat)))*dt/2.

    elif conv == "VortexI":    
        pass
        #curl[2, :, :, :] = real(ifftn(1j*KX*V_hat))-real(ifftn(1j*KY*U_hat))
        #curl[1, :, :, :] = real(ifftn(1j*KZ*U_hat))-real(ifftn(1j*KX*W_hat))
        #curl[0, :, :, :] = real(ifftn(1j*KY*W_hat))-real(ifftn(1j*KZ*V_hat))
        
        #fftUV[:] = fftn(0.5*(U*U+V*V+W*W))
        #dU[:] = dealias*(fftn(V*curl[2, :]-W*curl[1, :]) - 1j*KX*fftUV)*dt
        #dV[:] = dealias*(fftn(W*curl[0, :]-U*curl[2, :]) - 1j*KY*fftUV)*dt
        #dW[:] = dealias*(fftn(U*curl[1, :]-V*curl[0, :]) - 1j*KY*fftUV)*dt
        
    elif conv == "VortexII":
        ifftn_mpi(1j*KX*V_hat, V_tmp)
        ifftn_mpi(1j*KY*U_hat, U_tmp)
        curl[2, :, :, :] = V_tmp-U_tmp
        ifftn_mpi(1j*KZ*U_hat, V_tmp)
        ifftn_mpi(1j*KX*W_hat, U_tmp)
        curl[1, :, :, :] = V_tmp-U_tmp
        ifftn_mpi(1j*KY*W_hat, V_tmp)
        ifftn_mpi(1j*KZ*V_hat, U_tmp)
        curl[0, :, :, :] = V_tmp-U_tmp
        
        fftn_mpi(V*curl[2, :]-W*curl[1, :], V_hat_tmp)        
        dU[:] = dealias*V_hat_tmp*dt
        fftn_mpi(W*curl[0, :]-U*curl[2, :], V_hat_tmp)
        dV[:] = dealias*V_hat_tmp*dt
        fftn_mpi(U*curl[1, :]-V*curl[0, :], V_hat_tmp)
        dW[:] = dealias*V_hat_tmp*dt

    else:
        raise TypeError, "Wrong type of convection"
                 
    # Add contribution from diffusion
    dU[:] += -nu*dt*KK*U_hat
    dV[:] += -nu*dt*KK*V_hat
    dW[:] += -nu*dt*KK*W_hat

# Transform initial data
fftn_mpi(U, U_hat)
fftn_mpi(V, V_hat)
fftn_mpi(W, W_hat)
       
# Make it divergence free in case it is not
project(U_hat, V_hat, W_hat)

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
    t += dt
    tstep += 1
    U_hatold[:] = U_hat
    V_hatold[:] = V_hat
    W_hatold[:] = W_hat
    U_hatc[:] = U_hat
    V_hatc[:] = V_hat
    W_hatc[:] = W_hat
    for rk in range(4):
        ComputeRHS(dU, dV, dW, rk)
        project(dU, dV, dW) 

        if rk < 3:
            U_hat[:] = U_hatold+b[rk]*dU
            V_hat[:] = V_hatold+b[rk]*dV
            W_hat[:] = W_hatold+b[rk]*dW
        U_hatc[:] = U_hatc+a[rk]*dU
        V_hatc[:] = V_hatc+a[rk]*dV
        W_hatc[:] = W_hatc+a[rk]*dW
        
    U_hat[:] = U_hatc[:]
    V_hat[:] = V_hatc[:]
    W_hat[:] = W_hatc[:]
    project(U_hat, V_hat, W_hat)
    ifftn_mpi(U_hat, U)
    ifftn_mpi(V_hat, V)
    ifftn_mpi(W_hat, W)
    
    #if mod(tstep, plot_result) == 0:
        #if rank == 0:
            ##im.set_data(curl[2, 0, :, :])
            #im.set_data(U[1,:,:]*U[1,:,:])
            #im.autoscale()  
            #plt.pause(1e-6)  

    kk = comm.reduce(0.5*sum(U*U+V*V+W*W)*dx*dx*dx/L**3)
    if rank == 0:
        k.append(kk)
        #print t

if rank == 0:
    print "Time = ", time.time()-tic
    figure()
    k = array(k)
    dkdt = (k[1:]-k[:-1])/dt
    plot(-dkdt)
    show()
