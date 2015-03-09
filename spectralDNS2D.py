__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from MPI_knee import mpi_import, MPI
with mpi_import():
    from numpy import *
    from pylab import *
    from mpi4py import MPI
    import time
    from mpi.wrappyfftw import *
    from utilities.commandline import *

# Set up problem parameters using first hard-conded parameters, then possibly
# overloaded with commandline arguments
commandline_kwargs = parse_command_line(sys.argv[1:])
with mpi_import():
    exec("from problems.TwoD.{} import *".format(commandline_kwargs.get('problem', 'TaylorGreen')))
parameters.update(commandline_kwargs)
assert parameters['temporal'] in ['RK4', 'ForwardEuler', 'AB2']
vars().update(parameters)

# Set the size of the doubly periodic box N**2
N = 2**M
L = 2 * pi
dx = L / N

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_processes = comm.Get_size()
Np = N / num_processes

# Create the mesh
X = mgrid[rank*Np:(rank+1)*Np, :N].astype(float)*L/N

# Solution array and Fourier coefficients
# Because of real transforms and symmetries, N/2+1 coefficients are sufficient
Nf = N/2+1
Npf = Np/2+1 if rank+1 == num_processes else Np/2

U     = empty((2, Np, N))
U_hat = empty((2, N, Npf), dtype="complex")
P     = empty((Np, N))
P_hat = empty((N, Npf), dtype="complex")
curl   = empty((Np, N))
Uc_hat = empty((N, Npf), dtype="complex")
Uc_hatT = empty((Np, Nf), dtype="complex")
U_send = empty((num_processes, Np, Np/2), dtype="complex")
U_sendr = U_send.reshape((N, Np/2))

U_recv = empty((N, Np/2), dtype="complex")
fft_y = empty(N, dtype="complex")
fft_x = empty(N, dtype="complex")
plane_recv = empty(Np, dtype="complex")

# RK4 arrays
U_hat0 = empty((2, N, Npf), dtype="complex")
U_hat1 = empty((2, N, Npf), dtype="complex")
dU     = empty((2, N, Npf), dtype="complex")

# Set wavenumbers in grid
kx = fftfreq(N, 1./N)
ky = kx[:Nf].copy(); ky[-1] *= -1
K = array(meshgrid(kx, ky[rank*Np/2:(rank*Np/2+Npf)], indexing='ij'), dtype=int)
K2 = sum(K*K, 0)
K_over_K2 = array(K, dtype=float) / where(K2==0, 1, K2)

# Filter for dealiasing nonlinear convection
kmax = 2./3.*(N/2+1)
dealias = array((abs(K[0]) < kmax)*(abs(K[1]) < kmax), dtype=bool)

# RK4 parameters
a = array([1./6., 1./3., 1./3., 1./6.])*dt
b = array([0.5, 0.5, 1.])*dt

def project(u):
    u[:] -= sum(K*u, 0)*K_over_K2    

def rfft2_mpi(u, fu):
    if num_processes == 1:
        fu[:] = rfft2(u, axes=(0,1))
        return fu    
    
    Uc_hatT[:] = rfft(u, axis=1)
    Uc_hatT[:, 0] += 1j*Uc_hatT[:, -1]
    
    # Align data in x-direction
    for i in range(num_processes): 
        U_send[i] = Uc_hatT[:, i*Np/2:(i+1)*Np/2]
            
    # Communicate all values
    comm.Alltoall([U_send, MPI.DOUBLE_COMPLEX], [U_recv, MPI.DOUBLE_COMPLEX])
    
    fu[:, :Np/2] = fft(U_recv, axis=0)
        
    # Handle Nyquist frequency
    if rank == 0:        
        f = fu[:, 0]        
        fft_x[0] = f[0].real;
        fft_x[1:N/2] = 0.5*(f[1:N/2]+conj(f[:N/2:-1]))
        fft_x[N/2] = f[N/2].real        
        fu[:N/2+1, 0] = fft_x[:N/2+1]        
        fu[N/2+1:, 0] = conj(fft_x[(N/2-1):0:-1])
        
        fft_y[0] = f[0].imag
        fft_y[1:N/2] = -0.5*1j*(f[1:N/2]-conj(f[:N/2:-1]))
        fft_y[N/2] = f[N/2].imag
        fft_y[N/2+1:] = conj(fft_y[(N/2-1):0:-1])
        
        comm.Send([fft_y, MPI.DOUBLE_COMPLEX], dest=num_processes-1, tag=77)
        
    elif rank == num_processes-1:
        comm.Recv([fft_y, MPI.DOUBLE_COMPLEX], source=0, tag=77)
        fu[:, -1] = fft_y 
        
    return fu

def irfft2_mpi(fu, u):
    if num_processes == 1:
        u[:] = irfft2(fu, axes=(0,1))
        return u
        f   
    Uc_hat[:] = ifft(fu, axis=0)    
    U_sendr[:] = Uc_hat[:, :Np/2]

    comm.Alltoall([U_send, MPI.DOUBLE_COMPLEX], [U_recv, MPI.DOUBLE_COMPLEX])

    for i in range(num_processes): 
        Uc_hatT[:, i*Np/2:(i+1)*Np/2] = U_recv[i*Np:(i+1)*Np]
    
    if rank == num_processes-1:
        fft_y[:] = Uc_hat[:, -1]

    comm.Scatter(fft_y, plane_recv, root=num_processes-1)
    Uc_hatT[:, -1] = plane_recv
    
    u[:] = irfft(Uc_hatT, 1)
    return u

def ComputeRHS(dU, rk):
    if rk > 0: # For rk=0 the correct values are already in U, V, W
        U[0] = irfft2_mpi(U_hat[0], U[0])
        U[1] = irfft2_mpi(U_hat[1], U[1])

    curl[:] = irfft2_mpi(1j*(K[0]*U_hat[1] - K[1]*U_hat[0]), curl)
    dU[0] = rfft2_mpi(U[1]*curl, dU[0])
    dU[1] = rfft2_mpi(-U[0]*curl, dU[1])

    # Dealias the nonlinear convection
    dU[:] *= dealias

    # Compute pressure (To get actual pressure multiply by 1j/dt)
    P_hat[:] = sum(dU*K_over_K2, 0, out=P_hat)

    # Add pressure gradient
    dU[:] -= P_hat*K

    # Add contribution from diffusion
    dU[:] -= nu*K2*U_hat

U = initialize(**vars())

# Transform initial data
U_hat[0] = rfft2_mpi(U[0], U_hat[0])
U_hat[1] = rfft2_mpi(U[1], U_hat[1])

# Make it divergence free in case it is not
project(U_hat)

tic = time.time()
t = 0.0
tstep = 0

# initialize plot
if plot_result > 0:
    im = plt.imshow(zeros((N, N)))
    plt.colorbar(im)
    plt.draw()

# RK4 loop in time
t0 = array([time.time()])
while t < T:
    t += dt; tstep += 1

    if temporal == 'RK4':
        U_hat1[:] = U_hat0[:] = U_hat
        for rk in range(4):
            ComputeRHS(dU, rk)
            if rk < 3:
                #U_hat[:] = U_hat0 + b[rk]*dU
                U_hat[:] = U_hat0; U_hat += b[rk]*dU # Faster (no tmp array)
            U_hat1[:] += a[rk]*dU
        U_hat[:] = U_hat1[:]

    elif temporal == 'ForwardEuler' or tstep == 1:
        ComputeRHS(dU, 0)
        U_hat[:] += dU*dt
        if temporal == "AB2":
            U_hat0[:] = dU*dt

    else:
        ComputeRHS(dU, 0)
        U_hat[:] += (1.5*dU*dt - 0.5*U_hat0)
        U_hat0[:] = dU*dt

    for i in range(2): 
        U[i] = irfft2_mpi(U_hat[i], U[i])

    update(**vars())
    
    if tstep % plot_result == 0 and plot_result > 0:
        curl = irfft2_mpi(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        im.set_data(curl[:, :])
        im.autoscale()
        plt.pause(1e-6)
    
if rank == 0:
    print "Total computing time = ", time.time()-tic

regression_test(**vars())
