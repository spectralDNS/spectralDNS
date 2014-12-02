__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from numpy import *
from pylab import *
import time
from wrappyfftw import *
from commandline import *

params = {
    'M': 6,
    'temporal': 'RK4',
    'plot_result': 10,         # Show an image every..
    'nu': 0.001,
    'dt': 0.005,
    'T': 50.0
}
commandline_kwargs = parse_command_line(sys.argv[1:])
params.update(commandline_kwargs)
assert params['temporal'] in ['RK4', 'ForwardEuler', 'AB2']
vars().update(params)

# Set the size of the doubly periodic box N**2
N = 2**M
L = 2 * pi
dx = L / N

# Create the mesh
x = linspace(0, L, N+1)[:-1]
X = array(meshgrid(x, x, indexing='ij'))

# Solution array and Fourier coefficients
# Because of real transforms and symmetries, N/2+1 coefficients are sufficient
Nf = N/2+1
U     = empty((2, N, N))
U_hat = empty((2, N, Nf), dtype="complex")

# RK4 arrays
U_hat0 = empty((2, N, Nf), dtype="complex")
U_hat1 = empty((2, N, Nf), dtype="complex")
dU     = empty((2, N, Nf), dtype="complex")

# work arrays
U_tmp  = empty((2, N, N))
F_tmp  = empty((2, N, Nf), dtype="complex")
Uc_hat = empty((N, Nf), dtype="complex")
curl   = empty((N, N))

# Set wavenumbers in grid
kx = fftfreq(N, 1./N)
ky = kx[:Nf].copy(); ky[-1] *= -1
KX = array(meshgrid(kx, ky, indexing='ij'), dtype=int)
KK = sum(KX*KX, 0)
KX_over_Ksq = array(KX, dtype=float) / where(KK==0, 1, KK)

# Filter for dealiasing nonlinear convection
dealias = array((abs(KX[0]) < (2./3.)*max(kx))*(abs(KX[1]) < (2./3.)*max(kx)), dtype=bool)

# RK4 parameters
a = [1./6., 1./3., 1./3., 1./6.]
b = [0.5, 0.5, 1.]

def pressure():
    """Pressure is not really used, but may be recovered from the velocity.
    """
    for i in range(2):
        for j in range(2):
            U_tmp[j] = irfft2(1j*KX[j]*U_hat[i])
        F_tmp[i] = rfft2(sum(U*U_tmp, 0))
    return irfft2(1j*sum(KX_over_Ksq*F_tmp, 0))

def project(u):
    u[:] -= sum(KX*u, 0)*KX_over_Ksq
            
def ComputeRHS(dU, rk):
    if rk > 0: # For rk=0 the correct values are already in U, V, W
        U[:] = irfft2(U_hat)
    
    curl[:] = irfft2(1j*(KX[0]*U_hat[1] - KX[1]*U_hat[0]))
    dU[0] = rfft2(U[1]*curl)
    dU[1] = rfft2(-U[0]*curl)
        
    # Dealias the nonlinear convection
    dU[:] *= dealias*dt
        
    # Add pressure gradient
    dU[:] -= sum(dU*KX_over_Ksq, 0)*KX
    
    # Add contribution from diffusion
    dU[:] += -nu*dt*KK*U_hat
    
# Taylor-Green initialization
#U[0] = sin(X[0])*cos(X[1])
#U[1] =-cos(X[0])*sin(X[1])

# Initialize two vortices
#w=exp(-((X[0]-pi)**2+(X[1]-pi+pi/4)**2)/(0.2))+exp(-((X[0]-pi)**2+(X[1]-pi-pi/4)**2)/(0.2))-0.5*exp(-((X[0]-pi-pi/4)**2+(X[1]-pi-pi/4)**2)/(0.4))
#w_hat = rfft2(w)
#U[0] = irfft2(1j*KX_over_Ksq[1]*w_hat)
#U[1] = irfft2(-1j*KX_over_Ksq[0]*w_hat)

# Transform initial data
U_hat[:] = rfft2(U)

# Make it divergence free in case it is not
project(U_hat)

# initialize plot and list k for storing energy
im = plt.imshow(zeros((N, N)))
plt.colorbar(im)
plt.draw()

tic = time.time()
t = 0.0
tstep = 0

# RK4 loop in time
t0 = time.time()
while t < T:
    t += dt; tstep += 1
    
    if temporal == 'RK4':
        U_hat1[:] = U_hat0[:] = U_hat
        for rk in range(4):
            ComputeRHS(dU, rk)        
            project(dU)
            if rk < 3:
                U_hat[:] = U_hat0 + b[rk]*dU
            U_hat1[:] += a[rk]*dU        
        U_hat[:] = U_hat1[:]
        
    elif temporal == 'ForwardEuler' or tstep == 1:
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

    U[:] = irfft2(U_hat)
    
    # From here on it's only postprocessing
    if tstep % plot_result == 0:
        curl[:] = irfft2(1j*KX[0]*U_hat[1]-1j*KX[1]*U_hat[0])
        im.set_data(curl[:, :])
        im.autoscale()  
        plt.pause(1e-6) 
        
    print tstep, time.time()-t0
    t0 = time.time()
            
print "Time = ", time.time()-tic
plt.figure()
plt.quiver(X[0,::2,::2], X[1,::2,::2], U[0,::2,::2], U[1,::2,::2], pivot='mid', scale=2)
plt.draw();plt.show()
#print "Energy numeric = ", sum(U*U)*dx*dx/L**2
#u0 = sin(X[0])*cos(X[1])*exp(-2.*nu*t)
#u1 =-sin(X[1])*cos(X[0])*exp(-2.*nu*t)
#print "Energy exact   = ", sum(u0*u0+u1*u1)*dx*dx/L**2
#print "Error   = ", sum(sqrt((U[0]-u0)**2+(U[1]-u1)**2))*dx*dx/L**2
