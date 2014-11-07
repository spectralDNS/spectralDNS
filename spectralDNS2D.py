from numpy import *
from pylab import *
import time
import pyfftw
from commands import getoutput
from os import getpid, path

# Monkey patches for fft
fft2 = pyfftw.interfaces.numpy_fft.fft2
ifft2 = pyfftw.interfaces.numpy_fft.ifft2

# Keep fft objects in cache for efficiency
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1000)

N = 2**3
L = 2 * pi
dx = L / N
dt = eval(sys.argv[-2])
nu = 0.01
plot_result = 10000
T = 0.4
convection = {0: "Standard",
              1: "Divergence",
              2: "Skewed",
              3: "VortexI",
              4: "VortexII"}

conv = convection.get(eval(sys.argv[-1]), "Standard")

def empty(N, dtype="float", bytes=16):
    return pyfftw.n_byte_align_empty(N, bytes, dtype=dtype)

x = linspace(0, L, N+1)[:-1]
y = x.copy()
[X, Y] = meshgrid(x, y)
U = empty((N, N))
V = empty((N, N))
U_hatold = empty((N, N), dtype="complex")
V_hatold = empty((N, N), dtype="complex")
U_hatc = empty((N, N), dtype="complex")
V_hatc = empty((N, N), dtype="complex")
dU = empty((N, N), dtype="complex")
dV = empty((N, N), dtype="complex")
fftUV = empty((N, N), dtype="complex")
dUdx_hat = empty((N, N), dtype="complex")
dUdy_hat = empty((N, N), dtype="complex")
curl = empty((N, N))

kx = (mod(0.5+arange(0, N, dtype="float")/N, 1)-0.5)*2*pi/dx
ky = (mod(0.5+arange(0, N, dtype="float")/N, 1)-0.5)*2*pi/dx
[KX, KY] = meshgrid(kx, ky)
KK = KX**2 + KY**2
dealias = array((abs(KX) < (2./3.)*max(kx))*(abs(KY) < (2./3.)*max(ky)), dtype=int)
Ksq = where(KK==0, 1, KK)

# RK parameters
a=[1./6., 1./3., 1./3., 1./6.]
b=[0.5, 0.5, 1., 1.]

#w=exp(-((X-pi)**2+(Y-pi+pi/4)**2)/(0.2))+exp(-((X-pi)**2+(Y-pi-pi/4)**2)/(0.2))-0.5*exp(-((X-pi-pi/4)**2+(Y-pi-pi/4)**2)/(0.4))
#w_hat = fft2(w)
#psi_hat = w_hat / Ksq
#U[:] = real(ifft2(1j*KY*psi_hat))
#V[:] = real(ifft2(-1j*KX*psi_hat))

# Taylor-Green
U[:] = -sin(Y)*cos(X)
V[:] = sin(X)*cos(Y)

def pressure():
    U[:] = real(ifft2(U_hat))
    V[:] = real(ifft2(V_hat))
    p_hat = (1j*KX*fft2(U*ifft2(1j*KX*U_hat) + V*ifft2(1j*KY*U_hat))
            +1j*KY*fft2((U*ifft2(1j*KX*V_hat)) + V*ifft2(1j*KY*V_hat))) / Ksq
    return real(ifft2(p_hat))

def project(xU, xV):
    dUdx_hat[:] = (KX*xU + KY*xV)*KX / Ksq
    dUdy_hat[:] = (KX*xU + KY*xV)*KY / Ksq
    xU[:] = xU - dUdx_hat
    xV[:] = xV - dUdy_hat

def ComputeRHS(dU, dV, rk):
    if rk>0:
        U[:] = real(ifft2(U_hat))
        V[:] = real(ifft2(V_hat))
    
    if conv == "Standard":
        dU[:] = -dealias*(fft2(U*ifft2(1j*KX*U_hat)) + fft2(V*ifft2(1j*KY*U_hat)))*dt
        dV[:] = -dealias*(fft2(U*ifft2(1j*KX*V_hat)) + fft2(V*ifft2(1j*KY*V_hat)))*dt
        
    elif conv == "Divergence":
        fftUV[:] = fft2(U*V)    
        dU[:] = -dealias*(1j*KX*fft2(U*U) + 1j*KY*fftUV)*dt                
        dV[:] = -dealias*(1j*KX*fftUV + 1j*KY*fft2(V*V))*dt

    elif conv == "Skewed":
        fftUV[:] = fft2(U*V)    
        dU[:] = -dealias*(1j*KX*fft2(U*U) + 1j*KY*fftUV +
                fft2(U*ifft2(1j*KX*U_hat)) + fft2(V*ifft2(1j*KY*U_hat)))*dt/2.
                
        dV[:] = -dealias*(1j*KX*fftUV + 1j*KY*fft2(V*V) +
                fft2(U*ifft2(1j*KX*V_hat)) + fft2(V*ifft2(1j*KY*V_hat)))*dt/2.

    elif conv == "VortexI":    
        curl[:] = real(ifft2(1j*KX*V_hat))-real(ifft2(1j*KY*U_hat))
        fftUV[:] = fft2(0.5*(U*U+V*V))    
        dU[:] = dealias*(fft2(V*curl) - 1j*KX*fftUV)*dt
        dV[:] = dealias*(fft2(-U*curl) - 1j*KY*fftUV)*dt
        
    elif conv == "VortexII":
        curl[:] = real(ifft2(1j*KX*V_hat))-real(ifft2(1j*KY*U_hat))
        dU[:] = dealias*(fft2(V*curl))*dt
        dV[:] = dealias*(fft2(-U*curl))*dt

    else:
        raise TypeError, "Wrong type of convection"
                 
    # Add contribution from diffusion
    dU[:] += -nu*dt*KK*U_hat
    dV[:] += -nu*dt*KK*V_hat
    
# Transform initial data
U_hat = fft2(U)
V_hat = fft2(V)

# Make it divergence free in case it is not
project(U_hat, V_hat)

tic = time.time()
t = 0.0
tstep = 0

# initialize plot
im = plt.imshow(zeros((N, N)))
plt.colorbar(im)
plt.draw()

# RK loop in time
while t < T:
    t += dt
    tstep += 1
    U_hatold[:] = U_hat
    V_hatold[:] = V_hat
    U_hatc[:] = U_hat
    V_hatc[:] = V_hat
    for rk in range(4):
        ComputeRHS(dU, dV, rk)
        project(dU, dV) 

        if rk < 3:
            U_hat[:] = U_hatold+b[rk]*dU
            V_hat[:] = V_hatold+b[rk]*dV
        U_hatc[:] = U_hatc+a[rk]*dU
        V_hatc[:] = V_hatc+a[rk]*dV
        
    U_hat[:] = U_hatc[:]
    V_hat[:] = V_hatc[:]
    project(U_hat, V_hat)
    U[:] = real(ifft2(U_hat))
    V[:] = real(ifft2(V_hat))
    
    if mod(tstep, plot_result) == 0:
        curl[:] = real(ifft2(1j*KX*V_hat))-real(ifft2(1j*KY*U_hat))
        im.set_data(curl[::-1, :])
        #im.set_data(U*U+V*V)
        im.autoscale()  
        plt.pause(1e-6)  

#print time.time() - tic
print "Energy numeric = ", sum(U*U+V*V)*dx*dx/L**2
u0=-sin(Y)*cos(X)*exp(-2.*nu*t)
u1=sin(X)*cos(Y)*exp(-2.*nu*t)
print "Energy exact   = ", sum(u0*u0+u1*u1)*dx*dx/L**2

u0=-sin(Y)*cos(X)*exp(-2.*nu*t)
u1=sin(X)*cos(Y)*exp(-2.*nu*t)
print "Error   = ", sum((U-u0)**2+(V-u1)**2)*dx*dx/L**2

#print who()
