from cbcdns import config, get_solver
from pylab import plt, zeros, cm

compute_kinetic = -10
compute_amplitude = -10

def initialize(X, U, Ur, Ur_hat, exp, sin, cos, tanh, rho, Np, N, pi, fft2_mpi, **kwargs):


#--------------------------------------------------------------------------
#                 1. case
#--------------------------------------------------------------------------   

    #Um = 0.5*(U1 - U2)
    #U[1] = A*sin(X[0])
    #for i in range(Np):
    #for j in range(N):
        #if 0.0 <= X[1][i,j] < pi/2.0:
        #U[0][i,j] = U1 - Um*exp((X[1][i,j] - pi/2.0)/delta) 
        #elif pi/2.0 <= X[1][i,j] < pi:
        #U[0][i,j] = U2 + Um*exp(-1.0*(X[1][i,j] - pi/2.0)/delta) 
        #elif pi <= X[1][i,j] < 3*pi/2.0:
        #U[0][i,j] = U2 + Um*exp((X[1][i,j] - 3*pi/2.0)/delta) 
        #elif 3*pi/2. <= X[1][i,j] < 2*pi:
        #U[0][i,j] = U1 - Um*exp(-1.0*(X[1][i,j] - 3*pi/2.0)/delta)
    #for i in range(Np):
    #for j in range(N):
        #if 0.0 <= X[1][i,j] < pi/2.0:
            #rho[i, j] = 1.0 + 0.5*exp((X[1][i,j] - pi/2.0)/delta)
        #elif pi/2.0 <= X[1][i,j] < pi:
        #rho[i, j] = 2.0 - 0.5*exp(-1*(X[1][i,j] - pi/2.0)/delta)
        #elif pi <= X[1][i,j] < 3*pi/2.0:
        #rho[i, j] = 2.0 - 0.5*exp((X[1][i,j] - 3*pi/2.0)/delta)
        #elif 3*pi/2. <= X[1][i,j] < 2*pi:
        #rho[i, j] = 1.0 + 0.5*exp(-1.0*(X[1][i,j] - 3*pi/2.0)/delta)
   
#--------------------------------------------------------------------------
#                 Working case L = 2pi (This is the correct initial value)
#-------------------------------------------------------------------------- 
# This case shows SKHI for law resolution, but by increasing resolution
# e.g. M = 10, most of the SKHI disappear. No sign of periodic motion in 
# y-direction
  
  #rho[:] = sin(X[1]/2)
    #Um = 0.5*(U1 - U2)
    #U[1] = A*sin(0.5*X[0])
    #for i in range(Np):
    #for j in range(N):
        #if 0.0 <= X[1][i,j] < pi:
        #U[0][i,j] = U1 - Um*exp((X[1][i,j] - pi)/delta) 
        #elif pi <= X[1][i,j] < 2*pi:
        #U[0][i,j] = U2 + Um*exp(-1.0*(X[1][i,j] - pi)/delta) 
        #elif 2*pi <= X[1][i,j] < 3*pi:
        #U[0][i,j] = U2 + Um*exp((X[1][i,j] - 3*pi)/delta) 
        #elif 3*pi <= X[1][i,j] < 4*pi:
        #U[0][i,j] = U1 - Um*exp(-1.0*(X[1][i,j] - 3*pi)/delta)
    ##for i in range(Np):
    ##for j in range(N):
        ##if 0.0 <= X[1][i,j] < pi:
            ##rho[i, j] = 1.0 + 0.5*exp((X[1][i,j] - pi)/delta)
        ##elif pi <= X[1][i,j] < 2*pi:
        ##rho[i, j] = 2.0 - 0.5*exp(-1*(X[1][i,j] - pi)/delta)
        ##elif 2*pi <= X[1][i,j] < 3*pi:
        ##rho[i, j] = 2.0 - 0.5*exp((X[1][i,j] - 3*pi)/delta)
        ##elif 3*pi <= X[1][i,j] < 4*pi:
        ##rho[i, j] =1.0 + 0.5*exp(-1.0*(X[1][i,j] - 3*pi)/delta)

    #for i in range(Np):
    #for j in range(N):
        #if 0.0 <= X[1][i,j] < 2*pi:
        #rho[i, j] = tanh((X[1][i,j]-(pi))/delta)
        #elif 2*pi <= X[1][i,j] < 4*pi:
          #rho[i, j] = -tanh((X[1][i,j]-(3*pi))/delta)

#--------------------------------------------------------------------------
#                 Working case L = 2pi
#-------------------------------------------------------------------------- 
# This case shows SKHI for law resolution, but by increasing resolution
# e.g. M = 10, most of the SKHI disappear. No sign of periodic motion in 
# y-direction
  
    Um = 0.5*(config.U1 - config.U2)
    U[1] = config.A*sin(0.5*X[0])
    U[0, :, :N/4] = config.U1 - Um*exp((X[1,:, :N/4] - 0.5*pi)/config.delta) 
    U[0, :, N/4:N/2] = config.U2 + Um*exp(-1.0*(X[1, :, N/4:N/2] - 0.5*pi)/config.delta) 
    U[0, :, N/2:3*N/4] = config.U2 + Um*exp((X[1, :, N/2:3*N/4] - 1.5*pi)/config.delta) 
    U[0, :, 3*N/4:] = config.U1 - Um*exp(-1.0*(X[1, :, 3*N/4:] - 1.5*pi)/config.delta)

    
    #for i in range(Np):
        #for j in range(N):
            #if 0.0 <= X[1][i,j] < 0.5*pi:
                #U[0][i,j] = config.U1 - Um*exp((X[1][i,j] - 0.5*pi)/config.delta) 
            #elif 0.5*pi <= X[1][i,j] < pi:
                #U[0][i,j] = config.U2 + Um*exp(-1.0*(X[1][i,j] - 0.5*pi)/config.delta) 
            #elif pi <= X[1][i,j] < 1.5*pi:
                #U[0][i,j] = config.U2 + Um*exp((X[1][i,j] - 1.5*pi)/config.delta) 
            #elif 1.5*pi <= X[1][i,j] < 2*pi:
                #U[0][i,j] = config.U1 - Um*exp(-1.0*(X[1][i,j] - 1.5*pi)/config.delta)

                
    #for i in range(Np):
    #for j in range(N):
        #if 0.0 <= X[1][i,j] < pi:
            #rho[i, j] = 1.0 + 0.5*exp((X[1][i,j] - pi)/delta)
        #elif pi <= X[1][i,j] < 2*pi:
        #rho[i, j] = 2.0 - 0.5*exp(-1*(X[1][i,j] - pi)/delta)
        #elif 2*pi <= X[1][i,j] < 3*pi:
        #rho[i, j] = 2.0 - 0.5*exp((X[1][i,j] - 3*pi)/delta)
        #elif 3*pi <= X[1][i,j] < 4*pi:
        #rho[i, j] =1.0 + 0.5*exp(-1.0*(X[1][i,j] - 3*pi)/delta)

    for i in range(Np):
        for j in range(N):
            if 0.0 <= X[1][i,j] < pi:
                rho[i, j] = tanh((X[1][i,j]-(0.5*pi))/config.delta)
            elif pi <= X[1][i,j] < 2*pi:
                rho[i, j] = -tanh((X[1][i,j]-(1.5*pi))/config.delta)
          
    for i in range(3):
        Ur_hat[i] = fft2_mpi(Ur[i], Ur_hat[i]) 

#--------------------------------------------------------------------------
#                 Martines et al. 2006
#--------------------------------------------------------------------------  
# In this case the velocity and the density are given by the error-function
# This case was run for Pr = 12 Ri = 0.12 and Re = 6000 delta = 0.005 M = 10
# dt = 0.003. SKHI are observed. Almost the same behavior as Mashayek case.
    #U[1] = A*sin(4*(X[0]+X[1])/9.) #- A*sin(4*(X[0]+X[1])/18.)/1.5
    #for i in range(Np):
    #for j in range(N):
        #if 0.0 <= X[1][i,j] < 4.5*pi:
        #U[0][i,j] = math.erf((pi**0.5)*(X[1][i,j]-3*pi)/delta)
        #elif 4.5*pi <= X[1][i,j] < 9*pi:
        #U[0][i, j] = -math.erf((pi**0.5)*(X[1][i,j]-6*pi)/delta)
    ##for i in range(Np):
    ##for j in range(N):
        ##if 0.0 <= X[1][i,j] < 4.5*pi:
        ##U[1][i,j] *= 2*exp(-1.0*(X[1][i,j]-2.5*pi)**2/delta)/(pi**0.5)
        ##elif 4.5*pi <= X[1][i,j] < 9*pi:
        ##U[1][i,j] *= 2*exp(-1.0*(X[1][i,j]-6.5*pi)**2/delta)/(pi**0.5)
            
    ##rho[:] = A*sin(X[0])      
    #for i in range(Np):
    #for j in range(N):
        #if 0.0 <= X[1][i,j] < 4.5*pi:
        #rho[i, j] = -math.erf((pi**0.5)*0.4*(X[1][i,j]-3*pi)/delta)/0.4
        #elif 4.5*pi <= X[1][i,j] < 9*pi:
            #rho[i, j] = math.erf((pi**0.5)*0.4*(X[1][i,j]-6*pi)/delta)/0.4

    #for i in range(Np):
    #for j in range(N):
        #if 0.0 <= X[1][i,j] < pi:
        #omega_pert[i,j] = A*(cosh((X[1][i,j]-0.5*pi)/delta))**(-2) * ((k0**2 + 2.0-6.0*(tanh((X[1][i,j]-0.5*pi)/delta))**2)*cos(k0*(X[0][i,j]-pi)) - bb*(0.25*k0**2 + 2.0-6.0*(tanh((X[1][i,j]-0.5*pi)/delta))**2)*cos(0.5*k0*(X[0][i,j]-pi)))   
        #omega[i,j]  = (cosh((X[1][i,j]-0.5*pi)/delta))**(-2) + omega_pert[i,j]
        #elif pi <= X[1][i,j] < 2*pi: 
        #omega_pert[i,j] = A*(cosh((X[1][i,j]-1.5*pi)/delta))**(-2) * ((k0**2 + 2.0-6.0*(tanh((X[1][i,j]-1.5*pi)/delta))**2)*cos(k0*(X[0][i,j]-pi)) - bb*(0.25*k0**2 + 2.0-6.0*(tanh((X[1][i,j]-1.5*pi)/delta))**2)*cos(0.5*k0*(X[0][i,j]-pi)))   
        #omega[i,j]  = -(cosh((X[1][i,j]-1.5*pi)/delta))**(-2) - omega_pert[i,j]

#--------------------------------------------------------------------------
#                 From Mashayek & Peltier 2012
#--------------------------------------------------------------------------  
# This case was run for Pr = 12 Ri = 0.12 and Re = 6000 delta = 0.005 M = 10
# SKHI are observed
    #U[1] = A*sin(X[0])
    #for i in range(Np):
    #for j in range(N):
        #if 0.0 <= X[1][i,j] < 2*pi:
        #U[0][ i, j] = tanh((X[1][i,j]-pi)/delta)
        #elif 2*pi <= X[1][i,j] < 4*pi:
        #U[0][i, j] = -tanh((X[1][i,j]-3*pi)/delta)
    #rho[:] = A*sin(X[0])       
    #for i in range(Np):
    #for j in range(N):
        #if 0.0 <= X[1][i,j] < 2*pi:
        #rho[i, j] = tanh(1.1*(X[1][i,j]-pi)/delta)
        #elif 2*pi <= X[1][i,j] < 4*pi:
          #rho[i, j] = -tanh(1.1*(X[1][i,j]-3*pi)/delta)
          
#--------------------------------------------------------------------------
#                     Density Test
#--------------------------------------------------------------------------
# This is a test for the density
    ##U[0] = 0.0
    ##U[1] = 0.0
    #for i in range(Np):
    #for j in range(N):
        #rho[i,j] = sin(X[0][i,j]+X[1][i,j])
        #U[0][i,j] = 0.0
        #U[1][i,j] = 0.0
#--------------------------------------------------------------------------
#                     Taylor-Green Test
#--------------------------------------------------------------------------
# This test shows that the solver for the velocity is correct

    #U[0] = sin(X[0])*cos(X[1])
    #U[1] =-cos(X[0])*sin(X[1])
    #rho[:] = 0.0         
    return U, rho

im, im2 = None, None
def update(t, tstep, comm, rank, rho, N, L, curl, K, ifft2_mpi, U_hat, 
           P_hat, P, hdf5file, **kwargs):
    global im, im2
        
    if tstep == 1:
        fig, ax = plt.subplots(1,1)
        fig.suptitle('Density', fontsize=20)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        im = ax.imshow(zeros((N, N)),cmap=cm.bwr, extent=[0, L, 0, L])
        plt.colorbar(im)
        plt.draw() 

        fig2, ax2 = plt.subplots(1,1)
        fig2.suptitle('Vorticity', fontsize=20)   
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        im2 = ax2.imshow(zeros((N, N)),cmap=cm.bwr, extent=[0, L, 0, L])
        plt.colorbar(im2)
        plt.draw()
        globals().update(dict(im=im, im2=im2))

    if tstep % config.write_result == 0:
        P = ifft2_mpi(P_hat*1j, P)
        hdf5file.write(tstep)           

    if tstep % config.plot_result == 0 and config.plot_result > 0:
        print tstep
        curl[:] = ifft2_mpi(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        #curl = ifft2_mpi(1j*K[0]*U_hat[1]-1j*K[1]*U_hat[0], curl)
        im.set_data(rho[:, :].T)
        im.autoscale()
        plt.pause(1e-6)
        im2.set_data(curl[:,:].T)
        im2.autoscale()
        plt.pause(1e-6)
        
    if tstep == 1 or tstep % config.write_result == 0 or tstep % config.write_yz_slice[1] == 0:
        print tstep
        #hdf5file.write(Ur, curl, tstep)
    
    # Compute energy with double precision
    #kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx*dx/L**2/2) 
    #if rank == 0 and debug == True:
        #print "%d %2.8f %2.8f"%(tstep, time.time()-t0[0], kk)
        #t0[0] = time.time()

def regression_test(comm, U, X, dx, L, nu, t, sin, cos, sum, float64, exp, 
                    rank, **kwargs):
    print "---------regression_test--------------"
    # Check accuracy
    #kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx*dx/L**2/2)
    #u0 = sin(X[0])*cos(X[1])*exp(-2.*nu*t)
    #u1 =-sin(X[1])*cos(X[0])*exp(-2.*nu*t)
    #k1 = comm.reduce(sum(u0*u0+u1*u1)*dx*dx/L**2/2) # Compute energy with double precision)
    #if rank == 0:
        #print "Energy (exact, numeric, error)  = (%2.6f, %2.6f, %2.4e) " %(k1, kk, k1-kk)
        #assert abs(k1-kk)<1.e-10

if __name__ == "__main__":
    config.update(
    {
    'plot_result': 10,
    'nu': 1.0e-05,
    'dt': 0.001,
    'T': 1.0,
    'U1':-0.5,
    'U2':0.5,
    'l0': 0.001,    # Smoothing parameter
    'A': 0.01,      # Amplitude of perturbation
    'Ri': 0.167,    # Richardson number
    'Pr': 12.0,     # Prantl number
    'delta': 0.1,   # Width of perturbations
    'bb': 0.8,
    'k0': 2
    }
    )
    solver = get_solver(update)
    initialize(**vars(solver))
    solver.solve()
