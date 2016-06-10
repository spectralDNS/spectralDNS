from spectralDNS import config, get_solver
import matplotlib.pyplot as plt
from numpy import array, pi, zeros, sum
from numpy.linalg import norm
import sys

def initialize(**kw):
    if 'NS' in config.params.solver:
        initialize1(**kw)
    
    else:
        initialize2(**kw)
        
def initialize1(U, U_hat, X, sin, cos, FFT, **kw):    
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0 
    for i in range(3):
        U_hat[i] = FFT.fftn(U[i], U_hat[i])
        
def initialize2(U, W, W_hat, X, sin, cos, FFT, cross2, K, work, **kw):
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0         
    F_tmp = work[(W_hat, 0)]
    for i in range(3):
        F_tmp[i] = FFT.fftn(U[i], F_tmp[i])

    W_hat[:] = cross2(W_hat, K, F_tmp)
    for i in range(3):
        W[i] = FFT.ifftn(W_hat[i], W[i])        

k = []
w = []
im1 = None
kold = zeros(1)
def update(comm, rank, P, P_hat, U, curl, Curl, float64, params,
           hdf5file, FFT, X, U_hat, K, work, dU, ComputeRHS, **kw):
    global k, w, im1
    if hdf5file.check_if_write(params) or params.tstep % params.compute_energy == 0:
        P[:] = FFT.ifftn(P_hat*1j, P)
        if 'NS' in params.solver:
            curl = Curl(U_hat, curl)
        if params.solver == 'VV':
            U = Curl(kw['W_hat'], U)
        
        hdf5file.write(params)
        
    if im1 is None and rank == 0 and params.plot_step > 0:
        plt.figure()
        im1 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[0,:,:,10], 100)
        plt.colorbar(im1)
        plt.draw()
        plt.pause(1e-6)
        globals().update(im1=im1)
        
    if params.tstep % params.plot_step == 0 and rank == 0 and params.plot_step > 0:
        im1.ax.clear()
        im1.ax.contourf(X[1,:,:,0], X[0,:,:,0], U[0,:,:,10], 100) 
        im1.autoscale()
        plt.pause(1e-6)

    if params.tstep % params.compute_energy == 0:
        dx, L = params.dx, params.L
        if 'NS' in params.solver:
            ww = comm.reduce(sum(curl*curl)*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            
            duidxj = work[(((3,3)+FFT.real_shape()), FFT.float, 0)]
            for i in range(3):
                for j in range(3):
                    duidxj[i,j] = FFT.ifftn(1j*K[j]*U_hat[i], duidxj[i,j]) 
            ww2 = comm.reduce(sum(duidxj*duidxj)*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            
            ddU = work[(((3,)+FFT.real_shape()), FFT.float, 0)]
            dU = ComputeRHS(dU, U_hat)
            for i in range(3):
                ddU[i] = FFT.ifftn(dU[i], ddU[i]) 
            ww3 = comm.reduce(sum(ddU*U)*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            
            #if rank == 0:
                #print ww, params.nu*ww2, ww3, ww-ww2
                
        elif params.solver == 'VV':
            U = Curl(kw['W_hat'], U)
            ww = comm.reduce(sum(kw['W'].astype(float64)*kw['W'].astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            
        kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2) # Compute energy with double precision
        kold[0] = kk
        if rank == 0:
            k.append(kk)
            w.append(ww)
            print params.t, float(kk), float(ww)
    #if params.tstep % params.compute_energy == 1:
        #if 'NS' in params.solver:
            #kk2 = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)        
            #if rank == 0:
                #print 0.5*(kk2-kold[0])/params.dt

def regression_test(comm, U, U_hat, curl, Curl, float64, rank, FFT, params, **kw):
    dx, L = params.dx, params.L
    if 'NS' in params.solver:
        curl = Curl(U_hat, curl)
        w = comm.reduce(sum(curl.astype(float64)*curl.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
    elif params.solver == 'VV':
        U = Curl(kw['W_hat'], U)
        w = comm.reduce(sum(kw['W'].astype(float64)*kw['W'].astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)

    k = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2) # Compute energy with double precision
    if rank == 0:
        assert round(k - 0.124953117517, 7) == 0
        assert round(w - 0.375249930801, 7) == 0

def additional_callback(**kw):
    pass
    
if __name__ == "__main__":
    config.update(
        {
        'nu': 0.000625,             # Viscosity
        'dt': 0.01,                 # Time step
        'T': 0.1,                   # End time
        'L': [2*pi, 2*pi, 2*pi],
        'M': [5, 5, 5],
        #'decomposition': 'pencil',
        #'Pencil_alignment': 'Y',
        #'P1': 2
        },  "triplyperiodic"
    )
    config.triplyperiodic.add_argument("--compute_energy", type=int, default=2)
    config.triplyperiodic.add_argument("--plot_step", type=int, default=10)
    #solver = get_solver(update=update, regression_test=regression_test, 
                        #additional_callback=additional_callback, 
                        #mesh="triplyperiodic")
    solver = get_solver(update=update, mesh="triplyperiodic")
    
    #solver.hdf5file.fname = "NS7.h5"
    #solver.hdf5file.components["W0"] = solver.curl[0]
    #solver.hdf5file.components["W1"] = solver.curl[1]
    #solver.hdf5file.components["W2"] = solver.curl[2]
    initialize(**vars(solver))
    solver.solve()
    
    #config.params.convection = 'Standard'
    #initialize(**vars(solver))
    #solver.solve()
    
    #config.update(dict(M=[4, 4, 4]), 'triplyperiodic')
    ##config.params.M = [4, 4, 4]
    #VVsolver = get_solver(update=update, regression_test=regression_test, 
                          #mesh="triplyperiodic", parse_args=sys.argv[1:-1]+['VV'])
    #initialize(**vars(VVsolver))
    #print "VVsolver"
    #VVsolver.solve()

    
