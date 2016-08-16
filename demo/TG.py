from spectralDNS import config, get_solver

import matplotlib.pyplot as plt
from numpy import array, pi, zeros, sum, float64,sin, cos
from numpy.linalg import norm
import sys

def initialize(solver, **kw):
    if 'NS' in config.params.solver:
        initialize1(solver, **kw)
    
    else:
        initialize2(solver, **kw)
        
def initialize1(solver, U, U_hat, X, FFT, **kw):    
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0 
    for i in range(3):
        U_hat[i] = FFT.fftn(U[i], U_hat[i])
        
def initialize2(solver, U, W_hat, X, FFT, K, work, **kw):
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0
    F_tmp = work[(W_hat, 0)]
    for i in range(3):
        F_tmp[i] = FFT.fftn(U[i], F_tmp[i])

    W_hat = solver.cross2(W_hat, K, F_tmp)

k = []
w = []
im1 = None
kold = zeros(1)
def update(context):
    global k, w, im1
    c = context
    params = config.params
    
    if (params.tstep % params.compute_energy == 0 or
        params.tstep % params.plot_step == 0 and solver.rank == 0 and params.plot_step > 0):
        P = c.FFT.ifftn(c.P_hat*1j, c.P)
        U = solver.get_velocity(**c)
        curl = solver.get_curl(**c)
                
    if im1 is None and solver.rank == 0 and params.plot_step > 0:
        plt.figure()
        im1 = plt.contourf(c.X[1,:,:,0], c.X[0,:,:,0], U[0,:,:,10], 100)
        plt.colorbar(im1)
        plt.draw()
        plt.pause(1e-6)
        globals().update(im1=im1)
        
    if params.tstep % params.plot_step == 0 and solver.rank == 0 and params.plot_step > 0:
        im1.ax.clear()
        im1.ax.contourf(c.X[1,:,:,0], c.X[0,:,:,0], U[0,:,:,10], 100) 
        im1.autoscale()
        plt.pause(1e-6)

    if params.tstep % params.compute_energy == 0:
        dx, L = params.dx, params.L
        #if 'NS' in params.solver:
            #ww = comm.reduce(sum(curl*curl)*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            
            #duidxj = work[(((3,3)+FFT.real_shape()), FFT.float, 0)]
            #for i in range(3):
                #for j in range(3):
                    #duidxj[i,j] = FFT.ifftn(1j*K[j]*U_hat[i], duidxj[i,j]) 
            #ww2 = comm.reduce(sum(duidxj*duidxj)*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            
            #ddU = work[(((3,)+FFT.real_shape()), FFT.float, 0)]
            #dU = ComputeRHS(dU, U_hat)
            #for i in range(3):
                #ddU[i] = FFT.ifftn(dU[i], ddU[i]) 
            #ww3 = comm.reduce(sum(ddU*U)*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            
            #if rank == 0:
                #print ww, params.nu*ww2, ww3, ww-ww2
            
        ww = solver.comm.reduce(sum(curl.astype(float64)*curl.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
        kk = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2) # Compute energy with double precision
        kold[0] = kk
        if solver.rank == 0:
            k.append(kk)
            w.append(ww)
            print params.t, float(kk), float(ww)
    #if params.tstep % params.compute_energy == 1:
        #if 'NS' in params.solver:
            #kk2 = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)        
            #if rank == 0:
                #print 0.5*(kk2-kold[0])/params.dt

def regression_test(context):
    params = config.params
    dx, L = params.dx, params.L
    solver = regression_test.solver
    U = solver.get_velocity(**context)
    curl = solver.get_curl(**context)

    w = solver.comm.reduce(sum(curl.astype(float64)*curl.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
    k = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2) # Compute energy with double precision
    if solver.rank == 0:
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
        'planner_effort': {'dct': 'FFTW_EXHAUSTIVE'},
        #'decomposition': 'pencil',
        #'Pencil_alignment': 'Y',
        #'P1': 2
        },  "triplyperiodic"
    )
    config.triplyperiodic.add_argument("--compute_energy", type=int, default=2)
    config.triplyperiodic.add_argument("--plot_step", type=int, default=2)
    solver = get_solver(update=update, regression_test=regression_test, 
                        additional_callback=additional_callback, 
                        mesh="triplyperiodic")

    context = solver.setup()

    # Add curl to the stored results. For this we need to update the update_components
    # method used by the HDF5Writer class to compute the real fields that are stored
    context.hdf5file.fname = "NS7.h5"
    context.hdf5file.components["W0"] = context.curl[0]
    context.hdf5file.components["W1"] = context.curl[1]
    context.hdf5file.components["W2"] = context.curl[2]    
    def update_components(FFT, U, U_hat, P, P_hat, curl, work, K, **context):
        """Overload default because we want to store the curl as well"""
        for i in range(3):
            U[i] = FFT.ifftn(U_hat[i], U[i])
        P = FFT.ifftn(P_hat, P)
        curl = solver.get_curl(curl, U_hat, work, FFT, K)
        
    context.hdf5file.update_components = update_components
    initialize(solver, **context)
    solver.solve(context)
