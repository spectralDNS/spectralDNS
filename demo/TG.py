from spectralDNS import config, get_solver
import matplotlib.pyplot as plt
from numpy import array, pi, zeros, sum
from numpy.linalg import norm
import mpi4py
import h5py

def initialize(**kw):
    if config.solver == 'NS':
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
def update(comm, rank, P, P_hat, U, curl, Curl, float64, dx, L, 
           hdf5file, FFT, X, U_hat, K, work, dU, ComputeRHS, **kw):
    global k, w, im1
    if hdf5file.check_if_write(config.tstep) or config.tstep % config.compute_energy == 0:
        P[:] = FFT.ifftn(P_hat*1j, P)
        if config.solver == 'NS':
            curl = Curl(U_hat, curl)
        
        #hdf5file.write(config.tstep)
        
    if im1 is None and rank == 0 and config.plot_step > 0:
        plt.figure()
        im1 = plt.contourf(X[1,:,:,0], X[0,:,:,0], U[0,:,:,10], 100)
        plt.colorbar(im1)
        plt.draw()
        plt.pause(1e-6)
        globals().update(im1=im1)
        
    if config.tstep % config.plot_step == 0 and rank == 0 and config.plot_step > 0:
        im1.ax.clear()
        im1.ax.contourf(X[1,:,:,0], X[0,:,:,0], U[0,:,:,10], 100) 
        im1.autoscale()
        plt.pause(1e-6)

    if config.tstep % config.compute_energy == 0:
        if config.solver == 'NS':
            ww = comm.reduce(sum(curl*curl)*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            
            duidxj = work[(((3,3)+FFT.real_shape()), FFT.float, 0)]
            for i in range(3):
                for j in range(3):
                    duidxj[i,j] = FFT.ifftn(1j*K[j]*U_hat[i], duidxj[i,j]) 
            ww2 = comm.reduce(sum(duidxj*duidxj)*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            
            ddU = work[(((3,)+FFT.real_shape()), FFT.float, 0)]
            dU = ComputeRHS(dU, 0)
            for i in range(3):
                ddU[i] = FFT.ifftn(dU[i], ddU[i]) 
            ww3 = comm.reduce(sum(ddU*U)*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            
            if rank == 0:
                print ww, config.nu*ww2, ww3, ww-ww2
                
        elif config.solver == 'VV':
            U = Curl(kw['W_hat'], U)
            ww = comm.reduce(sum(kw['W'].astype(float64)*kw['W'].astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            
        kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2) # Compute energy with double precision
        kold[0] = kk
        if rank == 0:
            k.append(kk)
            w.append(ww)
            print config.t, float(kk), float(ww)
            
    if config.tstep % config.compute_energy == 1:
        if config.solver == 'NS':
            kk2 = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)        
            if rank == 0:
                print 0.5*(kk2-kold[0])/config.dt

def regression_test(comm, U, curl, float64, dx, L, rank,FFT, **kw):
    import numpy as np
    import os
    if os.path.isfile("out2.npy"):
        U_tmp = np.load("out2.npy")
        print "Difference is %.2e relative to %.2e"  % (np.linalg.norm(U-U_tmp), np.linalg.norm(U))
    np.save("out2.npy",U)


    if os.path.isfile("out2.h5"):
        fin = h5py.File("out2.h5", "r", driver="mpio", comm=FFT.comm)
        U_tmp = fin["U"]
        print "Difference is %.2e relative to %.2e"  % (norm(U-U_tmp), norm(U))
        fin.close()
    
    f = h5py.File("out2.h5", "w", driver="mpio", comm=FFT.comm)
    f.create_dataset("U", shape=U.shape, dtype=U.dtype)
    f["U"][:] = U
    f.close()

    if config.solver == 'NS':
        w = comm.reduce(sum(curl.astype(float64)*curl.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
    elif config.solver == 'VV':
        U = Curl(W_hat, U)
        w = comm.reduce(sum(kw['W'].astype(float64)*kw['W'].astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)

    k = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2) # Compute energy with double precision
    if rank == 0:
        assert round(k - 0.124953117517, 7) == 0
        assert round(w - 0.375249930801, 7) == 0

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
    solver = get_solver(update=update, mesh="triplyperiodic",regression_test=regression_test)
    """
    solver.hdf5file.fname = "NS7.h5"
    solver.hdf5file.components["W0"] = solver.curl[0]
    solver.hdf5file.components["W1"] = solver.curl[1]
    solver.hdf5file.components["W2"] = solver.curl[2]
    """
    initialize(**vars(solver))
    solver.solve()
"""
    from numpy.linalg import norm
    initialize(**vars(solver))
    solver.solve()

    U = solver.U.copy()
    initialize(**vars(solver))
    solver.solve()

    Uc = solver.U.copy()
    assert norm(U-Uc) == 0
    """


