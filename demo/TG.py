from spectralDNS import config, get_solver, solve
from numpy import array, pi, zeros, sum, float64, sin, cos
from numpy.linalg import norm
import warnings

try:
    import matplotlib.pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")
    plt = None

def initialize(solver, **context):
    if 'NS' in config.params.solver:
        initialize1(solver, **context)
    
    else:
        initialize2(solver, **context)
    config.params.t = 0.0
    config.params.tstep = 0
        
def initialize1(solver, U, U_hat, X, FFT, **context):    
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0 
    for i in range(3):
        U_hat[i] = FFT.fftn(U[i], U_hat[i])
        
def initialize2(solver, U, W_hat, X, FFT, K, work, **context):
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
    solver = config.solver
    
    if (params.tstep % params.compute_energy == 0 or
          params.tstep % params.plot_step == 0 and params.plot_step > 0):
        U = solver.get_velocity(**c)
        curl = solver.get_curl(**c)
        if params.solver == 'NS':
            P = solver.get_pressure(**c)

    if plt is not None:
        if params.tstep % params.plot_step == 0 and solver.rank == 0 and params.plot_step > 0:
            if im1 is None:
                plt.figure()
                im1 = plt.contourf(c.X[1,:,:,0], c.X[0,:,:,0], U[0,:,:,10], 100)
                plt.colorbar(im1)
                plt.draw()
                globals().update(im1=im1)
            else:
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
    solver = config.solver
    dx, L = params.dx, params.L
    U = solver.get_velocity(**context)
    curl = solver.get_curl(**context)
    w = solver.comm.reduce(sum(curl.astype(float64)*curl.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
    k = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2) # Compute energy with double precision
    if solver.rank == 0:
        assert round(w - 0.375249930801, params.ntol) == 0
        assert round(k - 0.124953117517, params.ntol) == 0

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
    sol = get_solver(update=update, regression_test=regression_test,
                     mesh="triplyperiodic")

    context = sol.get_context()

    # Add curl to the stored results. For this we need to update the update_components
    # method used by the HDF5Writer class to compute the real fields that are stored
    if config.params.solver == 'NS':
        context.hdf5file.fname = "NS7.h5"
        context.hdf5file.components["curlx"] = context.curl[0]
        context.hdf5file.components["curly"] = context.curl[1]
        context.hdf5file.components["curlz"] = context.curl[2]
        def update_components(**context):
            """Overload default because we want to store the curl as well"""
            U = sol.get_velocity(**context)
            P = sol.get_pressure(**context)
            curl = sol.get_curl(**context)
            
        context.hdf5file.update_components = update_components

    initialize(sol, **context)
    solve(sol, context)
