from pylab import figure, plot, show, array

compute_energy = 2         # Compute solution energy every..

def initialize(config, **kw):
    if config.solver == 'NS':
        initialize1(**kw)
    
    else:
        initialize2(**kw)
        
def initialize1(U, U_hat, X, sin, cos, fftn_mpi, **kw):    
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0 
    for i in range(3):
        U_hat[i] = fftn_mpi(U[i], U_hat[i])
        
def initialize2(U, W, W_hat, X, sin, cos, fftn_mpi, ifftn_mpi, F_tmp, 
                cross2, K, **kw):
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0         
    for i in range(3):
        F_tmp[i] = fftn_mpi(U[i], F_tmp[i])

    W_hat[:] = cross2(W_hat, K, F_tmp)
    for i in range(3):
        W[i] = ifftn_mpi(W_hat[i], W[i])        

k = []
w = []
def update(t, tstep, dt, comm, rank, P, P_hat, U, curl, float64, dx, L, sum, 
           hdf5file, ifftn_mpi, **kw):
    global k, w
    if tstep % config.write_result == 0 or tstep % config.write_yz_slice[1] == 0:
        P = ifftn_mpi(P_hat*1j, P)
        hdf5file.write(U, P, tstep)

    if tstep % compute_energy == 0:
        kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx*dx*dx/L**3/2) # Compute energy with double precision
        ww = comm.reduce(sum(curl.astype(float64)*curl.astype(float64))*dx*dx*dx/L**3/2)
        if rank == 0:
            k.append(kk)
            w.append(ww)
            print t, float(kk), float(ww)

def finalize(rank, dt, **soak):
    global k

    if rank == 0:
        figure()
        k = array(k)
        dkdt = (k[1:]-k[:-1])/dt
        plot(-dkdt)
        show()

if __name__ == "__main__":
    from cbcdns import config, get_solver
    config.update(
        {
        'nu': 0.000625,             # Viscosity
        'dt': 0.01,                 # Time step
        'T': 0.1,                   # End time
        }
    )
    
    solver = get_solver()
    initialize(**vars(solver))
    solver.update = update
    solver.solve()
    #finalize(**vars(solver))
