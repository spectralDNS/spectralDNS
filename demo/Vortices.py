from pylab import plt, randn, array, sqrt

def initialize(W, W_hat, fftn_mpi, **kwargs):
    W[:] = 0
    W[:] = 0.3*randn(*W.shape)
    for i in range(3):
        W_hat[i] = fftn_mpi(W[i], W_hat[i])
        
    return W, W_hat

def set_source(U, Source, fftn_mpi, N, **kwargs):
    U[:] = 0
    U[0, :, N/2, (2*N)/3] = 20
    U[0, :, N/3, (2*N)/3] = -10
    U[0, :, N/2, N/3] = -20

    Source[0] = fftn_mpi(U[0], Source[0])
    Source[1] = fftn_mpi(U[1], Source[1])
    Source[2] = fftn_mpi(U[2], Source[2])
    return Source

im=None    
def update(t, tstep, dt, comm, rank, P, P_hat, U, W, W_hat, Curl, hdf5file, 
           Source, ifftn_mpi, X, Nf, **soak):
    
    if tstep == 1 and rank == 0:
        plt.figure()
        im = plt.quiver(X[1, 0], X[2, 0], 
                        U[1, 0], U[2, 0], pivot='mid', scale=2)    
        plt.pause(1e-6)
        globals().update(im=im)
    
    if tstep % config.write_result == 0:
        U = Curl(W_hat, U)
        P[:] = sqrt(W[0]*W[0] + W[1]*W[1] + W[2]*W[2])
        hdf5file.write(U, P, tstep)

    if tstep == 100:
        Source[:] = 0
        
    if tstep % 2 == 0 and rank == 0:
        global im
        im.set_UVC(U[1, 0], U[2, 0])
        plt.pause(1e-6)
    
    print "Time = ", t
    
def finalize(rank, Nf, X, U, W_hat, Curl, **soak):
    global im
    im.set_UVC(U[1, 0], U[2, 0])
    plt.pause(1e-6)

if __name__ == "__main__":
    from cbcdns import config, get_solver
    config.update(
        {
        'nu': 0.000625,             # Viscosity
        'dt': 0.01,                  # Time step
        'T': 50,                    # End time
        'write_result': 100
        }
    )
    assert config.decomposition == 'slab'
        
    solver = get_solver()
    solver.update = update
    solver.W, solver.W_hat = initialize(**vars(solver))
    solver.Source = set_source(**vars(solver))
    solver.solve()
    finalize(**vars(solver))

