from cbcdns import config, get_solver
import matplotlib.pyplot as plt
from numpy import array, sqrt, random, exp, pi

def initialize(W, W_hat, fftn_mpi, X, **kwargs):
    W[:] = 0
    if config.init == 'random':
        W[:] = 0.3*random.randn(*W.shape)
        
    else:
        W[0, :, :, :] = exp(-((X[1]-pi)**2+(X[2]-pi+pi/4)**2)/(0.2)) \
                   +    exp(-((X[1]-pi)**2+(X[2]-pi-pi/4)**2)/(0.2)) \
                   -0.5*exp(-((X[1]-pi-pi/4)**2+(X[2]-pi-pi/4)**2)/(0.4))
    
    for i in range(3):
        W_hat[i] = fftn_mpi(W[i], W_hat[i])
        
    return W, W_hat

def set_source(U, Source, fftn_mpi, N, X, **kwargs):
    U[:] = 0
    if config.init == 'random':
        #U[0, :, N/2, (2*N)/3] = 200
        #U[0, :, N/3, (2*N)/3] = -50
        #U[0, :, N/2, N/3] = -200
        U[0, :, :, :] = 100*exp(-((X[1]-pi)**2+(X[2]-pi+pi/4)**2)/(0.01)) \
                +    100*exp(-((X[1]-pi)**2+(X[2]-pi-pi/4)**2)/(0.01)) \
                -0.5*100*exp(-((X[1]-pi-pi/4)**2+(X[2]-pi-pi/4)**2)/(0.04))
    
    else:
        pass
    
    Source[0] = fftn_mpi(U[0], Source[0])
    Source[1] = fftn_mpi(U[1], Source[1])
    Source[2] = fftn_mpi(U[2], Source[2])    
    
    return Source

im, im2 = None, None    
def update(t, tstep, dt, comm, rank, P, P_hat, U, W, W_hat, Curl, hdf5file, 
           Source, ifftn_mpi, X, Nf, N, **soak):    
    global im, im2
    if tstep == 1 and rank == 0:
        plt.figure()
        im = plt.quiver(X[1, 0], X[2, 0], 
                        U[1, 0], U[2, 0], pivot='mid', scale=2)    
        
        plt.figure()
        im2 = plt.imshow(W[0, 0, :, ::-1].T)
        plt.colorbar(im2)
        plt.draw()

        plt.pause(1e-6)
        globals().update(im=im, im2=im2)
    
    if tstep % config.write_result == 0:
        U = Curl(W_hat, U)
        P[:] = sqrt(W[0]*W[0] + W[1]*W[1] + W[2]*W[2])
        hdf5file.write(tstep)

    if tstep == 10:
        Source[:] = 0
        
    if tstep % config.plot_result == 0 and rank == 0:
        im.set_UVC(U[1, 0], U[2, 0])
        im2.set_data(W[0, 0, :, ::-1].T)
        im2.autoscale()
        plt.pause(1e-6)
    
    print "Time = ", t
    
def finalize(rank, Nf, X, U, W_hat, Curl, **soak):
    global im
    im.set_UVC(U[1, 0], U[2, 0])
    plt.pause(1e-6)

if __name__ == "__main__":
    config.update(
        {
        'solver': 'VV',
        'nu': 0.000625,              # Viscosity
        'dt': 0.01,                  # Time step
        'T': 50,                     # End time
        'write_result': 100
        }
    )        
    config.Isotropic.add_argument("--init", default='random', choices=('random', 'vortex'))
    config.Isotropic.add_argument("--plot_result", type=int, default=10) # required to allow overloading through commandline
    solver = get_solver(update)
    assert config.decomposition == 'slab'
    solver.W, solver.W_hat = initialize(**vars(solver))
    solver.Source = set_source(**vars(solver))
    solver.solve()
    finalize(**vars(solver))

