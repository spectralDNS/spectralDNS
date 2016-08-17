from spectralDNS import config, get_solver
import matplotlib.pyplot as plt
from numpy import array, sqrt, random, exp, pi

def initialize(curl, W_hat, FFT, X, **context):
    curl[:] = 0
    if config.params.init == 'random':
        curl[:] = 0.3*random.randn(*curl.shape)
        
    else:
        curl[0, :, :, :] = exp(-((X[1]-pi)**2+(X[2]-pi+pi/4)**2)/(0.2)) \
                   +    exp(-((X[1]-pi)**2+(X[2]-pi-pi/4)**2)/(0.2)) \
                   -0.5*exp(-((X[1]-pi-pi/4)**2+(X[2]-pi-pi/4)**2)/(0.4))
    
    for i in range(3):
        W_hat[i] = FFT.fftn(curl[i], W_hat[i])
        
def set_source(U, Source, FFT, X, **context):
    U[:] = 0
    if config.params.init == 'random':
        #U[0, :, N/2, (2*N)/3] = 200
        #U[0, :, N/3, (2*N)/3] = -50
        #U[0, :, N/2, N/3] = -200
        U[0, :, :, :] = 100*exp(-((X[1]-pi)**2+(X[2]-pi+pi/4)**2)/0.01) \
                   +    100*exp(-((X[1]-pi)**2+(X[2]-pi-pi/4)**2)/0.01) \
                   -0.5*100*exp(-((X[1]-pi-pi/4)**2+(X[2]-pi-pi/4)**2)/0.04)
    
    else:
        pass
    
    Source[0] = FFT.fftn(U[0], Source[0])
    Source[1] = FFT.fftn(U[1], Source[1])
    Source[2] = FFT.fftn(U[2], Source[2])    

im, im2 = None, None    
def update(context):    
    global im, im2
    c = context
    params = config.params
    
    if (params.tstep % params.plot_step == 0 and params.plot_step > 0 or
        params.tstep == 1):
        P = c.FFT.ifftn(c.P_hat*1j, c.P)
        U = solver.get_velocity(**c)
        curl = solver.get_curl(**c)
    
    if params.tstep == 1 and solver.rank == 0:
        plt.figure()
        im = plt.quiver(c.X[1, 0], c.X[2, 0], 
                        U[1, 0], U[2, 0], pivot='mid', scale=2)    
        
        plt.figure()
        im2 = plt.imshow(curl[0, 0, :, ::-1].T)
        plt.colorbar(im2)
        plt.draw()

        plt.pause(1e-6)
        globals().update(im=im, im2=im2)
    
    if params.tstep == 10:
        c.Source[:] = 0
        
    if params.tstep % params.plot_step == 0 and solver.rank == 0:
        im.set_UVC(U[1, 0], U[2, 0])
        im2.set_data(curl[0, 0, :, ::-1].T)
        im2.autoscale()
        plt.pause(1e-6)
    
    print "Time = ", params.t
    
def regression_test(context):
    global im
    U = solver.get_velocity(**context)
    im.set_UVC(U[1, 0], U[2, 0])
    plt.pause(1e-6)

if __name__ == "__main__":
    config.update(
        {
        'nu': 0.000625,              # Viscosity
        'dt': 0.01,                  # Time step
        'T': 50,                     # End time
        'write_result': 10
        }, 'triplyperiodic'
    )
    config.triplyperiodic.add_argument("--init", default='random', choices=('random', 'vortex'))
    config.triplyperiodic.add_argument("--plot_step", type=int, default=10) # required to allow overloading through commandline
    solver = get_solver(update=update, 
                        regression_test=regression_test,
                        mesh="triplyperiodic")
    
    assert config.params.decomposition == 'slab'
    context = solver.setup()
    initialize(**context)
    set_source(**context)
    solver.solve(context)
