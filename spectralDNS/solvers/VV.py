__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-01-02"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
Velocity-vorticity formulation

Derived by taking the curl of the momentum equation and solving for the curl.
The velocity that is required in the convective term is computed from the curl
in the computeU method.

This solver inherits most features from the NS solver. The global namespace 
of the NS solver is modified to avoid having two namespaces filled with arrays. 
Overloading just a few routines.
"""
from NS import *

setupNS = setup
def setup():
    d = setupNS()
    # Rename variable since we are working with a vorticity formulation
    Source = zeros((3,) + FFT.complex_shape(), dtype=complex) # Possible source term initialized to zero
    
    # Subclass HDF5Writer for appropriate updating of real components
    class VVWriter(HDF5Writer):

        def update_components(self, U, W_hat, curl, FFT, **kw):
            """Transform to real data when storing the solution"""
            U = computeU(W_hat, U)
            for i in range(3):
                curl[i] = FFT.ifftn(W_hat[i], curl[i])
    curl = d['curl']
    hdf5file = VVWriter({'U':U[0], 'V':U[1], 'W':U[2],
                         'curlx':curl[0], 'curly':curl[1], 'curlz':curl[2]},
                        chkpoint={'current':{'U':U, 'curl':curl}, 'previous':{}},
                        filename=params.solver+'.h5')
    
    d.update(dict(Source=Source,
                  hdf5file=hdf5file,
                  W_hat=d['U_hat']))
    return d

context = setupNS.func_globals
context.update(setup())
vars().update(context)

def computeU(a, c, dealias=None):
    """Compute u from curl(u)
    
    Follows from
    w = [curl(u)=] \nabla \times u
    curl(w) = \nabla^2(u) (since div(u)=0)
    FFT(curl(w)) = FFT(\nabla^2(u))
    ik \times w_hat = k^2 u_hat
    u_hat = (ik \times w_hat) / k^2
    u = iFFT(u_hat)
    """
    global work, K_over_K2, FFT
    F_tmp = work[(a, 0)]
    F_tmp = cross2(F_tmp, K_over_K2, a)
    c[0] = FFT.ifftn(F_tmp[0], c[0], dealias)
    c[1] = FFT.ifftn(F_tmp[1], c[1], dealias)
    c[2] = FFT.ifftn(F_tmp[2], c[2], dealias)    
    return c

def backward_velocity():
    """A common method for obtaining the transformed velocity
    
    Compute velocity from curl coefficients
    """
    global W_hat, U
    U = computeU(W_hat, U)
    return U

#@profile
def ComputeRHS(dU, W_hat):
    global work, FFT, K, K2, Source
    U_dealias = work[((3,)+FFT.work_shape(params.dealias), float, 0)]
    W_dealias = work[((3,)+FFT.work_shape(params.dealias), float, 1)]
    F_tmp = work[(dU, 0)]
    
    U_dealias[:] = computeU(W_hat, U_dealias, params.dealias)
    for i in range(3):
        W_dealias[i] = FFT.ifftn(W_hat[i], W_dealias[i], params.dealias)
    F_tmp[:] = Cross(U_dealias, W_dealias, F_tmp, params.dealias)
    dU = cross2(dU, K, F_tmp)    
    dU -= params.nu*K2*W_hat    
    dU += Source    
    return dU

def solve():
    global dU, W, W_hat, conv, integrate, profiler
    
    timer = Timer()
    params.t = 0.0
    params.tstep = 0
    # Set up function to perform temporal integration (using params.integrator parameter)
    integrate = getintegrator(**globals())
    conv = getConvection(params.convection)

    if params.make_profile: profiler = cProfile.Profile()

    dt_in = params.dt
    
    while params.t + params.dt <= params.T+1e-15:
        
        W_hat, params.dt, dt_took = integrate()

        #for i in range(3):
            #W[i] = FFT.ifftn(W_hat[i], W[i])

        params.t += dt_took
        params.tstep += 1
                 
        hdf5file.update(**globals())
        
        update(**globals())
        
        timer()
        
        if params.tstep == 1 and params.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()

        #Make sure that the last step hits T exactly.
        if params.t + params.dt >= params.T:
            params.dt = params.T - params.t
            if params.dt <= 1.e-14:
                break

    params.dt = dt_in
    
    dU = ComputeRHS(dU, W_hat)
    
    additional_callback(fU_hat=dU, **globals())

    timer.final(MPI, rank)
    
    if params.make_profile:
        results = create_profile(**globals())
        
    regression_test(**globals())
        
    hdf5file.close()
