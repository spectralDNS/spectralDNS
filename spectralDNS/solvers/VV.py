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

# Subclass HDF5Writer for appropriate updating of real components
class VVWriter(HDF5Writer):

    def update_components(self, U, W_hat, curl, FFT, K_over_K2, **kw):
        """Transform to real data when storing the solution"""
        U = computeU(U, W_hat, work, FFT, K_over_K2)
        for i in range(3):
            curl[i] = FFT.ifftn(W_hat[i], curl[i])

def setup():
    """Set up context for the solver"""
    
    d = setupNS()
    Source = zeros((3,) + d.FFT.complex_shape(), dtype=complex) # Possible source term initialized to zero
    hdf5file = VVWriter({'U':d.U[0], 'V':d.U[1], 'W':d.U[2],
                         'curlx':d.curl[0], 'curly':d.curl[1], 'curlz':d.curl[2]},
                        chkpoint={'current':{'U':d.U, 'curl':d.curl}, 'previous':{}},
                        filename=params.solver+'.h5')
    
    d.update(dict(Source=Source,
                  hdf5file=hdf5file,
                  W_hat=d.U_hat))
    return d

def computeU(c, a, work, FFT, K_over_K2, dealias=None):
    """Compute u from curl(u)
    
    Follows from
    w = [curl(u)=] \nabla \times u
    curl(w) = \nabla^2(u) (since div(u)=0)
    FFT(curl(w)) = FFT(\nabla^2(u))
    ik \times w_hat = k^2 u_hat
    u_hat = (ik \times w_hat) / k^2
    u = iFFT(u_hat)
    """
    F_tmp = work[(a, 0)]
    F_tmp = cross2(F_tmp, K_over_K2, a)
    c[0] = FFT.ifftn(F_tmp[0], c[0], dealias)
    c[1] = FFT.ifftn(F_tmp[1], c[1], dealias)
    c[2] = FFT.ifftn(F_tmp[2], c[2], dealias)    
    return c

def backward_velocity(W_hat, U, work, FFT, K_over_K2, **context):
    """A common method for obtaining the transformed velocity
    
    Compute velocity from curl coefficients
    """
    U = computeU(U, W_hat, work, FFT, K_over_K2)
    return U

def get_curl(curl, W_hat, FFT, **context):
    for i in range(3):
        curl[i] = FFT.ifftn(W_hat[i], curl[i])
    return curl

#@profile
def ComputeRHS(rhs, w_hat, work, FFT, K, K2, K_over_K2, Source, **context):
    u_dealias = work[((3,)+FFT.work_shape(params.dealias), float, 0)]
    w_dealias = work[((3,)+FFT.work_shape(params.dealias), float, 1)]
    F_tmp = work[(rhs, 0)]

    u_dealias[:] = computeU(u_dealias, w_hat, work, FFT, K_over_K2, params.dealias)
    for i in range(3):
        w_dealias[i] = FFT.ifftn(w_hat[i], w_dealias[i], params.dealias)
    F_tmp[:] = Cross(u_dealias, w_dealias, F_tmp, work, FFT, params.dealias)
    rhs = cross2(rhs, K, F_tmp)
    rhs -= params.nu*K2*w_hat    
    rhs += Source    
    return rhs

def solve(context):
    global conv, integrate, profiler, timer
    
    timer = Timer()
    params.t = 0.0
    params.tstep = 0

    # Set up function to perform temporal integration (using params.integrator parameter)
    integrate = getintegrator(ComputeRHS, context.dU, context.W_hat, params,
                              context, additional_callback)

    conv = getConvection(params.convection)

    profiler = None
    if params.make_profile: profiler = cProfile.Profile()

    dt_in = params.dt
    
    while params.t + params.dt <= params.T+1e-15:
        
        context.W_hat, params.dt, dt_took = integrate()

        params.t += dt_took
        params.tstep += 1
        
        update(context)

        context.hdf5file.update(params, **context)
                
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
    
    context.dU = ComputeRHS(context.dU, context.W_hat, **context)
    
    additional_callback(fU_hat=context.dU, **context)

    timer.final(MPI, rank)
    
    if params.make_profile:
        results = create_profile(**context)
        
    regression_test(context)
        
    context.hdf5file.close()
