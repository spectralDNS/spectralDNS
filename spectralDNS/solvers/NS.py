__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from .spectralinit import *
from ..mesh.triplyperiodic import Curl

def setup():
    """Set up context for solver
    
    All data structures and variables defined here will be added to the global
    namespace of the current solver.
    """
    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()    
    K2 = np.sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)    
    
    U     = empty((3,) + FFT.real_shape(), dtype=float)  
    U_hat = empty((3,) + FFT.complex_shape(), dtype=complex)
    P     = empty(FFT.real_shape(), dtype=float)
    P_hat = empty(FFT.complex_shape(), dtype=complex)

    # RHS array
    dU     = empty((3,) + FFT.complex_shape(), dtype=complex)
    curl   = empty((3,) + FFT.real_shape(), dtype=float)   
    Source = None
    work = work_arrays()
    
    # Subclass HDF5Writer for appropriate updating of real components
    class NSWriter(HDF5Writer):

        def update_components(self, U, U_hat, P, P_hat, FFT, **kw):
            """Transform to real data when storing the solution"""
            for i in range(3):
                U[i] = FFT.ifftn(U_hat[i], U[i])
            P = FFT.ifftn(P_hat, P)
    
    hdf5file = NSWriter({'U':U[0], 'V':U[1], 'W':U[2], 'P':P},
                        chkpoint={'current':{'U':U, 'P':P}, 'previous':{}},
                        filename=params.solver+'.h5')

    return config.ParamsBase(locals())

#vars().update(setup())

def forward_velocity(U, U_hat, FFT, **kw):
    """A common method for obtaining the transformed velocity"""
    for i in range(3):
        U_hat[i] = FFT.fftn(U[i], U_hat[i])
    return U_hat

def backward_velocity(U, U_hat, FFT, **kw):
    """A common method for obtaining the velocity"""
    for i in range(3):
        U[i] = FFT.ifftn(U_hat[i], U[i])
    return U

def standardConvection(c, U_dealias, U_hat, work, FFT, K, dealias=None):
    """c_i = u_j du_i/dx_j"""
    gradUi = work[(U_dealias, 2, False)]
    for i in range(3):
        for j in range(3):
            gradUi[j] = FFT.ifftn(1j*K[j]*U_hat[i], gradUi[j], dealias)
        c[i] = FFT.fftn(np.sum(U_dealias*gradUi, 0), c[i], dealias)
    return c

def divergenceConvection(c, U_dealias, work, FFT, K, dealias=None, add=False):
    """c_i = div(u_i u_j)"""
    if not add: c.fill(0)
    UUi_hat = work[(c, 0, False)]
    for i in range(3):
        UUi_hat[i] = FFT.fftn(U_dealias[0]*U_dealias[i], UUi_hat[i], dealias)
    c[0] += 1j*np.sum(K*UUi_hat, 0)
    c[1] += 1j*K[0]*UUi_hat[1]
    c[2] += 1j*K[0]*UUi_hat[2]
    UUi_hat[0] = FFT.fftn(U_dealias[1]*U_dealias[1], UUi_hat[0], dealias)
    UUi_hat[1] = FFT.fftn(U_dealias[1]*U_dealias[2], UUi_hat[1], dealias)
    UUi_hat[2] = FFT.fftn(U_dealias[2]*U_dealias[2], UUi_hat[2], dealias)
    c[1] += (1j*K[1]*UUi_hat[0] + 1j*K[2]*UUi_hat[1])
    c[2] += (1j*K[1]*UUi_hat[1] + 1j*K[2]*UUi_hat[2])
    return c

#@profile
def Cross(a, b, c, work, FFT, dealias=None):
    """c_k = F_k(a x b)"""
    Uc = work[(a, 2, False)]
    Uc = cross1(Uc, a, b)
    c[0] = FFT.fftn(Uc[0], c[0], dealias)
    c[1] = FFT.fftn(Uc[1], c[1], dealias)
    c[2] = FFT.fftn(Uc[2], c[2], dealias)
    return c

def getConvection(convection):
    """Return function used to compute convection"""
    if convection == "Standard":

        def Conv(dU, U_hat, work, FFT, K):
            U_dealias = work[((3,)+FFT.work_shape(params.dealias),
                              float, 0, False)]
            for i in range(3):
                U_dealias[i] = FFT.ifftn(U_hat[i], U_dealias[i], params.dealias)
            dU = standardConvection(dU, U_dealias, U_hat, work, FFT, K, params.dealias)
            dU[:] *= -1
            return dU

    elif convection == "Divergence":

        def Conv(dU, U_hat, work, FFT, K):
            U_dealias = work[((3,)+FFT.work_shape(params.dealias),
                              float, 0, False)]
            for i in range(3):
                U_dealias[i] = FFT.ifftn(U_hat[i], U_dealias[i], params.dealias)
            dU = divergenceConvection(dU, U_dealias, work, FFT, K, params.dealias, False)
            dU[:] *= -1
            return dU

    elif convection == "Skewed":

        def Conv(dU, U_hat, work, FFT, K):
            U_dealias = work[((3,)+FFT.work_shape(params.dealias),
                              float, 0, False)]
            for i in range(3):
                U_dealias[i] = FFT.ifftn(U_hat[i], U_dealias[i], params.dealias)
            dU = standardConvection(dU, U_dealias, U_hat, work, FFT, K, params.dealias)
            dU = divergenceConvection(dU, U_dealias, work, FFT, K, params.dealias, True)
            dU *= -0.5
            return dU

    elif convection == "Vortex":

        #@profile
        def Conv(dU, U_hat, work, FFT, K):
            U_dealias = work[((3,)+FFT.work_shape(params.dealias),
                              float, 0, False)]
            curl_dealias = work[((3,)+FFT.work_shape(params.dealias),
                                 float, 1, False)]
            for i in range(3):
                U_dealias[i] = FFT.ifftn(U_hat[i], U_dealias[i], params.dealias)

            curl_dealias[:] = Curl(U_hat, curl_dealias, work, FFT, K, params.dealias)
            dU = Cross(U_dealias, curl_dealias, dU, work, FFT, params.dealias)
            return dU

    return Conv

@optimizer
def add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu):
    """Add contributions from pressure and diffusion to the rhs"""

    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat = np.sum(dU*K_over_K2, 0, out=P_hat)

    # Subtract pressure gradient
    dU -= P_hat*K

    # Subtract contribution from diffusion
    dU -= nu*K2*U_hat

    return dU

#@profile
def ComputeRHS(dU, U_hat, K, K2, P_hat, K_over_K2, FFT, work):
    """Compute and return entire rhs contribution"""
    dU = conv(dU, U_hat, work, FFT, K)

    dU = add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, params.nu)

    return dU

#@profile
def solve(context):
    #global dU, U, U_hat, conv, integrate, profiler, timer
    global conv

    timer = Timer()
    params.t = 0.0
    params.tstep = 0
    args = (context.K, context.K2, context.P_hat, context.K_over_K2, 
            context.FFT, context.work)
    integrate = getintegrator(ComputeRHS, context.dU, context.U_hat, params, args)
    conv = getConvection(params.convection)

    profiler = None
    if params.make_profile: profiler = cProfile.Profile()

    context.update(dict(conv=conv, integrate=integrate,
                        profiler=profiler, timer=timer))

    dt_in = params.dt

    while params.t + params.dt <= params.T+1e-15:

        context.U_hat, params.dt, dt_took = integrate()
        
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

    dU = ComputeRHS(context.dU, context.U_hat, *args)

    additional_callback(fU_hat=context.dU, **context)

    timer.final(MPI, rank)

    if params.make_profile:
        results = create_profile(**context)

    regression_test(context)

    context.hdf5file.close()
