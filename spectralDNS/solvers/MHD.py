__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *

def setup():
    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()
    K2 = np.sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)

    UB = empty((6,) + FFT.real_shape(), dtype=float)
    P  = empty(FFT.real_shape(), dtype=float)
    curl = empty((3,) + FFT.real_shape(), dtype=float)
    UB_hat = empty((6,) + FFT.complex_shape(), dtype=complex)
    P_hat = empty(FFT.complex_shape(), dtype=complex)
    dU = empty((6,) + FFT.complex_shape(), dtype=complex)
    Source = None

    # Create views into large data structures
    U     = UB[:3]
    U_hat = UB_hat[:3]
    B     = UB[3:]
    B_hat = UB_hat[3:]

    work = work_arrays()
    
    class MHDWriter(HDF5Writer):
        def update_components(self, UB, UB_hat, P, P_hat, FFT, **kw):
            """Transform to real data when storing the solution"""
            for i in range(6):
                UB[i] = FFT.ifftn(UB_hat[i], UB[i])
            P = FFT.ifftn(P_hat, P)

    hdf5file = MHDWriter({'U':U[0], 'V':U[1], 'W':U[2], 'P':P,
                         'Bx':B[0], 'By':B[1], 'Bz':B[2]},
                         chkpoint={'current':{'UB':UB, 'P':P}, 'previous':{}},
                         filename="MHD.h5")

    return locals()

# Put the datastructures in this solvers global namespace
vars().update(setup())

def backward():
    for i in range(6):
        UB[i] = FFT.ifftn(UB_hat[i], UB[i])

def forward():
    for i in range(6):
        UB_hat[i] = FFT.fftn(UB[i], UB_hat[i])

def set_Elsasser(c, F_tmp, K):
    c[:3] = -1j*(K[0]*(F_tmp[:, 0] + F_tmp[0, :])
                +K[1]*(F_tmp[:, 1] + F_tmp[1, :])
                +K[2]*(F_tmp[:, 2] + F_tmp[2, :]))/2.0

    c[3:] =  1j*(K[0]*(F_tmp[0, :] - F_tmp[:, 0])
                +K[1]*(F_tmp[1, :] - F_tmp[:, 1])
                +K[2]*(F_tmp[2, :] - F_tmp[:, 2]))/2.0
    return c

def divergenceConvection(z0, z1, c, dealias=None):
    """Divergence convection using Elsasser variables
    z0=U+B
    z1=U-B
    """
    F_tmp = work[((3, 3) + FFT.complex_shape(), complex, 0)]
    for i in range(3):
        for j in range(3):
            F_tmp[i, j] = FFT.fftn(z0[i]*z1[j], F_tmp[i, j], dealias)

    c = set_Elsasser(c, F_tmp, K)
    return c

def ComputeRHS(dU, UB_hat):
    """Compute and return entire rhs contribution"""
    UB_dealias = work[((6,)+FFT.work_shape(params.dealias), float, 0)]
    for i in range(6):
        UB_dealias[i] = FFT.ifftn(UB_hat[i], UB_dealias[i], params.dealias)

    U_dealias = UB_dealias[:3]
    B_dealias = UB_dealias[3:]
    # Compute convective term and place in dU
    dU = divergenceConvection(U_dealias+B_dealias, U_dealias-B_dealias, dU,
                              params.dealias)

    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat[:] = np.sum(dU[:3]*K_over_K2, 0)

    # Add pressure gradient
    dU[:3] -= P_hat*K

    # Add contribution from diffusion
    dU[:3] -= params.nu*K2*U_hat
    dU[3:] -= params.eta*K2*B_hat

    return dU

def solve():
    global dU, UB, UB_hat, integrate, timer, profiler

    timer = Timer()
    params.t = 0.0
    params.tstep = 0
    # Set up function to perform temporal integration (using params.integrator parameter)
    integrate = getintegrator(**globals())

    if params.make_profile: profiler = cProfile.Profile()

    dt_in = params.dt
    while params.t + params.dt <= params.T+1e-15:

        UB_hat, params.dt, dt_took = integrate()

        params.t += dt_took
        params.tstep += 1

        update(**globals())

        hdf5file.update(**globals())

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

    dU = ComputeRHS(dU, UB_hat)

    additional_callback(fU_hat=dU, **globals())

    timer.final(MPI, rank)

    if params.make_profile:
        results = create_profile(**vars())

    regression_test(**globals())

    hdf5file.close()
