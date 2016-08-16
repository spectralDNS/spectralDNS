__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from .spectralinit import *

def setup():
    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()
    U     = empty((2,) + FFT.real_shape(), dtype=float)
    U_hat = empty((2,) + FFT.complex_shape(), dtype=complex)
    P     = empty(FFT.real_shape(), dtype=float)
    P_hat = empty(FFT.complex_shape(), dtype=complex)
    curl  = empty(FFT.real_shape(), dtype=float)
    dU    = empty((2,) + FFT.complex_shape(), dtype=complex)
    K2 = np.sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)
    work = work_arrays()

    class NS2DWriter(HDF5Writer):
        def update_components(self, U, U_hat, P, P_hat, FFT, **kw):
            """Transform to real data when storing the solution"""
            for i in range(2):
                U[i] = FFT.ifft2(U_hat[i], U[i])
            P = FFT.ifft2(P_hat, P)

    hdf5file = NS2DWriter({"U":U[0], "V":U[1], "P":P}, 
                          filename=params.solver+".h5",
                          chkpoint={'current':{'U':U, 'P':P}, 'previous':{}})

    return locals()

vars().update(setup())

def add_pressure_diffusion(dU, P_hat, U_hat, K, K2, K_over_K2, nu):
    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat[:] = np.sum(dU*K_over_K2, 0, out=P_hat)

    # Add pressure gradient
    dU -= P_hat*K

    # Add contribution from diffusion
    dU -= nu*K2*U_hat
    
    return dU

def ComputeRHS(dU, U_hat):
    curl_hat = work[(FFT.complex_shape(), complex, 0)]    
    U_dealias = work[((2,)+FFT.work_shape(params.dealias), float, 0)]
    curl_dealias = work[(FFT.work_shape(params.dealias), float, 0)]
    
    curl_hat = cross2(curl_hat, K, U_hat)    
    curl_dealias = FFT.ifft2(curl_hat, curl_dealias, params.dealias)
    U_dealias[0] = FFT.ifft2(U_hat[0], U_dealias[0], params.dealias)
    U_dealias[1] = FFT.ifft2(U_hat[1], U_dealias[1], params.dealias)
    dU[0] = FFT.fft2(U_dealias[1]*curl_dealias, dU[0], params.dealias)
    dU[1] = FFT.fft2(-U_dealias[0]*curl_dealias, dU[1], params.dealias)
    dU = add_pressure_diffusion(dU, P_hat, U_hat, K, K2, K_over_K2, params.nu)    
    return dU

def regression_test(**kw):
    pass

def solve():
    global dU, U, U_hat
    
    timer = Timer()
    params.t = 0.0
    params.tstep = 0
    integrate = getintegrator(**globals())   

    if params.make_profile: profiler = cProfile.Profile()
    
    dt_in = params.dt

    while params.t + params.dt <= params.T+1e-15:

        U_hat, params.dt, dt_took = integrate()

        for i in range(2): 
            U[i] = FFT.ifft2(U_hat[i], U[i])

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
    
    dU = ComputeRHS(dU, U_hat)
    
    additional_callback(fU_hat=dU, **globals())
 
    timer.final(MPI, rank)

    if params.make_profile:
        results = create_profile(**globals())
    
    regression_test(**globals())
    
    hdf5file.close()
