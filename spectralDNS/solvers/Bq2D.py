__author__ = "Mikael Mortensen <mikaem@math.uio.no> and Diako Darian <diako.darian@mn.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *

def setup():
    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)    
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()
    Ur     = empty((3,) + FFT.real_shape(), dtype=float)
    Ur_hat = empty((3,) + FFT.complex_shape(), dtype=complex)
    P      = empty(FFT.real_shape(), dtype=float)
    P_hat  = empty(FFT.complex_shape(), dtype=complex)
    curl   = empty(FFT.real_shape(), dtype=float)
    dU     = empty((3,) + FFT.complex_shape(), dtype=complex)
    # Create views into large data structures
    rho     = Ur[2]
    rho_hat = Ur_hat[2]
    U       = Ur[:2] 
    U_hat   = Ur_hat[:2]
    K2 = np.sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)
    work = work_arrays()
    return locals()

vars().update(setup())

hdf5file = HDF5Writer({"U":U[0], "V":U[1], "rho":rho, "P":P},
                      chkpoint={'current':{'U':Ur, 'P':P}, 'previous':{}},
                      filename="Bq2D.h5")

def update_components(Ur, Ur_hat, P, P_hat, FFT, params, **kw):
    """Transform to real data when storing the solution"""
    if hdf5file.check_if_write(params) or params.tstep % params.checkpoint == 0:
        for i in range(3):
            Ur[i] = FFT.ifft2(Ur_hat[i], Ur[i])
        P = FFT.ifft2(P_hat, P)

hdf5file.update_components = update_components

@optimizer
def add_pressure_diffusion(dU, P_hat, U_hat, rho_hat, K_over_K2, K, K2, nu, Ri, Pr):
    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat  = np.sum(dU[:2]*K_over_K2, 0, out=P_hat)
    
    P_hat -= Ri*rho_hat*K_over_K2[1]
    
    # Add pressure gradient
    dU[:2] -= P_hat*K

    # Add contribution from diffusion                      
    dU[0] -= nu*K2*U_hat[0]
    
    dU[1] -= (nu*K2*U_hat[1] + Ri*rho_hat)
    dU[2] -= nu * K2 * rho_hat/Pr  
    return dU

def ComputeRHS(dU, rk):
    Ur_dealias = work[((3,)+FFT.work_shape(params.dealias), float, 0)]
    curl_dealias = work[(FFT.work_shape(params.dealias), float, 0)]
    F_tmp = work[(dU, 0)]
    
    for i in range(3):
        Ur_dealias[i] = FFT.ifft2(Ur_hat[i], Ur_dealias[i], params.dealias)
        
    U_dealias = Ur_dealias[:2]
    rho_dealiased = Ur_dealias[2]

    F_tmp[0] = cross2(F_tmp[0], K, U_hat)
    curl_dealias = FFT.ifft2(F_tmp[0], curl_dealias, params.dealias)
    dU[0] = FFT.fft2(U_dealias[1]*curl_dealias, dU[0], params.dealias)
    dU[1] = FFT.fft2(-U_dealias[0]*curl_dealias, dU[1], params.dealias)
   
    F_tmp[0] = FFT.fft2(U_dealias[0]*rho_dealiased, F_tmp[0], params.dealias)
    F_tmp[1] = FFT.fft2(U_dealias[1]*rho_dealiased, F_tmp[1], params.dealias)
    dU[2] = -1j*(K[0]*F_tmp[0]+K[1]*F_tmp[1])
    
    #U_tmp[0] = FFT.ifft2(1j*K[0]*rho_hat, U_tmp[0])
    #U_tmp[1] = FFT.ifft2(1j*K[1]*rho_hat, U_tmp[1])          
    #F_tmp[0] = FFT.fft2(U[0]*U_tmp[0], F_tmp[0])      
    #F_tmp[1] = FFT.fft2(U[1]*U_tmp[1], F_tmp[1])    
    #dU[2] = -1.0*(F_tmp[0] + F_tmp[1])    
    
    dU = add_pressure_diffusion(dU, P_hat, U_hat, rho_hat, K_over_K2, K, K2, params.nu, params.Ri, params.Pr)
    
    return dU

def solve():
    global dU, Ur, Ur_hat, integrate, profiler, timer
    
    timer = Timer()
    params.t = 0.0
    params.tstep = 0
    
    integrate = getintegrator(**globals())

    if params.make_profile: profiler = cProfile.Profile()
    
    dt_in = params.dt    

    while params.t < params.T-1e-8:
        
        Ur_hat, params.dt, dt_took = integrate()

        for i in range(3):
            Ur[i] = FFT.ifft2(Ur_hat[i], Ur[i])

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
    
    dU = ComputeRHS(dU, Ur_hat)
    
    additional_callback(fU_hat=dU, **globals())

    timer.final(MPI, rank)
    
    if params.make_profile:
        results = create_profile(**globals())
        
    regression_test(**globals())        
        
    hdf5file.close()
