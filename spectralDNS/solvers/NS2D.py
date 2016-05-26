__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *
from ..optimization import optimizer
import numpy as np
import spectralDNS.maths.integrators

def initializeContext(context,args):
    context.NS2D = {}
    U = context.mesh_vars["U"]
    P = context.mesh_vars["P"]
    FFT = context.FFT

    assert context.decomposition == 'line'

    context.hdf5file = HDF5Writer(context, {"U":U[0], "V":U[1], "P":P}, context.solver_name+".h5")
    # Set up function to perform temporal integration (using config.integrator parameter)
    integrate = spectralDNS.maths.integrators.getintegrator(context,ComputeRHS,f=None,g=None,ginv=None,hphi=None,gexp=None)

    context.time_integrator["integrate"] = integrate

    # Shape of work arrays used in convection with dealiasing. Different shape whether or not padding is involved
    context.mesh_vars["work_shape"] = FFT.real_shape_padded() if context.dealias_name == '3/2-rule' else FFT.real_shape()

def add_pressure_diffusion(context,dU,U_hat):
    K2 = context.mesh_vars["K2"]
    K = context.mesh_vars["K"]
    K_over_K2 = context.mesh_vars["K_over_K2"]
    P_hat = context.mesh_vars["P_hat"]
    nu = context.model_params["nu"]

    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat[:] = np.sum(dU*K_over_K2, 0, out=P_hat)

    # Add pressure gradient
    dU -= P_hat*K

    # Add contribution from diffusion
    dU -= nu*K2*U_hat
    
    return dU

def ComputeRHS(context,U,U_hat,dU, rk):
    float = context.types["float"]
    complex = context.types["complex"]
    work_shape = context.mesh_vars["work_shape"]
    K = context.mesh_vars["K"]
    FFT = context.FFT

    curl_hat = context.work[(FFT.complex_shape(), complex, 0)]    

    U_dealiased = context.work[((2,)+work_shape, float, 0)]
    curl_dealiased = context.work[(work_shape,float,0)]
    
    curl_hat = cross2(curl_hat, K, U_hat)
    curl_dealiased[:] = FFT.ifft2(curl_hat, curl_dealiased, context.dealias_name)
    U_dealiased[0] = FFT.ifft2(U_hat[0], U_dealiased[0], context.dealias_name)
    U_dealiased[1] = FFT.ifft2(U_hat[1], U_dealiased[1], context.dealias_name)
    dU[0] = FFT.fft2(U_dealiased[1]*curl_dealiased, dU[0], context.dealias_name)
    dU[1] = FFT.fft2(-U_dealiased[0]*curl_dealiased, dU[1], context.dealias_name)
    dU = add_pressure_diffusion(context,dU,U_hat)
    return dU

def solve(context):
    U_hat = context.mesh_vars["U_hat"]
    U = context.mesh_vars["U"]
    dU = context.mesh_vars["dU"]
    dt = context.time_integrator["dt"]

    T = context.model_params["T"]
    t = context.model_params["t"]

    timer = Timer()
    tstep = 0
    FFT = context.FFT

    while t + dt <= T + 1.e-15: #The 1.e-15 term is for rounding errors
        dt_prev = dt 
        kwargs = {
                "additional_callback":context.callbacks["additional_callback"],
                "t":t,
                "dt":dt,
                "tstep": tstep,
                "T": T,
                "context":context,
                "ComputeRHS":ComputeRHS
                }
        U_hat[:],dt,dt_took = context.time_integrator["integrate"](t, tstep, dt,kwargs)

        for i in range(2):
            U[i] = FFT.ifft2(U_hat[i], U[i])
                 
        t += dt_took
        tstep += 1

        context.callbacks["update"](t,dt, tstep, context)
        
        timer()
 
        if tstep == 1 and context.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
        if t + dt >= T:
            dt = T - t
            if dt <= 1.e-14:
                break
                
    kwargs = {
            "additional_callback":context.callbacks["additional_callback"],
            "t":t,
            "dt":dt,
            "tstep": tstep,
            "T": T,
            "context":context,
            "ComputeRHS":ComputeRHS
            }
    ComputeRHS(context,U,U_hat,dU,0)
    context.callbacks["additional_callback"](fU_hat=dU,**kwargs)


    timer.final(context.MPI, FFT.rank)

    if context.make_profile:
        results = create_profile(**globals())
    
    context.callbacks["regression_test"](t,tstep,context)
        
    context.hdf5file.close()   
