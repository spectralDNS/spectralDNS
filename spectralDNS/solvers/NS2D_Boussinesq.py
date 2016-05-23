__author__ = "Mikael Mortensen <mikaem@math.uio.no> and Diako Darian <diako.darian@mn.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"


from spectralinit import *
from spectralDNS.maths.cross import *
from ..optimization import optimizer
import numpy as np
import spectralDNS.maths.integrators


def initializeContext(context,args):
    Ur = context.mesh_vars["Ur"]
    U = Ur[:2]
    rho = Ur[2]
    P = context.mesh_vars["P"]
    FFT = context.FFT
    assert context.decomposition == 'line'

    context.hdf5file = HDF5Writer(context, {"U":U[0], "V":U[1], "rho":rho, "P":P}, context.solver_name+".h5")

    integrate = spectralDNS.maths.integrators.getintegrator(context,ComputeRHS,f=None,g=None,ginv=None,hphi=None,gexp=None)

    context.time_integrator["integrate"] = integrate

    context.model_params["Pr"] = args.Pr
    context.model_params["Ri"] = args.Ri


#@optimizer
def add_pressure_diffusion(context,dUr,Ur_hat):
    K = context.mesh_vars["K"]
    K2 = context.mesh_vars["K2"]
    K_over_K2 = context.mesh_vars["K_over_K2"]
    P_hat = context.mesh_vars["P_hat"]
    nu = context.model_params["nu"]
    Ri = context.model_params["Ri"]
    Pr = context.model_params["Pr"]

    rho_hat = Ur_hat[2]

    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat  = np.sum(dUr[:2]*K_over_K2, 0, out=P_hat)
    
    P_hat += Ri*rho_hat*K_over_K2[1]
    
    # Add pressure gradient
    dUr[:2] -= P_hat*K

    # Add contribution from diffusion                      
    dUr[0] -= nu*K2*Ur_hat[0]
    
    dUr[1] -= (nu*K2*Ur_hat[1] - Ri*rho_hat)
    dUr[2] -= nu * K2 * rho_hat/Pr  
    return dUr

def ComputeRHS(context,Ur,Ur_hat,dUr, rk):
    float = context.types["float"]
    FFT = context.FFT
    curl = context.mesh_vars["curl"]
    K = context.mesh_vars["K"]
    U_hat = Ur_hat[:-1]

    Ur_dealiased = context.work[((3,)+FFT.real_shape(), float, 0)]
    F_tmp = context.work[(dUr, 0)]
    
    for i in range(3):
        Ur_dealiased[i] = FFT.ifft2(Ur_hat[i], Ur_dealiased[i], context.dealias_name)
        
    U_dealiased = Ur_dealiased[:2]
    rho_dealiased = Ur_dealiased[2]

    F_tmp[0] = cross2(F_tmp[0], K, U_hat)
    curl[:] = FFT.ifft2(F_tmp[0], curl, context.dealias_name)
    dUr[0] = FFT.fft2(U_dealiased[1]*curl, dUr[0], context.dealias_name)
    dUr[1] = FFT.fft2(-U_dealiased[0]*curl, dUr[1], context.dealias_name)
   
    F_tmp[0] = FFT.fft2(U_dealiased[0]*rho_dealiased, F_tmp[0], context.dealias_name)
    F_tmp[1] = FFT.fft2(U_dealiased[1]*rho_dealiased, F_tmp[1], context.dealias_name)
    dUr[2] = -1j*(K[0]*F_tmp[0]+K[1]*F_tmp[1])
    
    #U_tmp[0] = FFT.ifft2(1j*K[0]*rho_hat, U_tmp[0])
    #U_tmp[1] = FFT.ifft2(1j*K[1]*rho_hat, U_tmp[1])          
    #F_tmp[0] = FFT.fft2(U[0]*U_tmp[0], F_tmp[0])      
    #F_tmp[1] = FFT.fft2(U[1]*U_tmp[1], F_tmp[1])    
    #dU[2] = -1.0*(F_tmp[0] + F_tmp[1])    
    
    dUr = add_pressure_diffusion(context,dUr,Ur_hat)
    
    return dUr

def solve(context):
    Ur_hat = context.mesh_vars["Ur_hat"]
    Ur = context.mesh_vars["Ur"]
    dUr = context.mesh_vars["dUr"]
    dt = context.time_integrator["dt"]

    timer = Timer(silent=context.silent)
    tstep = 0
    T = context.model_params["T"]
    t = context.model_params["t"]
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
        Ur_hat[:],dt,dt_took = context.time_integrator["integrate"](t, tstep, dt,kwargs)

        for i in range(3):
            Ur[i] = FFT.ifft2(Ur_hat[i], Ur[i])
                 
        t += dt_took
        tstep += 1

        context.callbacks["update"](t,dt, tstep, context)
        
        timer()
        
        if tstep == 1 and context.make_profile:
            #Enable profiling after first step is finished
            context.profiler.enable()
        #Make sure that the last step hits T exactly.
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
    ComputeRHS(context,Ur,Ur_hat,dUr,0)
    context.callbacks["additional_callback"](fU_hat=dUr,**kwargs)

    timer.final(context.MPI, FFT.rank)
    
    ##TODO:Make sure the lines below work
    if context.make_profile:
        results = create_profile(**globals())
        
    context.callbacks["regression_test"](t,tstep,context)
        
    context.hdf5file.close()
