__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *

from ..optimization import optimizer
import numpy as np
import spectralDNS.maths.integrators


def initializeContext(context,args):
    context.NS = {}
    U = context.mesh_vars["U"]
    P = context.mesh_vars["P"]
    FFT = context.FFT

    context.hdf5file = HDF5Writer(context, {"U":U[0], "V":U[1], "W":U[2], "P":P}, context.solver_name+".h5")
    # Set up function to perform temporal integration (using config.integrator parameter)
    integrate = spectralDNS.maths.integrators.getintegrator(context,ComputeRHS,f=nonlinearTerm,g=linearTerm,ginv=inverseLinearTerm,hphi=hphi,gexp=expLinearTerm)

    context.time_integrator["integrate"] = integrate

    context.NS["convection"] = args.convection
    context.mesh_vars["work_shape"] = FFT.real_shape_padded() if context.dealias_name == '3/2-rule' else FFT.real_shape()
    context.NS["conv"] = getConvection(context)

    # Shape of work arrays used in convection with dealiasing. Different shape whether or not padding is involved

def standardConvection(context,c,U_hat,dealias=None):
    """c_i = u_j du_i/dx_j"""
    FFT=context.FFT
    dealias = context.mesh_vars["dealias"]
    K = context.mesh_vars["K"]
    U_tmp = context.mesh_vars["U_tmp"]

    Uc = FFT.get_workarray((3,)+context.mesh_vars["work_shape"], 2)

    for i in range(3):
        for j in range(3):
            Uc[j] = FFT.ifftn(1j*K[j]*U_hat[i], Uc[j], dealias)
        c[i] = FFT.fftn(sum(U_dealiased*Uc, 0), c[i], dealias)
    return c

def divergenceConvection(context,c, U_dealiased, dealias=None,add=False):
    """c_i = div(u_i u_j)"""
    FFT = context.FFT
    F_tmp = context.mesh_vars["F_tmp"]
    K = context.mesh_vars["K"]

    if not add: c.fill(0)

    for i in range(3):
        F_tmp[i] = FFT.fftn(U_dealiased[0]*U_dealiased[i], F_tmp[i], dealias)
    c[0] += 1j*sum(K*F_tmp, 0)

    c[1] += 1j*K[0]*F_tmp[1]
    c[2] += 1j*K[0]*F_tmp[2]
    F_tmp[0] = FFT.fftn(U_dealiased[1]*U_dealiased[1], F_tmp[0], dealias)
    F_tmp[1] = FFT.fftn(U_dealiased[1]*U_dealiased[2], F_tmp[1], dealias)
    F_tmp[2] = FFT.fftn(U_dealiased[2]*U_dealiased[2], F_tmp[2], dealias)
    c[1] += (1j*K[1]*F_tmp[0] + 1j*K[2]*F_tmp[1])
    c[2] += (1j*K[1]*F_tmp[1] + 1j*K[2]*F_tmp[2])
    return c

#@profile
def Cross(context,a, b, c,dealias=None):
    """c_k = F_k(a x b)"""
    U_tmp = context.mesh_vars["U_tmp"]
    FFT = context.FFT
    Uc = FFT.get_workarray(a, 2)
    Uc[:] = cross1(Uc, a, b)
    c[0] = FFT.fftn(Uc[0], c[0], dealias)
    c[1] = FFT.fftn(Uc[1], c[1], dealias)
    c[2] = FFT.fftn(Uc[2], c[2], dealias)
    return c

#@profile
def Curl(context,a, c, dealias=None):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    F_tmp = context.mesh_vars["F_tmp"]
    K = context.mesh_vars["K"]
    FFT = context.FFT

    F_tmp[:] = cross2(F_tmp, K, a)
    c[0] = FFT.ifftn(F_tmp[0], c[0], dealias)
    c[1] = FFT.ifftn(F_tmp[1], c[1], dealias)
    c[2] = FFT.ifftn(F_tmp[2], c[2], dealias)    
    return c

def getConvection(context):

    """Return function used to compute convection"""
    convection = context.NS["convection"]
    FFT = context.FFT
    work_shape = context.mesh_vars["work_shape"]

    float = context.types["float"]
    complex = context.types["complex"]

    if convection == "Standard":
        
        def Conv(dU,U_hat):
            U_dealiased = FFT.get_workarray(((3,)+work_shape, float), 0)
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], context.dealias_name)
            dU = standardConvection(context,dU, U_hat, context.dealias_name)
            dU[:] *= -1 
            return dU
        
    elif convection == "Divergence":
        def Conv(dU,U_hat):
            U_dealiased = FFT.get_workarray(((3,)+work_shape, float), 0)
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], context.dealias_name)
            dU = divergenceConvection(context,dU, U_dealiased,U_hat, context.dealias_name, False)
            dU[:] *= -1
            return dU
        
    elif convection == "Skewed":
        
        def Conv(dU,U_hat):
            U_dealiased = FFT.get_workarray(((3,)+work_shape, float), 0)
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], context.dealias_name)
            dU = standardConvection(context,dU, U_hat, context.dealias_name)
            dU = divergenceConvection(context,dU, U_dealiased,U_hat,context.dealias_name, True)
            dU *= -0.5
            return dU
        
    elif convection == "Vortex":
        
        def Conv(dU,U_hat):
            U_dealiased = FFT.get_workarray(((3,)+work_shape, float), 0)
            curl_dealiased = FFT.get_workarray(((3,)+context.mesh_vars["work_shape"], context.types["float"]), 1)
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], context.dealias_name)
            
            curl_dealiased[:] = Curl(context,U_hat, curl_dealiased, context.dealias_name)
            dU = Cross(context,U_dealiased, curl_dealiased, dU, context.dealias_name)
            return dU
    else:
        raise AssertionError("Invalid convection specified")

        
    return Conv           


@optimizer
def add_pressure(context,dU, U_hat):
    """Add contributions from pressure and diffusion to the rhs"""
    K = context.mesh_vars["K"]
    K_over_K2 = context.mesh_vars["K_over_K2"]
    P_hat = context.mesh_vars["P_hat"]
    
    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat = np.sum(dU*K_over_K2, 0, out=P_hat)
        
    # Subtract pressure gradient
    dU -= P_hat*K
    
    return dU

#@profile
def ComputeRHS(context,U,U_hat,dU, rk):
    """Compute and return entire rhs contribution"""
    conv = context.NS["conv"]
    FFT = context.FFT
    K2 = context.mesh_vars["K2"]
    nu = context.model_params["nu"]

    if rk > 0: # For rk=0 the correct values are already in U
        for i in range(3):
            U[i] = FFT.ifftn(U_hat[i], U[i])
                        
    dU = conv(dU,U_hat)

    dU = add_pressure(context,dU, U_hat)
    # Subtract contribution from diffusion
    dU -= nu*K2*U_hat
        
    return dU


def nonlinearTerm(context,U,U_hat,dU,rk):
    conv = context.NS["conv"]
    FFT = context.FFT
    if rk > 0: # For rk=0 the correct values are already in U
        for i in range(3):
            U[i] = FFT.ifftn(U_hat[i], U[i])
    dU = conv(dU,U_hat)
    dU = add_pressure(context,dU, U_hat)
    return dU

 
def linearTerm(context,U,U_hat,dU,rk):
    K2 = context.mesh_vars["K2"]
    nu = context.model_params["nu"]
    dU[:] = -nu*K2*U_hat
    return dU

def inverseLinearTerm(context,U,U_hat,dU,rk,factor):
    #We want to calculate (I-factorg)^-1
    nu = context.model_params["nu"]
    K2 = context.mesh_vars["K2"]
    #TODO Make this more efficient
    dU[:] = (1./(1 + nu*factor*K2[:]))*U_hat[:]
    return dU

#TODO: Change the function signature here.
def expLinearTerm(context,U,U_hat,dU,rk,dt):
    nu = context.model_params["nu"]
    K2 = context.mesh_vars["K2"]
    #TODO:Make this more efficient
    U_hat[:] = np.exp(-nu*dt*K2[:])*U_hat[:]

#TODO: Change the function signature here.
def hphi(context,k,U,U_hat,rk,dt):
    K2 = context.mesh_vars["K2"]
    nu = context.model_params["nu"]
    FFT = context.FFT
    if k == 1:
        if FFT.rank != 0:
            U_hat[:] = 1/(-nu*K2[:])*(np.exp(-nu*dt*K2[:]) - 1)*U_hat[:]
        else:
            k0 = U_hat[:,0,0,0]
            K2[0,0,0] = -nu
            U_hat[:] = 1/(-nu*K2[:])*(np.exp(-nu*dt*K2[:]) - 1)*U_hat[:]
            K2[0,0,0] = 0
            U_hat[:,0,0,0] = dt*k0


def solve(context):
    U_hat = context.mesh_vars["U_hat"]
    U = context.mesh_vars["U"]
    dU = context.mesh_vars["dU"]
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
        U_hat[:],dt,dt_took = context.time_integrator["integrate"](t, tstep, dt,kwargs)

        for i in range(3):
            U[i] = FFT.ifftn(U_hat[i], U[i])
                 
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
    ComputeRHS(context,U,U_hat,dU,0)
    context.callbacks["additional_callback"](fU_hat=dU,**kwargs)

    timer.final(context.MPI, FFT.rank)
    
    ##TODO:Make sure the lines below work
    if context.make_profile:
        results = create_profile(**globals())
        
    context.callbacks["regression_test"](t,tstep,context)
        
    context.hdf5file.close()
