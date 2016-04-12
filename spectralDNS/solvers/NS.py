__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

#TODO:Import spectralinit instead
from ..optimization import optimizer
from spectralDNS.h5io import *
import spectralDNS.maths.integrators
from spectralDNS.utilities import *
from spectralDNS.maths import *
import numpy as np


def initializeContext(context,args):
    context.NS = {}
    U = context.mesh_vars["U"]
    P = context.mesh_vars["P"]

    context.hdf5file = HDF5Writer(context, {"U":U[0], "V":U[1], "W":U[2], "P":P}, context.solver_name+".h5")
    # Set up function to perform temporal integration (using config.integrator parameter)
    #TODO:Consider passing ComputeRHS within context below
    integrate = spectralDNS.maths.integrators.getintegrator(context,ComputeRHS)
    context.time_integrator["integrate"] = integrate

    context.NS["convection"] = args.convection
    context.NS["conv"] = getConvection(context)

def standardConvection(context,c, U_dealiased):
    """c_i = u_j du_i/dx_j"""
    for i in range(3):
        for j in range(3):
            U_tmp[j] = FFT.ifftn(1j*K[j]*U_hat[i]*context.mesh_vars["dealias"], U_tmp[j])
        c[i] = FFT.fftn(np.sum(U_dealiased*U_tmp, 0), c[i])
    return c

def divergenceConvection(context,c, U_dealiased, add=False):
    FFT = context.FFT
    F_tmp = context.mesh_vars["F_tmp"]
    K = context.mesh_vars["K"]

    """c_i = div(u_i u_j)"""
    if not add: c.fill(0)
    for i in range(3):
        F_tmp[i] = FFT.fftn(U_dealiased[0]*U_dealiased[i], F_tmp[i])
    c[0] += 1j*np.sum(K*F_tmp, 0)
    c[1] += 1j*K[0]*F_tmp[1]
    c[2] += 1j*K[0]*F_tmp[2]
    F_tmp[0] = FFT.fftn(U_dealiased[1]*U_dealiased[1], F_tmp[0])
    F_tmp[1] = FFT.fftn(U_dealiased[1]*U_dealiased[2], F_tmp[1])
    F_tmp[2] = FFT.fftn(U_dealiased[2]*U_dealiased[2], F_tmp[2])
    c[1] += (1j*K[1]*F_tmp[0] + 1j*K[2]*F_tmp[1])
    c[2] += (1j*K[1]*F_tmp[1] + 1j*K[2]*F_tmp[2])
    return c

#@profile
def Cross(context,a, b, c):
    U_tmp = context.mesh_vars["U_tmp"]
    FFT = context.FFT
    """c_k = F_k(a x b)"""
    U_tmp[:] = cross1(U_tmp, a, b)
    c[0] = FFT.fftn(U_tmp[0], c[0])
    c[1] = FFT.fftn(U_tmp[1], c[1])
    c[2] = FFT.fftn(U_tmp[2], c[2])
    return c

#@profile
def Curl(context,a, c, dealiasing=True):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    F_tmp = context.mesh_vars["F_tmp"]
    K = context.mesh_vars["K"]
    FFT = context.FFT

    dealias = context.mesh_vars["dealias"]
    F_tmp[:] = cross2(F_tmp, K, a)
    if dealiasing:
        F_tmp[:] = dealias_rhs(F_tmp, dealias)
    c[0] = FFT.ifftn(F_tmp[0], c[0])
    c[1] = FFT.ifftn(F_tmp[1], c[1])
    c[2] = FFT.ifftn(F_tmp[2], c[2])    
    return c

def getConvection(context):
    """Return function used to compute convection"""
    convection = context.NS["convection"]
    U_dealiased = context.mesh_vars["U_dealiased"]
    dealias = context.mesh_vars["dealias"]
    dU = context.mesh_vars["dU"]
    U_hat = context.mesh_vars["U_hat"]
    curl = context.mesh_vars["curl"]
    FFT = context.FFT

    if convection == "Standard":
        
        def Conv(dU):
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i]*dealias, U_dealiased[i])
            dU = standardConvection(context,dU, U_dealiased)
            dU[:] *= -1 
            return dU
        
    elif convection == "Divergence":
        
        def Conv(dU):
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i]*dealias, U_dealiased[i])
            dU = divergenceConvection(context,dU, U_dealiased, False)
            dU[:] *= -1
            return dU
        
    elif convection == "Skewed":
        
        def Conv(dU):
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i]*dealias, U_dealiased[i])
            dU = standardConvection(context,dU, U_dealiased)
            dU = divergenceConvection(context,dU, U_dealiased, True)
            dU *= -0.5
            return dU
        
    elif convection == "Vortex":
        
        def Conv(dU):
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i]*dealias, U_dealiased[i])
            curl[:] = Curl(context,U_hat, curl)
            dU = Cross(context,U_dealiased, curl, dU)
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
def ComputeRHS(context,U,U_hat,dU, rk):
    #TODO: Reset this...
    FFT = context.FFT
    K_over_K2 = context.mesh_vars["K_over_K2"]
    K = context.mesh_vars["K"]
    K2 = context.mesh_vars["K2"]
    P_hat = context.mesh_vars["P_hat"]
    curl = context.mesh_vars["curl"]
    nu = context.model_params["nu"]
    dealias = FFT.get_dealias_filter()


    if rk > 0:
        for i in range(3):
            U[i] = FFT.ifftn(U_hat[i], U[i])
    def Curl(a, c):
        K = context.mesh_vars["K"]
        c[2] = FFT.ifftn(1j*(K[0]*a[1]-K[1]*a[0]), c[2])
        c[1] = FFT.ifftn(1j*(K[2]*a[0]-K[0]*a[2]), c[1])
        c[0] = FFT.ifftn(1j*(K[1]*a[2]-K[2]*a[1]), c[0])
        return c



    curl[:] = Curl(U_hat, curl)
    dU = Cross(context,U, curl, dU)
    dU *= dealias
    P_hat[:] = np.sum(dU*K_over_K2, 0, out=P_hat)
    dU -= P_hat*K
    dU -= nu*K2*U_hat
 
    return dU





    """Compute and return entire rhs contribution"""
    conv = context.NS["conv"]
    K2 = context.mesh_vars["K2"]
    K = context.mesh_vars["K"]
    P_hat = context.mesh_vars["P_hat"]
    K_over_K2 = context.mesh_vars["K_over_K2"]
    nu = context.model_params["nu"]
    FFT = context.FFT


    
    if rk > 0: # For rk=0 the correct values are already in U
        for i in range(3):
            U[i] = FFT.ifftn(U_hat[i], U[i])
                        
    dU = conv(dU)

    dU = add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu)
        
    return dU

def regression_test(context,t, tstep, **kw):
    pass


def solve(context):
    U_hat = context.mesh_vars["U_hat"]
    U = context.mesh_vars["U"]
    dU = context.mesh_vars["dU"]
    dt = context.time_integrator["dt"]

    timer = Timer()
    t = 0.0
    tstep = 0
    T = context.model_params["T"]
    FFT = context.FFT

    while t + dt <= T + 1.e-15: #The 1.e-15 term is for rounding errors
        
        kwargs = {
                "additional_callback":context.callbacks["additional_callback"],
                "t":t,
                "dt":dt,
                "tstep": tstep,
                "T": T,
                "context":context
                }
        U_hat[:] = context.time_integrator["integrate"](t, tstep, dt,kwargs)

        for i in range(3):
            U[i] = FFT.ifftn(U_hat[i], U[i])
                 
        t += dt
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
            "T": T,#TODO: Fix the line below
            "global_vars":globals()
            }
    ComputeRHS(context,U,U_hat,dU,0)
    context.callbacks["additional_callback"](context,dU=dU,**kwargs)

    timer.final(context.MPI, FFT.rank)
    
    ##TODO:Make sure the lineis below work
    if context.make_profile:
        results = create_profile(**globals())
        
    regression_test(t, tstep, context)
        
    context.hdf5file.close()
