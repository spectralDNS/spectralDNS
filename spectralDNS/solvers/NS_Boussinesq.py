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
    rho = context.mesh_vars["rho"]
    FFT = context.FFT

    context.hdf5file = HDF5Writer(context, {"U":U[0], "V":U[1], "W":U[2], "rho":rho, "P":P}, context.solver_name+".h5")
    # Set up function to perform temporal integration (using config.integrator parameter)
    integrate = spectralDNS.maths.integrators.getintegrator(context,ComputeRHS,f=nonlinearTerm,g=linearTerm,ginv=inverseLinearTerm,hphi=hphi,gexp=expLinearTerm)

    context.time_integrator["integrate"] = integrate

    context.NS["convection"] = args.convection
    # Shape of work arrays used in convection with dealiasing. Different shape whether or not padding is involved
    context.mesh_vars["work_shape"] = FFT.real_shape_padded() if context.dealias_name == '3/2-rule' else FFT.real_shape()
    context.NS["conv"] = getConvection(context)
    
    #TODO: Make this user_defined
    context.model_params["Pr"] = args.Pr
    context.model_params["Ri"] = args.Ri

def standardConvection(context,c,U_hat,U_dealiased,dealias=None):
    """c_i = u_j du_i/dx_j"""
    FFT=context.FFT
    K = context.mesh_vars["K"]

    gradUi = context.work[(U_dealiased, 2)]
    for i in range(3):
        for j in range(3):
            gradUi[j] = FFT.ifftn(1j*K[j]*U_hat[i], gradUi[j], dealias)
        c[i] = FFT.fftn(np.sum(U_dealiased*gradUi, 0), c[i], dealias)
    return c

def divergenceConvection(context,c, U_dealiased, dealias=None,add=False):
    """c_i = div(u_i u_j)"""
    FFT = context.FFT
    F_tmp = context.mesh_vars["F_tmp"]
    K = context.mesh_vars["K"]

    if not add: c.fill(0)

    UUi_hat = context.work[(c, 0)]
    for i in range(3):
        UUi_hat[i] = FFT.fftn(U_dealiased[0]*U_dealiased[i], UUi_hat[i], dealias)
    c[0] += 1j*sum(K*UUi_hat, 0)
    c[1] += 1j*K[0]*UUi_hat[1]
    c[2] += 1j*K[0]*UUi_hat[2]
    UUi_hat[0] = FFT.fftn(U_dealiased[1]*U_dealiased[1], UUi_hat[0], dealias)
    UUi_hat[1] = FFT.fftn(U_dealiased[1]*U_dealiased[2], UUi_hat[1], dealias)
    UUi_hat[2] = FFT.fftn(U_dealiased[2]*U_dealiased[2], UUi_hat[2], dealias)
    c[1] += (1j*K[1]*UUi_hat[0] + 1j*K[2]*UUi_hat[1])
    c[2] += (1j*K[1]*UUi_hat[1] + 1j*K[2]*UUi_hat[2])
    return c

#@profile
def Cross(context,a, b, c,dealias=None):
    """c_k = F_k(a x b)"""
    FFT = context.FFT
    Uc = context.work[(a, 2)]
    Uc = cross1(Uc, a, b)
    c[0] = FFT.fftn(Uc[0], c[0], dealias)
    c[1] = FFT.fftn(Uc[1], c[1], dealias)
    c[2] = FFT.fftn(Uc[2], c[2], dealias)
    return c

#@profile
def Curl(context,a, c, dealias=None):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    K = context.mesh_vars["K"]
    FFT = context.FFT

    curl_hat = context.work[(a, 0)]
    curl_hat = cross2(curl_hat, K, a)
    c[0] = FFT.ifftn(curl_hat[0], c[0], dealias)
    c[1] = FFT.ifftn(curl_hat[1], c[1], dealias)
    c[2] = FFT.ifftn(curl_hat[2], c[2], dealias)    
    return c

def getConvection(context):

    """Return function used to compute convection"""
    convection = context.NS["convection"]
    FFT = context.FFT
    work_shape = context.mesh_vars["work_shape"]

    float = context.types["float"]
    complex = context.types["complex"]

    if convection == "Standard":
        def Conv(dUr,Ur_hat):
            Ur_dealiased = context.work[((4,)+work_shape, float, 0)]
            for i in range(4):
                Ur_dealiased[i] = FFT.ifftn(Ur_hat[i], Ur_dealiased[i], context.dealias_name)
            dU = dUr[:3]
            U_hat = Ur_hat[:3]
            U_dealiased = Ur_dealiased[:3]
            standardConvection(context,dU, U_hat, U_dealiased, context.dealias_name)
            dU[:] *= -1 
            rho_convection(context,Ur_dealiased,dUr)
        
    elif convection == "Divergence":
        def Conv(dUr,Ur_hat):
            Ur_dealiased = context.work[((4,)+work_shape, float, 0)]
            for i in range(4):
                Ur_dealiased[i] = FFT.ifftn(Ur_hat[i], Ur_dealiased[i], context.dealias_name)
            dU = dUr[:3]
            U_hat = Ur_hat[:3]
            U_dealiased = Ur_dealiased[:3]
            divergenceConvection(context,dU, U_dealiased, context.dealias_name, False)
            dU[:] *= -1
            rho_convection(context,Ur_dealiased,dUr)
        
    elif convection == "Skewed":
        def Conv(dUr,Ur_hat):
            Ur_dealiased = context.work[((4,)+work_shape, float, 0)]
            for i in range(4):
                Ur_dealiased[i] = FFT.ifftn(Ur_hat[i], Ur_dealiased[i], context.dealias_name)
            dU = dUr[:3]
            U_hat = Ur_hat[:3]
            U_dealiased = Ur_dealiased[:3]
            standardConvection(context,dU,U_hat, U_dealiased, context.dealias_name)
            divergenceConvection(context,dU, U_dealiased, context.dealias_name, True)
            dU *= -0.5
            rho_convection(context,Ur_dealiased,dUr)
        
    elif convection == "Vortex":
        
        def Conv(dUr,Ur_hat):
            Ur_dealiased = context.work[((4,)+work_shape, float, 0)]
            curl_dealiased = context.work[((3,)+work_shape, float, 1)]
            for i in range(4):
                Ur_dealiased[i] = FFT.ifftn(Ur_hat[i], Ur_dealiased[i], context.dealias_name)
            dU = dUr[:3]
            U_hat = Ur_hat[:3]
            U_dealiased = Ur_dealiased[:3]
           
            curl_dealiased[:] = Curl(context,U_hat, curl_dealiased, context.dealias_name)
            Cross(context,U_dealiased, curl_dealiased, dU, context.dealias_name)
            rho_convection(context,Ur_dealiased,dUr)
    else:
        raise AssertionError("Invalid convection specified")
        
    return Conv           
def rho_convection(context,Ur_dealiased,dUr):
    FFT = context.FFT
    K = context.mesh_vars["K"]
    work_shape = context.mesh_vars["work_shape"]

    rho_dealiased = Ur_dealiased[3]
    F_tmp = context.work[((3,)+context.mesh_vars["Ur_hat"][0].shape,context.types["complex"],0)]

    FFT.fftn(Ur_dealiased[0]*rho_dealiased, F_tmp[0], context.dealias_name)
    FFT.fftn(Ur_dealiased[1]*rho_dealiased, F_tmp[1], context.dealias_name)
    FFT.fftn(Ur_dealiased[2]*rho_dealiased, F_tmp[2], context.dealias_name)
    dUr[3] = -1j*(K[0]*F_tmp[0]+K[1]*F_tmp[1] + K[2]*F_tmp[2])
 

#@optimizer
def add_pressure(context,dUr, Ur_hat):
    """Add contributions from pressure and diffusion to the rhs"""
    K = context.mesh_vars["K"]
    K_over_K2 = context.mesh_vars["K_over_K2"]
    P_hat = context.mesh_vars["P_hat"]
    Ri = context.model_params["Ri"]
    
    dU = dUr[:3] 
    rho_hat = Ur_hat[3]

    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat = np.sum(dU*K_over_K2, 0, out=P_hat)
    P_hat -= Ri*rho_hat*K_over_K2[2]#TODO:Is this correct?

    # Subtract pressure gradient
    dU -= P_hat*K
    
    return dU

#TODO:Change the function signature of this function.
#@profile
def ComputeRHS(context,Ur,Ur_hat,dUr, rk):
    """Compute and return entire rhs contribution"""
    conv = context.NS["conv"]
    FFT = context.FFT
    K2 = context.mesh_vars["K2"]
    nu = context.model_params["nu"]
    work_shape = context.mesh_vars["work_shape"]
    Ri = context.model_params["Ri"]
    Pr = context.model_params["Pr"]

    U = Ur[:3]
    dU = dUr[:3]
    U_hat = Ur_hat[:3]
    rho_hat = Ur_hat[3]
                        
    conv(dUr,Ur_hat)

    add_pressure(context,dUr, Ur_hat)
    # Subtract contribution from diffusion
    dU[0] -= nu*K2*U_hat[0]
    dU[1] -= nu*K2*U_hat[1]

    dU[2] -= (nu*K2*U_hat[2] + Ri*rho_hat)

    dUr[3] -= nu * K2 * rho_hat/Pr  
    return dUr

def nonlinearTerm(context,U,U_hat,dU,rk):
    conv = context.NS["conv"]

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

        for i in range(4):
            Ur[i] = FFT.ifftn(Ur_hat[i], Ur[i])
                 
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
