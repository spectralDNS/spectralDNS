__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-07"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..optimization import optimizer, wraps
from numpy import array
import spectralDNS.context

__all__ = ['getintegrator']

def adaptiveRK(context,A,b,bhat,err_order, dU,fY_hat,fY, fsal, aTOL,rTOL,f,dt,tstep,kw):
    U = context.mesh_vars["U"]
    U_hat = context.mesh_vars["U_hat"]
    FFT = context.FFT
    fftn = FFT.fftn
    ifftn = FFT.ifftn


    s = A.shape[0]
    
    #Some parameters for adaptive time-stepping. See p167, Hairer, Norsett and Wanner. "Solving Ordinary Differential Equations 1"
    #for details.
    facmax_default = 2
    facmax = facmax_default
    fac = 0.8
    facmin = 0.01

    #This is used for the FSAL property to save computational cost.
    offset = 0

    #We may need to repeat the time-step until a small enough value is used.
    while True:
        if fsal:
            offset = (offset - 1) % s
        dt_prev = dt
        for i in range(0,s):
            if not fsal or (tstep == 0 or i != 0 ):
                fY_hat[(i + offset) % s] =  U_hat
                for j in range(0,i):
                    fY_hat[(i+offset) % s] += dt*A[i,j]*fY_hat[(j+offset) % s]
                #Compute iverse Fourier Transform of Y
                if i != 0:
                    for k in range(dim): fY[(i + offset) % s][k] = ifftn(fY_hat[(i + offset) % s][k],fY[(i+offset)%s][k])
                else:
                    fY[(i+offset)%s] = U #As this is an explicit method
                #Cpmpute F(Y)
                dU = f(dU,fY[(i+offset)%s],fY_hat[(i+offset)%s],configuration)
                fY_hat[(i+offset)%s] = dU
            if i == 0:
                kw["additional_callback"](dU=dU,**kw)
 
        #Calculate the new value
        U_hat_new[:] = U_hat
        U_hat_new[:] += dt*b[0]*fY_hat[(0+offset)%s]
        err[:] = dt*(b[0] - bhat[0])*fY_hat[(0+offset)%s]

        for j in range(1,s):
            U_hat_new[:] += dt*b[j]*fY_hat[(j+offset)%s]
            err[:] += dt*(b[j] - bhat[j])*fY_hat[(j+offset)%s]

        est = 0.0
        sc[:] = aTOL + np.maximum(np.abs(U_hat),np.abs(U_hat_new))*rTOL
        if errnorm == "2":
            est_to_bcast = None
            nsquared = np.zeros(dim,dtype=U.dtype)
            for k in range(dim):
                nsquared[k] = FFT.comm.reduce(np.sum(np.power(np.abs(err[k]/sc[k]),2)))
            if FFT.comm.rank == 0:
                est_to_bcast = np.zeros(1,dtype=U.dtype)
                est = np.max(np.sqrt(nsquared))
                est /= np.sqrt(N[0]*N[1]*(N[2]/2 + 1))
                est_to_bcast[0] = est
            est_to_bcast = FFT.comm.bcast(est_to_bcast,root=0)
            est = est_to_bcast[0]
        elif errnorm == "inf":
            #TODO: Test this error norm
            sc[:] = aTOL + np.maximum(np.abs(U_hat),np.abs(U_hat_new))*rTOL
            err[:] = err[:]/sc[:]
            err = np.abs(err,out=err)
            asdf = np.max(err)
            #TODO:add dtype below
            x = np.zeros(asdf.shape)
            FFT.comm.Allreduce(asdf,x,op=MPI.MAX)
            est = np.abs(np.max(x))
            est /= np.sqrt(N[0]*N[1]*(N[2]/2 + 1))
        else:
           assert False,"Wrong error norm"

        #Check error estimate
        factor = min(facmax,max(facmin,fac*pow((1/est),1.0/(err_order+1))))
        if adaptive:
            dt = dt*factor
            if  est > 1.0:
                facmax = 1
                if not timestep_rejected_callback is None:
                    timestep_rejected_callback(tstep,t,dt_prev)
                #The offset gets decreased in the  next step, which is something we do not want.
                if fsal:
                    offset += 1
                continue

    #Update U_hat and U
    U_hat[:] = U_hat_new
    for k in range(dim):
        U[k] = ifftn(U_hat[k],U[k])

    #If we successfully made a timestep, make it possible to increase the timestep again
        facmax = facmax_default

   


@optimizer
def RK4(context,u0, u1, u2, dU, a, b, dt, ComputeRHS,kw):
    """Runge Kutta fourth order"""
    U = context.mesh_vars["U"]
    u2[:] = u1[:] = u0
    for rk in range(4):
        dU = ComputeRHS(context,U,u0,dU, rk)
        if rk == 0 and "additional_callback" in kw:
            kw["additional_callback"](dU=dU,**kw)
        if rk < 3:
            u0[:] = u1 + b[rk]*dt*dU
        u2 += a[rk]*dt*dU
    u0[:] = u2
    return u0,dt

@optimizer
def ForwardEuler(context,u0, u1, dU, dt, ComputeRHS,kw):
    U = context.mesh_vars["U"]
    dU = ComputeRHS(context,U,u0,dU, 0)        
    u0 += dU*dt
    return u0,dt

@optimizer
def AB2(context,u0, u1, dU, dt, tstep, ComputeRHS,kw):
    #TODO: Implement this.
    raise AssertionError("Not yet implemented") # Need to use u0 and u1 for ComputeRHS
    dU = ComputeRHS(context,U,dU, 0)
    if tstep == 1:
        u0 += dU*dt
    else:
        u0 += (1.5*dU*dt - 0.5*u1)        
    u1[:] = dU*dt    
    return u0

def getintegrator(context,ComputeRHS):
    dU = context.mesh_vars["dU"]
    float = context.types["float"]
    """Return integrator using choice in global parameter integrator.
    """
    if context.solver_name in ("NS", "VV", "NS2D"):
        u0 = context.mesh_vars['U_hat']
    elif context.solver_name == "MHD":
        u0 = context.mesh_vars['UB_hat']
    elif context.solver_name == "Bq2D":
        u0 = context.mesh_vars['Ur_hat']
    u1 = u0.copy()    
        
    if context.time_integrator["time_integrator_name"] == "RK4": 
        # RK4 parameters
        a = array([1./6., 1./3., 1./3., 1./6.], dtype=float)
        b = array([0.5, 0.5, 1.], dtype=float)
        u2 = u0.copy()
        @wraps(RK4)
        def func(t, tstep, dt,additional_args = {}):
            return RK4(context,u0, u1, u2, dU, a, b, dt, ComputeRHS,additional_args)
        return func
            
    elif context.time_integrator["time_integrator_name"] == "ForwardEuler":  
        @wraps(ForwardEuler)
        def func(t, tstep, dt,additional_args = {}):
            return ForwardEuler(context,u0, u1, dU, dt, ComputeRHS,additional_args)
        return func
    
    else:
        @wraps(AB2)
        def func(t, tstep, dt,additional_args = {}):
            return AB2(context,u0, u1, dU, dt, tstep, ComputeRHS,additional_args)
        return func
