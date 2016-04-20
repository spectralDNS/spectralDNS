__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-07"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..optimization import optimizer, wraps
from numpy import array
import spectralDNS.context
import nodepy
import numpy as np

__all__ = ['getintegrator']

def imexDIRK(context,A,b,A_hat,b_hat,U_tmp,K,K_hat,dU,f,g,ginv,dt,tstep,kw):
    s = A.shape[0] - 1
    U = context.mesh_vars["U"]
    U_hat = context.mesh_vars["U_hat"]

    K_hat[0] = f(context,U,U_hat,dU,0)

    for i in range(s):
        K[i] = U_hat
        for j in range(i):
            K[i] += dt*A[i,j]*K[j][:]
            K[i] += dt*A_hat[i+1,j]*K_hat[j][:]
        K[i] += dt*A_hat[i+1,i]*K_hat[i]
        dU = g(context,U_tmp,K[i],dU,i)
        K[i] = ginv(context,U,dU,K[i],i,A[i,i]*dt)

        #TODO: Does K_hat[i] model anything?
        if i == 0 and "additional_callback" in kw:
            kw["additional_callback"](fU_hat=K_hat[i],**kw)

        K_hat[i+1] = f(context,U_tmp,K[i],dU,i+1)
    for i in range(s):
       U_hat[:] += dt*b[i]*K[i]
       U_hat[:] += dt*b_hat[i]*K_hat[i]
    U_hat[:] += dt*b_hat[i]*K_hat[s]
    return U_hat,dt,dt

def getIMEXOneStep(context,dU,f,g,ginv):
    U = context.mesh_vars["U"]
    U_hat = context.mesh_vars["U_hat"]

    A = np.array([[0,0],[0,1]],dtype=np.float64)
    b = np.array([0,1],dtype=np.float64)
    A_hat = np.array([[0,0],[1,0]],dtype=np.float64)
    b_hat = np.array([1,0],dtype=np.float64)

    s = A.shape[0] - 1
    K = np.empty((s,) + U_hat.shape, dtype=U_hat.dtype)
    K_hat = np.empty((s+1,)+U_hat.shape,dtype=U_hat.dtype)
    U_tmp = np.empty(U.shape,dtype=U.dtype)
#TODO: Do we need to use @wraps here?
    def IMEXOneStep(t,tstep,dt,additional_args = {}):
        return imexDIRK(context,A,b,A_hat,b_hat,U_tmp,K,K_hat,dU,f,g,ginv,dt,tstep,additional_args)
    return IMEXOneStep


@optimizer
def adaptiveRK(context,A,b,bhat,err_order, fY_hat,U_tmp,U_hat_new,sc,err, fsal,offset, aTOL,rTOL,adaptive,errnorm,dU,ComputeRHS,dt,tstep,kw):
    U = context.mesh_vars["U"]
    U_hat = context.mesh_vars["U_hat"]
    N = context.model_params["N"]

    FFT = context.FFT
    fftn = FFT.fftn
    ifftn = FFT.ifftn
    #TODO: Set this dynamically.
    dim = 3


    s = A.shape[0]
    
    #Some parameters for adaptive time-stepping. See p167, Hairer, Norsett and Wanner. "Solving Ordinary Differential Equations 1"
    #for details.
    facmax_default = 2
    facmax = facmax_default
    fac = 0.8
    facmin = 0.01

    #We may need to repeat the time-step until a small enough value is used.
    while True:
        dt_prev = dt
        if fsal:
            offset[0] = (offset[0] - 1) % s
        for i in range(0,s):
            if not fsal or (tstep == 0 or i != 0 ):
                fY_hat[(i + offset[0]) % s] =  U_hat
                for j in range(0,i):
                    fY_hat[(i+offset[0]) % s] += dt*A[i,j]*fY_hat[(j+offset[0]) % s]
                #ComputeRHS does not calculate ifft if i = 0
                if i==0:
                    U_tmp[:] = U 
                #Compute F(Y)
                dU = ComputeRHS(context,U_tmp,fY_hat[(i+offset[0])%s],dU,i)
                fY_hat[(i+offset[0])%s] = dU
            if i == 0 and "additional_callback" in kw:
                kw["additional_callback"](fU_hat=fY_hat[(0+offset[0]) % s],**kw)
 
        #Calculate the new value
        U_hat_new[:] = U_hat
        U_hat_new[:] += dt*b[0]*fY_hat[(0+offset[0])%s]
        err[:] = dt*(b[0] - bhat[0])*fY_hat[(0+offset[0])%s]

        for j in range(1,s):
            U_hat_new[:] += dt*b[j]*fY_hat[(j+offset[0])%s]
            err[:] += dt*(b[j] - bhat[j])*fY_hat[(j+offset[0])%s]

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
                #TODO: Make sure this works for other dimensions too.
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
            x = np.zeros(asdf.shape,U.dtype)
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
                kw["additional_callback"](is_step_rejected_callback=True,dt_rejected=dt_prev,**kw)
                #The offset gets decreased in the  next step, which is something we do not want.
                if fsal:
                    offset[0] += 1
                continue
        break

    #Update U_hat and U
    U_hat[:] = U_hat_new
    return U_hat,dt,dt_prev
   
@optimizer
def getBS5(context,dU,ComputeRHS,aTOL,rTOL,adaptive=True):
    U = context.mesh_vars["U"]
    U_hat = context.mesh_vars["U_hat"]

    A = nodepy.rk.loadRKM("BS5").A.astype(np.float64)
    b = nodepy.rk.loadRKM("BS5").b.astype(np.float64)
    bhat = nodepy.rk.loadRKM("BS5").bhat.astype(np.float64)
    err_order = 4
    errnorm = "2"
    fsal = True

    #Offset for fsal stuff. #TODO: infer this from tstep
    offset = [0]

    s = A.shape[0]
    U_tmp = np.zeros(U.shape, dtype=U.dtype)
    fY_hat = np.zeros((s,) + U_hat.shape, dtype = U_hat.dtype)
    sc = np.zeros(U_hat.shape,dtype=U_hat.dtype)
    err = np.zeros(U_hat.shape,dtype=U_hat.dtype)
    U_hat_new = np.zeros(U_hat.shape,dtype=U_hat.dtype)

    @wraps(adaptiveRK)
    def BS5(t,tstep,dt,additional_args = {}):
        return adaptiveRK(context,A,b,bhat,err_order, fY_hat,U_tmp,U_hat_new,sc,err, fsal,offset, aTOL,rTOL,adaptive,errnorm,dU,ComputeRHS,dt,tstep,additional_args)
    return BS5

@optimizer
def RK4(context,u0, u1, u2, dU, a, b, dt, ComputeRHS,kw):
    """Runge Kutta fourth order"""
    U = context.mesh_vars["U"]
    u2[:] = u1[:] = u0
    for rk in range(4):
        dU = ComputeRHS(context,U,u0,dU, rk)
        if rk == 0 and "additional_callback" in kw:
            kw["additional_callback"](fU_hat=dU,**kw)
        if rk < 3:
            u0[:] = u1 + b[rk]*dt*dU
        u2 += a[rk]*dt*dU
    u0[:] = u2
    return u0,dt,dt

@optimizer
def ForwardEuler(context,u0, u1, dU, dt, ComputeRHS,kw):
    U = context.mesh_vars["U"]
    dU = ComputeRHS(context,U,u0,dU, 0)        
    if "additional_callback" in kw:
        kw["additional_callback"](fU_hat=dU,**kw)
    u0 += dU*dt
    return u0,dt,dt

#TODO: Check whether we are really only doing forward euler at first and
#last step
@optimizer
def AB2(context,u0, u1,multistep_dt, dU, dt, tstep, ComputeRHS,kw):
    U = context.mesh_vars["U"]
    dU = ComputeRHS(context,U,u0,dU, 0)
    if "additional_callback" in kw:
        kw["additional_callback"](fU_hat=dU,**kw)
    if tstep == 0 or multistep_dt[0] != dt:
        multistep_dt[0] = dt
        u0 += dU*dt
    else:
        u0 += (1.5*dU*dt - 0.5*u1)        
    u1[:] = dU*dt    
    return u0,dt,dt

def getintegrator(context,ComputeRHS,f=None,g=None,ginv=None):
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

    if ComputeRHS is None and context.time_integrator["time_integrator_name"] != "IMEX1":
        raise AssertionError("No ComputeRHS given for fully explicit time integrator")
        
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
    elif context.time_integrator["time_integrator_name"] == "BS5_adaptive": 
        TOL = context.time_integrator["TOL"]
        return getBS5(context,dU,ComputeRHS,aTOL=TOL,rTOL=TOL,adaptive=True)
    elif context.time_integrator["time_integrator_name"] == "BS5_fixed":
        TOL = 100 #This shouldn't get used
        return getBS5(context,dU,ComputeRHS,aTOL=TOL,rTOL=TOL,adaptive=False)
    elif context.time_integrator["time_integrator_name"] == "AB2":
        multistep_dt = [-1]
        @wraps(AB2)
        def func(t, tstep, dt,additional_args = {}):
            return AB2(context,u0, u1,multistep_dt, dU, dt, tstep, ComputeRHS,additional_args)
        return func
    elif context.time_integrator["time_integrator_name"] == "IMEX1":
        return getIMEXOneStep(context,dU,f,g,ginv)
    else:
        raise AssertionError("Please specifiy a  time integrator")
