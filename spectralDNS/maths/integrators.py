__author__ = "Mikael Mortensen <mikaem@math.uio.no> and Nathanael Schilling <nathanael.schilling@in.tum.de>"
__date__ = "2015-04-07"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..optimization import optimizer, wraps
import numpy as np

__all__ = ['getintegrator']

def adaptiveRK(A, b, bhat, err_order, fY_hat, U_hat_new, sc, err, fsal, offset, 
               aTOL, rTOL, adaptive, errnorm, dU, U_hat, ComputeRHS, dt, tstep,
               FFT, additional_callback, params, args, predictivecontroller=False):
    """
    Take a step using any Runge-Kutta method.
    Parameters
    ----------
    A, b, bhat : arrays
        Runge-Kutta coefficients
    err_order : int
        Order of embedded method
    fY_hat, U_tmp, U_hat_new, sc, err : work arrays
    fsal : boolean
        Whether method is first-same-as-last
    offset : length-1 array of int
        Where to find the previous RHS evaluation (for FSAL methods).  This can probably be eliminated.
    aTOL, rTOL : float
        Error tolerances
    adaptive : boolean
        If true, adapt the step size
    errnorm : str
        Which norm to use in computing the error estimate.  One of {"2", "inf"}.
    dU : array
        RHS evaluation
    U_hat : array
        solution value (returned)
    ComputeRHS : callable
        RHS of evolution equation
    dt : float
        time step size
    tstep : int
        Number of steps taken so far
    predictivecontroller : boolean
        If True use PI controller
    """
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
        for i in range(0, s):
            if not fsal or (tstep == 0 or i != 0 ):
                fY_hat[(i + offset[0]) % s] =  U_hat
                for j in range(0,i):
                    fY_hat[(i+offset[0]) % s] += dt*A[i,j]*fY_hat[(j+offset[0]) % s]
                #Compute F(Y)
                dU = ComputeRHS(dU, fY_hat[(i+offset[0]) % s], *args)
                fY_hat[(i+offset[0]) % s] = dU
                
            if i == 0:
                additional_callback(fU_hat=fY_hat[(0+offset[0]) % s])
                
        #Calculate the new value
        U_hat_new[:] = U_hat
        U_hat_new[:] += dt*b[0]*fY_hat[(0+offset[0]) % s]
        err[:] = dt*(b[0] - bhat[0])*fY_hat[(0+offset[0]) % s]

        for j in range(1,s):
            U_hat_new[:] += dt*b[j]*fY_hat[(j+offset[0])%s]
            err[:] += dt*(b[j] - bhat[j])*fY_hat[(j+offset[0])%s]

        est = 0.0
        sc[:] = aTOL + np.maximum(np.abs(U_hat),np.abs(U_hat_new))*rTOL
        if errnorm == "2":
            est_to_bcast = None
            nsquared = np.zeros(U_hat.shape[0])
            for k in range(U_hat.shape[0]):
                nsquared[k] = FFT.comm.reduce(np.sum(np.power(np.abs(err[k]/sc[k]), 2)))
            if FFT.comm.rank == 0:
                est_to_bcast = np.zeros(1)
                est = np.max(np.sqrt(nsquared))
                est /= np.sqrt(np.array(FFT.global_complex_shape()).prod())
                est_to_bcast[0] = est
            est_to_bcast = FFT.comm.bcast(est_to_bcast,root=0)
            est = est_to_bcast[0]
            
        elif errnorm == "inf":
            raise AssertionError("Don't use this, not sure if it works")
            #TODO: Test this error norm
            sc[:] = aTOL + np.maximum(np.abs(U_hat),np.abs(U_hat_new))*rTOL
            err[:] = err[:] / sc[:]
            err = np.abs(err, out=err)
            asdf = np.max(err)
            x = np.zeros(asdf.shape)
            FFT.comm.Allreduce(asdf, x, op=MPI.MAX)
            est = np.abs(np.max(x))
            est /= np.sqrt(np.array(FFT.global_complex_shape()).prod())
        else:
            assert False, "Wrong error norm"

        #Check error estimate
        exponent = 1.0 / (err_order + 1)
        if not predictivecontroller:
            factor = min(facmax, max(facmin, fac*pow((1/est), exponent)))
        else:
            if not "last_dt" in vars(params):
                params.last_dt = dt
            if not "last_est" in vars(params):
                params.last_est = est

            last_dt = params.last_dt
            last_est = params.last_est
            factor = min(facmax, max(facmin, fac*pow((1/est), exponent)*dt/last_dt*pow(last_est/est, exponent)))
        if adaptive:
            dt = dt*factor
            if  est > 1.0:
                facmax = 1
                additional_callback(is_step_rejected_callback=True, dt_rejected=dt_prev)
                #The offset gets decreased in the  next step, which is something we do not want.
                if fsal:
                    offset[0] += 1
                continue

        #if predictivecontroller:
            #context.time_integrator["last_dt"] = dt_prev
            #context.time_integrator["last_est"] = est
        break


    #Update U_hat and U
    U_hat[:] = U_hat_new
    return U_hat, dt, dt_prev

@optimizer
def RK4(u0, u1, u2, dU, a, b, dt, ComputeRHS, args):
    """Runge Kutta fourth order"""
    u2[:] = u1[:] = u0
    for rk in range(4):
        dU = ComputeRHS(dU, u0, *args)
        if rk < 3:
            u0[:] = u1 + b[rk]*dt*dU
        u2 += a[rk]*dt*dU
    u0[:] = u2
    return u0, dt, dt

@optimizer
def ForwardEuler(u0, u1, dU, dt, ComputeRHS, args):
    dU = ComputeRHS(dU, u0, *args) 
    u0 += dU*dt
    return u0, dt, dt

@optimizer
def AB2(u0, u1, dU, dt, tstep, ComputeRHS, args):
    dU = ComputeRHS(dU, u0, *args)
    if tstep == 0:
        u0 += dU*dt
    else:
        u0 += (1.5*dU*dt - 0.5*u1)        
    u1[:] = dU*dt    
    return u0, dt, dt

def getintegrator(ComputeRHS, dU, u0, params, args):
    """Return integrator using choice in global parameter integrator.
    """
    #if params.solver in ("NS", "VV", "NS2D"):
        #u0 = U_hat
    #elif params.solver == "MHD":
        #u0 = kw['UB_hat']
    #elif params.solver == "Bq2D":
        #u0 = kw['Ur_hat']
    u1 = u0.copy()    
        
    if params.integrator == "RK4": 
        # RK4 parameters
        a = np.array([1./6., 1./3., 1./3., 1./6.], dtype=float)
        b = np.array([0.5, 0.5, 1.], dtype=float)
        u2 = u0.copy()
        @wraps(RK4)
        def func():
            return RK4(u0, u1, u2, dU, a, b, params.dt, ComputeRHS, args)
        return func

    elif params.integrator in ("BS5_adaptive", "BS5_fixed"): 
        import nodepy
        A = nodepy.rk.loadRKM("BS5").A.astype(float)
        b = nodepy.rk.loadRKM("BS5").b.astype(float)
        bhat = nodepy.rk.loadRKM("BS5").bhat.astype(float)
        err_order = 4
        errnorm = "2"
        fsal = True
        adaptive = True if params.integrator == "BS5_adaptive" else False

        #Offset for fsal stuff. #TODO: infer this from tstep
        offset = [0]
        s = A.shape[0]
        U = kw['U']
        fY_hat = np.zeros((s,) + u0.shape, dtype=u0.dtype)
        sc = np.zeros_like(u0)
        err = np.zeros_like(u0)

        @wraps(adaptiveRK)
        def func():
            return adaptiveRK(A, b, bhat, err_order, fY_hat, u1, sc, err, fsal, offset, 
                              params.TOL, params.TOL, adaptive, errnorm, dU, u0, ComputeRHS, 
                              params.dt, params.tstep, FFT, kw['additional_callback'], params,
                              args)

        return func

    elif params.integrator == "ForwardEuler":  
        @wraps(ForwardEuler)
        def func():
            return ForwardEuler(u0, u1, dU, params.dt, ComputeRHS, args)
        return func
    
    else:
        @wraps(AB2)
        def func():
            return AB2(u0, u1, dU, params.dt, params.tstep, ComputeRHS, args)
        return func
