__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-07"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import config
from ..optimization import optimizer, wraps

__all__ = ['getintegrator']

@optimizer
def RK4(U_hat, U_hat0, U_hat1, dU, a, b, dt, ComputeRHS):
    """Runge Kutta fourth order"""
    U_hat1[:] = U_hat0[:] = U_hat
    for rk in range(4):
        dU = ComputeRHS(dU, rk)
        if rk < 3:
            U_hat[:] = U_hat0 + b[rk]*dt*dU
        U_hat1 += a[rk]*dt*dU
    U_hat[:] = U_hat1
    return U_hat

@optimizer
def ForwardEuler(U_hat, U_hat0, dU, dt, ComputeRHS):
    dU = ComputeRHS(dU, 0)        
    U_hat += dU*dt
    return U_hat

@optimizer
def AB2(U_hat, U_hat0, dU, dt, tstep, ComputeRHS):
    dU = ComputeRHS(dU, 0)
    if tstep == 1:
        U_hat += dU*dt
    else:
        U_hat += (1.5*dU*dt - 0.5*U_hat0)        
    U_hat0[:] = dU*dt    
    return U_hat

def getintegrator(U_hat, U_hat0, U_hat1, dU, ComputeRHS, float, array, **soak):
    """Return integrator using choice in global parameter integrator.
    """
    if config.integrator == "RK4": 
        # RK4 parameters
        a = array([1./6., 1./3., 1./3., 1./6.], dtype=float)
        b = array([0.5, 0.5, 1.], dtype=float)
        @wraps(RK4)
        def func(t, tstep, dt):
            return RK4(U_hat, U_hat0, U_hat1, dU, a, b, dt, ComputeRHS)
        return func
            
    elif config.integrator == "ForwardEuler":  
        @wraps(ForwardEuler)
        def func(t, tstep, dt):
            return ForwardEuler(U_hat, U_hat0, dU, dt, ComputeRHS)
        return func
    else:
        @wraps(AB2)
        def func(t, tstep, dt):
            return AB2(U_hat, U_hat0, dU, dt, tstep, ComputeRHS)
        return func
