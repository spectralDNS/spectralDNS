__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-07"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from cbcdns import config
from ..optimization import optimizer, wraps

__all__ = ['getintegrator']

@optimizer
def RK4(u0, u1, u2, dU, a, b, dt, ComputeRHS):
    """Runge Kutta fourth order"""
    u2[:] = u1[:] = u0
    for rk in range(4):
        dU = ComputeRHS(dU, rk)
        if rk < 3:
            u0[:] = u1 + b[rk]*dt*dU
        u2 += a[rk]*dt*dU
    u0[:] = u2
    return u0

@optimizer
def ForwardEuler(u0, u1, dU, dt, ComputeRHS):
    dU = ComputeRHS(dU, 0)        
    u0 += dU*dt
    return u0

@optimizer
def AB2(u0, u1, dU, dt, tstep, ComputeRHS):
    dU = ComputeRHS(dU, 0)
    if tstep == 1:
        u0 += dU*dt
    else:
        u0 += (1.5*dU*dt - 0.5*u1)        
    u1[:] = dU*dt    
    return u0

def getintegrator(dU, ComputeRHS, float, array, **kw):
    """Return integrator using choice in global parameter integrator.
    """
    if config.solver in ("NS", "VV", "NS2D", "ChannelRK4"):
        u0 = kw['U_hat']
    elif config.solver == "MHD":
        u0 = kw['UB_hat']
    elif config.solver == "Bq2D":
        u0 = kw['Ur_hat']
    u1 = u0.copy()    
        
    if config.integrator == "RK4": 
        # RK4 parameters
        a = array([1./6., 1./3., 1./3., 1./6.], dtype=float)
        b = array([0.5, 0.5, 1.], dtype=float)
        u2 = u0.copy()
        @wraps(RK4)
        def func(t, tstep, dt):
            return RK4(u0, u1, u2, dU, a, b, dt, ComputeRHS)
        return func
            
    elif config.integrator == "ForwardEuler":  
        @wraps(ForwardEuler)
        def func(t, tstep, dt):
            return ForwardEuler(u0, u1, dU, dt, ComputeRHS)
        return func
    else:
        @wraps(AB2)
        def func(t, tstep, dt):
            return AB2(u0, u1, dU, dt, tstep, ComputeRHS)
        return func
