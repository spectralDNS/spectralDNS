__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-07"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..optimization import optimizer, wraps
from numpy import array
import spectralDNS.context

__all__ = ['getintegrator']


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
    return u0

@optimizer
def ForwardEuler(context,u0, u1, dU, dt, ComputeRHS,kw):
    U = context.mesh_vars["U"]
    dU = ComputeRHS(context,U,u0,dU, 0)        
    u0 += dU*dt
    return u0

@optimizer
def AB2(context,u0, u1, dU, dt, tstep, ComputeRHS,kw):
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
