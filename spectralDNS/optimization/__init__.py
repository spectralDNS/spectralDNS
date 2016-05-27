__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-03-24"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from functools import wraps
from spectralDNS import config

def optimizer(func):
    """Decorator used to wrap calls to optimized versions of functions.
    
    Optimized versions of functions have the same name as
    in the main module. Two-dimensional version may be given the 
    postfix "_2D" and solver-specific implementations may be given
    as postfix the name of the solver. For example, the 
    "add_pressure_diffusion_NS" function defined in cython_solvers.py
    is an optimized version of "add_pressure_diffusion" for the 
    NS solver.
    
    """
    
    try: # Look for optimized version of function
        mod = eval("_".join((config.params.optimization, config.params.precision)))
            
        # Check for generic implementation first, then solver specific 
        name = func.func_name
        if config.params.decomposition == 'line':
            fun = getattr(mod, name+"_2D", None)
            
        else:
            fun = getattr(mod, name, None)
            
        if not fun:
            fun = getattr(mod, name+"_"+config.params.solver)
        
        @wraps(func)
        def wrapped_function(*args, **kwargs): 
            u0 = fun(*args, **kwargs)
            return u0
        
    except: # Otherwise revert to default numpy implementation
        #print func.func_name + ' not optimized'
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            u0 = func(*args, **kwargs)
            return u0

    return wrapped_function

try:
    import cython_double, cython_single
           
except:
    pass

try:   
    import numba_single, numba_double
    
except:
    pass

try:
    import numexpr_module

except:
    pass
