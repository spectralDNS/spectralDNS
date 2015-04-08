__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-03-24"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from functools import wraps
import config

def optimizer(func):
    """Decorator used to wrap calls to optimized versions of functions.
    
    Optimized versions of functions are located in different modules 
    in src/optimization, where the functions have the same name as
    in the main module.
    
    """
    try: # Look for optimized version of function
        if config.optimization == "numexpr":
            fun = eval(".".join(("numexpr_module", func.func_name)))
        else:
            fun = eval("{0}_{1}.".format(config.optimization, config.precision)+func.func_name)
        @wraps(func)
        def wrapped_function(*args, **kwargs): 
            if config.optimization == "weave":
                fun(*args, **kwargs)
                u0 = args[0]
            else:
                u0 = fun(*args, **kwargs)
            return u0
        
    except: # Otherwise revert to default numpy implementation
        #print func.func_name + " not optimized"
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            u0 = func(*args, **kwargs)
            return u0

    return wrapped_function

try:
    import cython_single, cython_double
           
except:
    pass

try:
    import weave_single, weave_double
    
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

    