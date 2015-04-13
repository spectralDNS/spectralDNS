__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-03-24"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from functools import wraps
from cbcdns import config

def optimizer(func):
    """Decorator used to wrap calls to optimized versions of functions.
    
    Optimized versions of functions are located in different modules 
    in src/optimization, where the functions have the same name as
    in the main module.
    
    """

    try: # Look for optimized version of function
        name = func.func_name
        if config.decomposition == 'line':
            name += '_2D'
        if config.optimization in ('numexpr', ):
            fun = eval('.'.join(('{0}_module'.format(config.optimization), name)))
        else:
            fun = eval('{0}_{1}.'.format(config.optimization, config.precision)+name)
            
        @wraps(func)
        def wrapped_function(*args, **kwargs): 
            if config.optimization == 'weave':
                fun(*args, **kwargs)
                u0 = args[0]
            else:
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
    #import cython_module
           
except:
    pass

try:
    import weave_single, weave_double
    from numpy import int64
    
    def crossi(c, a, b):
        if a.dtype == int64:
            c = weave_single.cross2a(c, a, b)
        else:
            c = weave_single.cross2b(c, a, b)
        return c
    weave_single.cross2 = crossi
    
    def crossd(c, a, b):
        if a.dtype == int64:
            c = weave_double.cross2a(c, a, b)
        else:
            c = weave_double.cross2b(c, a, b)
        return c
    weave_double.cross2 = crossd
    
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

    