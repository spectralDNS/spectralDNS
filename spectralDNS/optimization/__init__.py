__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-03-24"
__copyright__ = "Copyright (C) 2015-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

import importlib
from functools import wraps
from spectralDNS import config

#pylint: disable=bare-except,no-member

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
        mod = globals()["_".join((config.params.optimization,
                                  config.params.precision))]

        # Check for generic implementation first, then solver specific
        name = func.__name__
        if len(config.params.N) == 2:
            fun = getattr(mod, name+"_2D", None)

        else:
            fun = getattr(mod, name, None)

        if not fun:
            fun = getattr(mod, name+"_"+config.params.solver, None)

        if not fun:
            fun = getattr(mod, name+"_"+config.mesh, None)

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            u0 = fun(*args, **kwargs)
            return u0

    except: # Otherwise revert to default numpy implementation
        print(func.__name__ + ' not optimized')
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            u0 = func(*args, **kwargs)
            return u0

    return wrapped_function

try:
    from . import cython_double, cython_single

except:
    pass

try:
    from . import numba_module as numba_single
    from . import numba_module as numba_double
except:
    pass

try:
    from . import numexpr_module

except:
    pass

try:
    from . import pythran_module as pythran_single
    from . import pythran_module as pythran_double
except:
    pass
