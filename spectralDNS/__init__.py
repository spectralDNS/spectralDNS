__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-09"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import config
import importlib

def get_solver(update=None,
               regression_test=None,
               additional_callback=None,
               mesh="triplyperiodic", parse_args=None):
    assert parse_args is None or isinstance(parse_args, list)
    args = getattr(eval('.'.join(('config', mesh))),
                    'parse_args')(parse_args)
    config.params.update(vars(args))
    
    try:
        solver = importlib.import_module('.'.join(('spectralDNS.solvers',
                                                   config.params.solver)))
    except AttributeError:
        raise AttributeError("Wrong solver!")

    if update:
        solver.update = update

    if regression_test:
        solver.regression_test = regression_test

    if additional_callback:
        solver.additional_callback = additional_callback
        
    # Create link to solver module in config
    config.solver = solver

    return solver
