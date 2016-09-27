__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-09"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import config
import importlib
import cProfile

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

def solve(solver, context):
    """Solve triply periodic Navier Stokes equations

    args:
        solver       The solver (e.g., NS or VV) module
        context      The solver's context from setup()

    global args:
        params       Dictionary (config.params) of parameters
                     that control the integration.
                     See spectralDNS.config.py for details
    """
    
    solver.timer = solver.Timer()
    params = solver.params
    
    solver.conv = solver.getConvection(params.convection)
    
    integrate = solver.getintegrator(context.dU, # rhs array
                                     context.u,  # primary variable
                                     solver,
                                     context)

    dt_in = params.dt

    while params.t + params.dt <= params.T+1e-15:

        u, params.dt, dt_took = integrate()

        params.t += dt_took
        params.tstep += 1

        solver.update(context)

        context.hdf5file.update(params, **context)

        solver.timer()

        if len(solver.profiler.getstats()) == 0 and params.make_profile:
            #Enable profiling after first step is finished
            solver.profiler.enable()

        # Make sure that the last step hits T exactly.
        if params.t + params.dt >= params.T:
            params.dt = params.T - params.t
            if params.dt <= 1.e-14:
                break

    params.dt = dt_in

    solver.timer.final(solver.MPI, solver.rank)

    if params.make_profile:
        solver.results = solver.create_profile(solver.profiler, solver.comm,
                                               solver.MPI, solver.rank)

    solver.regression_test(context)

    context.hdf5file.close()
