"""
SpectralDNS

High performance spectral Navier-Stokes (and similar) solvers implemented in
Python.
"""
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-09"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

import importlib
import cProfile
from . import config

#pylint: disable=eval-used,unused-variable

def get_solver(update=None,
               regression_test=None,
               additional_callback=None,
               mesh="triplyperiodic", parse_args=None):
    """Return solver based on global config (see spectralDNS/config.py)

    args:
        update               Update function called on each timestep.
                             Typically used for plotting or computing
                             intermediate results
        regression_test      Function called at the end of simulations.
                             Typically used for checking results
        additional_callback  Function used by some integrators that require
                             additional callback
        mesh                 Type of problem ('triplyperiodic',
                                              'doublyperiodic',
                                              'channel')
        parse_args           Used to specify arguments to config.
                             If parse_args is None then Commandline arguments
                             are used.

    global args:
        config               See spectralDNS/config.py for details.

    """
    assert parse_args is None or isinstance(parse_args, list)
    args = getattr(getattr(config, mesh), 'parse_args')(parse_args)
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
    """Generic solver for spectralDNS

    args:
        solver       The solver (e.g., NS or VV) module
        context      The solver's context

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

    while params.t + params.dt <= params.T+1e-12:

        u, params.dt, dt_took = integrate()

        params.t += dt_took
        params.tstep += 1

        solver.update(context)

        context.hdf5file.update(params, **context)

        solver.timer()

        if not solver.profiler.getstats() and params.make_profile:
            #Enable profiling after first step is finished
            solver.profiler.enable()

        if solver.end_of_tstep(context):
            break

    params.dt = dt_in

    solver.timer.final(params.verbose)

    if params.make_profile:
        solver.results = solver.create_profile(solver.profiler)

    solver.regression_test(context)

    context.hdf5file.close()
