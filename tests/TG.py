from TG_link import get_solver, config, solve, initialize, regression_test, pi
import sys

config.update(
    {
    'nu': 0.000625,             # Viscosity
    'dt': 0.01,                 # Time step
    'T': 0.1,                   # End time
    'L': [2*pi, 2*pi, 2*pi],
    'M': [4, 4, 4]
    }
)

for sol in ['NS', 'VV']:
    solver = get_solver(regression_test=regression_test, 
                        parse_args=sys.argv[1:]+[sol])
    context = solver.get_context()
    initialize(solver, **context)
    solve(solver, context)

    config.params.make_profile = 1
    config.params.dealias = '3/2-rule'
    initialize(solver, **context)
    solve(solver, context)
    
    config.params.dealias = '2/3-rule'
    config.params.optimization = 'cython'
    initialize(solver, **context)
    solve(solver, context)

    config.params.write_result = 1
    config.params.checkpoint = 1
    config.params.dt = 0.01
    config.params.t = 0.0
    config.params.tstep = 0
    config.params.T = 0.04
    solver.regression_test = lambda c: None
    solve(solver, context)
