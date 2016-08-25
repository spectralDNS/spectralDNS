from TG import *

config.update(
    {
    'nu': 0.000625,             # Viscosity
    'dt': 0.01,                 # Time step
    'T': 0.1,                   # End time
    'L': [2*pi, 2*pi, 2*pi],
    'M': [5, 5, 5]
    }
)
for sol in ['NS', 'VV']:
    solver = get_solver(regression_test=regression_test,
                        parse_args=[sol])
    context = solver.get_context()
    for integrator in ('RK4', 'ForwardEuler', 'AB2', 'BS5_adaptive', 'BS5_fixed'):
        if integrator in ('ForwardEuler', 'AB2'):
            config.params.ntol = 4
        else:
            config.params.ntol = 7
        config.params.integrator = integrator
        initialize(solver, **context)
        solve(solver, context)
