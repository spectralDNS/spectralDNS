import pytest
from spectralDNS import config, get_solver, solve
from TG2D import initialize, regression_test, pi

config.update(
{
    'nu': 0.01,
    'dt': 0.05,
    'T': 10}, 'doublyperiodic'
)

@pytest.fixture(params=('1', '2'))
def args(request):
    """Check for uniform and non-uniform cube"""
    if request.param[-1] == '1':
        _args = ['--M', '5', '5', '--L', '2*pi', '2*pi']
    else:
        _args = ['--M', '6', '4', '--L', '6*pi', '4*pi']
    
    return _args + ['NS2D']

def test_NS2D(args):
    solver = get_solver(regression_test=regression_test,
                        mesh='doublyperiodic',
                        parse_args=args)
    context = solver.get_context()
    initialize(**context)
    solve(solver, context)

    config.params.dealias = '3/2-rule'
    initialize(**context)
    solve(solver, context)

    config.params.dealias = '2/3-rule'
    config.params.optimization = 'cython'
    initialize(**context)
    solve(solver, context)

    config.params.write_result = 1
    config.params.checkpoint = 1
    config.params.dt = 0.01
    config.params.t = 0.0
    config.params.tstep = 0    
    config.params.T = 0.04
    solver.regression_test = lambda c: None
    initialize(**context)
    solve(solver, context)
