import pytest
from spectralDNS import config, get_solver, solve
from OrrSommerfeld import initialize, regression_test, set_Source, pi
config.update(
    {
    'Re': 8000.,
    'nu': 1./8000.,             # Viscosity
    'dt': 0.001,                 # Time step
    'T': 0.01,                   # End time
    'L': [2, 2*pi, 4*pi/3.],
    'M': [7, 5, 2]
    },  "channel"
)

@pytest.fixture(params=('KMM', 'KMMRK3', 'IPCS', 'IPCSR'))
def sol(request):
    """Check for uniform and non-uniform cube"""
    return request.param

def test_channel(sol):
    solver = get_solver(regression_test=regression_test,
                        mesh="channel",
                        parse_args=[sol])
    context = solver.get_context()
    initialize(solver, context)
    set_Source(**context)
    solve(solver, context)

    config.params.dealias = '3/2-rule'
    config.params.optimization = 'cython'
    initialize(solver, context)
    solve(solver, context)

    config.params.dealias_cheb = True
    config.params.checkpoint = 5
    config.params.write_result = 2
    initialize(solver, context)
    solve(solver, context)

if __name__=='__main__':
    test_channel('KMM')
