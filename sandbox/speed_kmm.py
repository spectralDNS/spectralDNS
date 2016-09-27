from numpy import array, pi, log, arange
from MKM import config, get_solver, solve, initialize, set_Source
from mpi4py import MPI

comm = MPI.COMM_WORLD

N = arange(5, 13)
M = 1

config.update(
    {
    'nu': 1./590.,                  # Viscosity
    'Re_tau': 590., 
    'dt': 0.001,                    # Time step
    'T': 0.01,                       # End time
    'L': [2, 2*pi, pi],
    'optimization': 'cython',
    'make_profile': 1,
    'M': [5, 5, 5]
    },  "channel"
)

solver = get_solver(mesh="channel")
params = config.params

t = []
for n in N:
    params.M = [n, 4, 4]
    context = solver.get_context()
    initialize(solver, context)
    params.t = 0
    params.tstep = 0
    set_Source(**context)
    solve(solver, context)
    t.append([comm.reduce(solver.timer.fastest_timestep, op=MPI.MIN, root=0),
              solver.results['ComputeRHS'][2],
              solver.results['solve_linear'][2]
              ])
    solver.profiler.clear()
    
print t
print "$N_x$ & Total ($\\frac{t_k }{t_{k-1}} \\frac{\log N-1}{2 \log N}$) & Assemble ($\\frac{t_k }{t_{k-1}} \\frac{\log N-1}{2 \log N}$) & Solve ($\\frac{t_k}{t_{k-1}} \\frac{1}{2}$) \\\ "
print "\hline"
for i, n in enumerate(N):
    err = str(2**n)
    err += " & {:2.3f} ({:2.2f}) & {:2.2e} ({:2.2f}) & {:2.2e} ({:2.2f}) \\\ ".format(t[i][0], 0 if i == 0 else t[i][0]/t[i-1][0]/(2.*log(n)/log(n-1)), 
                                                                                      t[i][1], 0 if i == 0 else t[i][1]/t[i-1][1]/(2.*log(n)/log(n-1)),
                                                                                      t[i][2], 0 if i == 0 else t[i][2]/t[i-1][2]/2.)
    print err
