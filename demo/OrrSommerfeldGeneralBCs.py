from OrrSommerfeld import *

if __name__ == "__main__":
    config.update(
        {
        'solver': 'IPCS_GeneralBCs',
        'Re': 8000.,
        'nu': 1./8000.,             # Viscosity
        'dt': 0.01,                 # Time step
        'T': 0.02,                   # End time
        'L': [2, 2*pi, 4*pi/3.],
        'M': [7, 6, 1]
        },  "ShenGeneralBCs"
    )
    config.ShenGeneralBCs.add_argument("--compute_energy", type=int, default=1)
    config.ShenGeneralBCs.add_argument("--plot_step", type=int, default=10)
    solver = get_solver(update=update, regression_test=regression_test, family="ShenGeneralBCs")    
    vars(solver).update(initialize(**vars(solver)))
    set_Source(**vars(solver))
    solver.solve()
