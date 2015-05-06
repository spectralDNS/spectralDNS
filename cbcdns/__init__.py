__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-09"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

#from utilities.MPI_knee import mpi_import
#with mpi_import():
from cbcdns import config

def get_solver(update=None):
    args = config.parser.parse_args()
    if args.solver in ('NS2D', 'Bq2D'):
        args.decomposition = 'line'
    vars(config).update(vars(args))
        
    #with mpi_import():
    if config.solver == 'NS':
        import cbcdns.solvers.spectralDNS as solver
        
    elif config.solver == 'VV':
        import cbcdns.solvers.spectralDNSVV as solver
        
    elif config.solver == 'NS2D':
        config.L = [config.L[0], config.L[1]]
        config.M = [config.M[0], config.M[1]]    
        import cbcdns.solvers.spectralDNS2D as solver
        
    elif config.solver == 'MHD':
        import cbcdns.solvers.spectralMHD3D as solver

    elif config.solver == 'Bq2D':
        config.L = [config.L[0], config.L[1]]
        config.M = [config.M[0], config.M[1]]    
        import cbcdns.solvers.spectralDNS2D_Boussinesq as solver
        
    else:
        raise AttributeError("Wrong solver!")
            
    if update: solver.update = update
    return solver
