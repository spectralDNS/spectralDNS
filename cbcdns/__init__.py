__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-09"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpi4py import MPI
comm = MPI.COMM_WORLD
import config

def get_solver(update=None, regression_test=None, mesh="triplyperiodic"):
    
    config.mesh = mesh
    if mesh is "triplyperiodic":
        
        args = config.triplyperiodic.parse_args()     
        vars(config).update(vars(args))
            
        if config.solver == 'NS':
            import cbcdns.solvers.spectralDNS as solver
            
        elif config.solver == 'VV':
            import cbcdns.solvers.spectralDNSVV as solver

        elif config.solver == 'MHD':
            import cbcdns.solvers.spectralMHD3D as solver
            
        else:
            raise AttributeError("Wrong solver!")

    elif mesh is "doublyperiodic":        
        args = config.doublyperiodic.parse_args()     
        vars(config).update(vars(args))
    
        if config.solver == 'NS2D':
            import cbcdns.solvers.spectralDNS2D as solver

        elif config.solver == 'Bq2D':
            import cbcdns.solvers.spectralDNS2D_Boussinesq as solver
            
        else:
            raise AttributeError("Wrong solver!")

    elif mesh is "channel":
        args = config.channel.parse_args()     
        vars(config).update(vars(args))
        
        if config.solver == 'IPCS':
            import cbcdns.solvers.ShenDNS as solver           
            
        elif config.solver == 'IPCSR':
            import cbcdns.solvers.ShenDNSR as solver
            
        elif config.solver == 'KMM':
            import cbcdns.solvers.ShenKMM as solver

        elif config.solver == 'KMMRK3':
            import cbcdns.solvers.ShenKMMRK3 as solver
        
        else:
            raise AttributeError("Wrong solver!")

    #elif family is "ShenMHD":
        #args = config.ShenMHD.parse_args()
        #vars(config).update(vars(args))

        #if config.solver == 'IPCS_MHD':
            #import cbcdns.solvers.ShenMHD as solver            
        
        #else:
            #raise AttributeError("Wrong solver!")

    #elif family is "ShenGeneralBCs":
        #args = config.ShenGeneralBCs.parse_args()
        #vars(config).update(vars(args))
        
        #if config.solver == 'IPCS_GeneralBCs':
            #import cbcdns.solvers.ShenDNSGeneralBCs as solver
            
        #else:
            #raise AttributeError("Wrong solver!")
    
    if update: solver.update = update
    if regression_test: solver.regression_test = regression_test
    return solver
