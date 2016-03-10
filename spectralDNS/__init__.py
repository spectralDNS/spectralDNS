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
            import spectralDNS.solvers.NS as solver
            
        elif config.solver == 'VV':
            import spectralDNS.solvers.VV as solver

        elif config.solver == 'MHD':
            import spectralDNS.solvers.MHD as solver
            
        else:
            raise AttributeError("Wrong solver!")

    elif mesh is "doublyperiodic":        
        args = config.doublyperiodic.parse_args()     
        vars(config).update(vars(args))
    
        if config.solver == 'NS2D':
            import spectralDNS.solvers.NS2D as solver

        elif config.solver == 'Bq2D':
            import spectralDNS.solvers.NS2D_Boussinesq as solver
            
        else:
            raise AttributeError("Wrong solver!")

    elif mesh is "channel":
        args = config.channel.parse_args()     
        vars(config).update(vars(args))
        
        if config.solver == 'IPCS':
            import spectralDNS.solvers.ShenDNS as solver           
            
        elif config.solver == 'IPCSR':
            import spectralDNS.solvers.ShenDNSR as solver
            
        elif config.solver == 'KMM':
            import spectralDNS.solvers.ShenKMM as solver

        elif config.solver == 'KMMRK3':
            import spectralDNS.solvers.ShenKMMRK3 as solver
        
        else:
            raise AttributeError("Wrong solver!")

    #elif family is "ShenMHD":
        #args = config.ShenMHD.parse_args()
        #vars(config).update(vars(args))

        #if config.solver == 'IPCS_MHD':
            #import spectralDNS.solvers.ShenMHD as solver            
        
        #else:
            #raise AttributeError("Wrong solver!")

    #elif family is "ShenGeneralBCs":
        #args = config.ShenGeneralBCs.parse_args()
        #vars(config).update(vars(args))
        
        #if config.solver == 'IPCS_GeneralBCs':
            #import spectralDNS.solvers.ShenDNSGeneralBCs as solver
            
        #else:
            #raise AttributeError("Wrong solver!")
    
    if update: solver.update = update
    if regression_test: solver.regression_test = regression_test
    return solver
