__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-09"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpi4py import MPI
from mpiFFT4py import Slab_R2C, Pencil_R2C, Line_R2C
from numpy import array
import config
import sys

def get_solver(update=None, regression_test=None, additional_callback=None, 
               mesh="triplyperiodic", parse_args=None):
    
    config.mesh = mesh
    if mesh is "triplyperiodic":
        
        if parse_args is None:
            args = config.triplyperiodic.parse_args()
        elif isinstance(parse_args, list):
            args = config.triplyperiodic.parse_args(parse_args)
        else:
            args = {}
        config.params.update(vars(args))

        if config.params.solver == 'NS':
            import spectralDNS.solvers.NS as solver
            
        elif config.params.solver == 'VV':
            import spectralDNS.solvers.VV as solver

        elif config.params.solver == 'MHD':
            import spectralDNS.solvers.MHD as solver
            
        else:
            raise AttributeError("Wrong solver!")

    elif mesh is "doublyperiodic":   
        
        if parse_args is None:
            args = config.doublyperiodic.parse_args()
        elif isinstance(parse_args, list):
            args = config.doublyperiodic.parse_args(parse_args)
        else:
            args = {}
        config.params.update(vars(args))

        if config.params.solver == 'NS2D':
            import spectralDNS.solvers.NS2D as solver

        elif config.params.solver == 'Bq2D':
            import spectralDNS.solvers.NS2D_Boussinesq as solver
            
        else:
            raise AttributeError("Wrong solver!")

    elif mesh is "channel":
        if parse_args is None:
            args = config.channel.parse_args()
        elif isinstance(parse_args, list):
            args = config.channel.parse_args(parse_args)
        else:
            args = {}
        config.params.update(vars(args))
        
        if config.params.solver == 'IPCS':
            import spectralDNS.solvers.ShenDNS as solver           
            
        elif config.params.solver == 'IPCSR':
            import spectralDNS.solvers.ShenDNSR as solver
            
        elif config.params.solver == 'KMM':
            import spectralDNS.solvers.ShenKMM as solver

        elif config.params.solver == 'KMMRK3':
            import spectralDNS.solvers.ShenKMMRK3 as solver
        
        else:
            raise AttributeError("Wrong solver!")
    
    if update: solver.update = update
    if regression_test: solver.regression_test = regression_test
    if additional_callback: solver.additional_callback = additional_callback        
    return solver
