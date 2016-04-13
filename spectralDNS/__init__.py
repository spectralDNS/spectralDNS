__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-09"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpi4py import MPI
comm = MPI.COMM_WORLD

from spectralDNS.context import *

def get_solver(update=None, regression_test=None, mesh="triplyperiodic",manually_specified_argv=None,additional_callback=None):
    argv = manually_specified_argv or sys.argv[1:]
    context = Context(mesh,argv)
    if update: context.callbacks["update"] = update
    if additional_callback: context.callbacks["additional_callback"] = additional_callback
    if regression_test: context.callbacks["regression_test"] = regression_test
    #TODO: what about set_source

    #TODO:uncomment below
    return context
