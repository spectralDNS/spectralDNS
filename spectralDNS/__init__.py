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
