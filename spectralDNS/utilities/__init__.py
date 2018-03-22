"""
Module for spectralDNS utilities
"""
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from time import time
import types
from mpi4py import MPI
from .create_profile import create_profile, reset_profile
from .memoryprofiler import MemoryUsage

class Timer(object):
    """Class for timing solvers in spectralDNS

    The Timer class is designed to sample the computing time of complete
    time steps of the solvers. We store the fastest and slowest time step
    on each process, and then the results may be reduced calling method
    final (in the end).
    """

    def __init__(self):
        self.fastest_timestep = 1e8
        self.slowest_timestep = 0
        self.t0 = time()
        self.tic = self.t0

    def __call__(self):
        """Call for intermediate sampling of timings of complete time steps"""
        t1 = time()
        dt = t1 - self.t0
        self.fastest_timestep = min(dt, self.fastest_timestep)
        self.slowest_timestep = max(dt, self.slowest_timestep)
        self.t0 = t1

    def final(self, verbose=True):
        """Called at the end of a simulation.

        Reduces the results of individual processors to the zero rank,
        and prints the results if verbose is True.

        The results computed as as such:

        Each processor samples timings each timestep and stores its own
        fastest and slowest timestep.

        Fastest: (Fastest of the fastest measurements across all processors,
                  Fastest of the slowest measurements across all processors)

        Slowest: (Slowest of the fastest measurements across all processors,
                  Slowest of the slowest measurements across all processors)
        """
        comm = MPI.COMM_WORLD
        fast = (comm.reduce(self.fastest_timestep, op=MPI.MIN, root=0),
                comm.reduce(self.slowest_timestep, op=MPI.MIN, root=0))
        slow = (comm.reduce(self.fastest_timestep, op=MPI.MAX, root=0),
                comm.reduce(self.slowest_timestep, op=MPI.MAX, root=0))

        toc = time() - self.tic
        if comm.Get_rank() == 0 and verbose:
            print("Time = {}".format(toc))
            print("Fastest = {}".format(fast))
            print("Slowest = {}".format(slow))


def inheritdocstrings(cls):
    """Method for inheriting docstrings from parent class"""
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls
