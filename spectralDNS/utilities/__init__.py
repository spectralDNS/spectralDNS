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
import numpy as np
from mpiFFT4py import dct
from spectralDNS import config
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

def dx(u, FST):
    """Compute integral of u over domain for channel solvers"""
    uu = np.sum(u, axis=(1, 2))
    sl = FST.local_slice(False)[0]
    M = FST.shape()[0]
    c = np.zeros(M)
    cc = np.zeros(M)
    cc[sl] = uu
    FST.comm.Reduce(cc, c, op=MPI.SUM, root=0)
    quad = FST.bases[0].quad
    if FST.comm.Get_rank() == 0:
        if quad == 'GL':
            ak = np.zeros_like(c)
            ak = dct(c, ak, 1, axis=0)
            ak /= (M-1)
            w = np.arange(0, M, 1, dtype=float)
            w[2:] = 2./(1-w[2:]**2)
            w[0] = 1
            w[1::2] = 0
            return sum(ak*w)*config.params.L[1]*config.params.L[2]/config.params.N[1]/config.params.N[2]

        assert quad == 'GC'
        d = np.zeros(M)
        k = 2*(1 + np.arange((M-1)//2))
        d[::2] = (2./M)/np.hstack((1., 1.-k*k))
        w = np.zeros_like(d)
        w = dct(d, w, type=3, axis=0)
        return np.sum(c*w)*config.params.L[1]*config.params.L[2]/config.params.N[1]/config.params.N[2]
    return 0
