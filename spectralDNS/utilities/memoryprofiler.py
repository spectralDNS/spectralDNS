"""
Module for inspecting memory usage of spectralDNS
"""
import subprocess
from os import getpid
from mpi4py import MPI

__all__ = ['MemoryUsage']

def _getMemoryUsage(rss=True):
    mypid = getpid()
    if rss:
        mymemory = subprocess.check_output(["ps -o rss %s" % mypid], shell=True).split()[1]
    else:
        mymemory = subprocess.check_output(["ps -o vsz %s" % mypid], shell=True).split()[1]
    return int(mymemory) // 1024

class MemoryUsage:
    """Class for inspecting memory usage

    args:
        s          str        Descriptive name
    """
    def __init__(self, s):
        self.memory = 0
        self.memory_vm = 0
        self.first = True
        self(s)

    def __call__(self, s, verbose=True):
        prev = self.memory
        prev_vm = self.memory_vm
        self.memory = MPI.COMM_WORLD.reduce(_getMemoryUsage())
        self.memory_vm = MPI.COMM_WORLD.reduce(_getMemoryUsage(False))
        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            if self.first:
                print("Memory usage                    RSS accum     RSS total   Virtual  Virtual total")
                self.first = False
            out = "{0:26s}  {1:10d} MB {2:10d} MB {3:10d} MB {4:10d} MB"
            print(out.format(s, self.memory-prev, self.memory,
                             self.memory_vm-prev_vm, self.memory_vm))
