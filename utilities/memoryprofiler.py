import subprocess
from os import getpid

__all__ = ['MemoryUsage']

def getMemoryUsage(rss=True):
    mypid = getpid()
    if rss:
        mymemory = subprocess.check_output(["ps -o rss %s" % mypid], shell=True).split()[1]
    else:
        mymemory = subprocess.check_output(["ps -o vsz %s" % mypid], shell=True).split()[1]
    return eval(mymemory) / 1024

class MemoryUsage:
    def __init__(self, s, comm):
        self.memory = 0
        self.memory_vm = 0
        self.comm = comm
        self.first = True
        self(s)
        
    def __call__(self, s, verbose=True):
        self.prev = self.memory
        self.prev_vm = self.memory_vm
        self.memory = self.comm.reduce(getMemoryUsage())
        self.memory_vm = self.comm.reduce(getMemoryUsage(False))
        if self.comm.Get_rank() == 0 and verbose:
            if self.first:
                print 'Memory usage                    RSS accum     RSS total   Virtual acc Virtual total'
                self.first = False
            print '{0:26s}  {1:10d} MB {2:10d} MB {3:10d} MB {4:10d} MB'.format(s, 
                   self.memory-self.prev, self.memory, self.memory_vm-self.prev_vm, self.memory_vm)
