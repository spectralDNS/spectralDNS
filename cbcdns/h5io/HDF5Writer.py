__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-19"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""Wrap call to hdf5 to allow running without installing h5py
"""
from cbcdns import config

__all__ = ['HDF5Writer']

try:
    import h5py
    class HDF5Writer(object):
    
        def __init__(self, comm, N, dtype, comps, filename="U.h5"):
            self.comm = comm
            self.components = comps
            self.fname = filename
            self.dtype = dtype
            self.dt = config.dt
            self.N = N
            self.f = None
            
        def init_h5file(self):
            self.f = h5py.File(self.fname, "w", driver="mpio", comm=self.comm)            
            self.f.create_group("3D")
            self.f.create_group("2D")
            for c in self.components:
                self.f["3D"].create_group(c)
                self.f["2D"].create_group(c)
            self.f.attrs.create("dt", self.dt)
            self.f.attrs.create("N", self.N)    
            self.f["2D"].attrs.create("i", config.write_yz_slice[0])            
            
        def write(self, tstep):
            if not self.f: self.init_h5file() 
            
            if tstep % config.write_result == 0 and config.decomposition == 'slab':
                rank = self.comm.Get_rank()
                N = self.f.attrs["N"]
                Np =  N / self.comm.Get_size()
                
                for comp, val in self.components.iteritems():
                    self.f["3D/"+comp].create_dataset(str(tstep), shape=(N, N, N), dtype=self.dtype)
                    self.f["3D/%s/%d"%(comp,tstep)][rank*Np:(rank+1)*Np] = val

            elif tstep % config.write_result == 0 and config.decomposition == 'pencil':
                N = self.f.attrs["N"]
                
                x1, x2 = self.x1, self.x2
                for comp, val in self.components.iteritems():
                    self.f["3D/"+comp].create_dataset(str(tstep), shape=(N, N, N), dtype=self.dtype)
                    self.f["3D/%s/%d"%(comp, tstep)][x1, x2, :] = val
                    
            elif tstep % config.write_result == 0 and config.decomposition == 'line':
                rank = self.comm.Get_rank()
                N = self.f.attrs["N"]
                Np =  N / self.comm.Get_size()
                
                for comp, val in self.components.iteritems():
                    self.f["2D/"+comp].create_dataset(str(tstep), shape=(N, N), dtype=self.dtype)
                    self.f["2D/%s/%d"%(comp,tstep)][rank*Np:(rank+1)*Np] = val                
                    
            if tstep % config.write_yz_slice[1] == 0 and config.decomposition == 'slab':
                i = config.write_yz_slice[0]
                rank = self.comm.Get_rank()
                N = self.f.attrs["N"]
                Np =  N / self.comm.Get_size()     
                for comp in self.components:
                    self.f["2D/"+comp].create_dataset(str(tstep), shape=(N, N), dtype=self.dtype)
                                    
                if i >= rank*Np and i < (rank+1)*Np:
                    for comp, val in self.components.iteritems():                
                        self.f["2D/%s/%d"%(comp, tstep)][:] = val[i-rank*Np]

            elif tstep % config.write_yz_slice[1] == 0 and config.decomposition == 'pencil':
                i = config.write_yz_slice[0]
                N = self.f.attrs["N"]
                x1, x2 = self.x1, self.x2
                for comp in self.components:
                    self.f["2D/"+comp].create_dataset(str(tstep), shape=(N, N), dtype=self.dtype)
                    for comp, val in self.components.iteritems():                                    
                        self.f["2D/%s/%d"%(comp, tstep)][x1, x2] = val[:, :, i]
                            
        def close(self):
            if self.f: self.f.close()
                            
except:
    class HDF5Writer(object):
        def __init__(self, comm, dt, N, comps, filename="U.h5"):
            if comm.Get_rank() == 0:
                print Warning("Need to install h5py to allow storing results")
        
        def write(self, U, P, tstep):
            pass
        
        def close(self):
            del self
