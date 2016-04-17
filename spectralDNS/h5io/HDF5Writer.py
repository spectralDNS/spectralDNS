__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-19"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""Wrap call to hdf5 to allow running without installing h5py
"""
from spectralDNS import config
from numpy.linalg import norm

__all__ = ['HDF5Writer']
try:
    import h5py
    class HDF5Writer(object):

        def __init__(self, FFT, dtype, comps, filename="U.h5", mesh={}, fromfile=None):
            self.FFT = FFT
            self.components = comps
            self.fname = filename
            self.dtype = dtype
            self.f = None
            self.N = 2**config.M
            self.dim = len(comps[comps.keys()[0]].shape)
            self.mesh = mesh
            
        def init_h5file(self):
            self.f = h5py.File(self.fname, "w", driver="mpio", comm=self.FFT.comm)
            self.f.create_group("3D")
            self.f.create_group("2D")
            self.f["2D"].create_group("xy") # For slices in 3D geometries
            self.f["2D"].create_group("xz")
            self.f["2D"].create_group("yz")
            for c in self.components:
                self.f["2D"].create_group(c)
                self.f["3D"].create_group(c)
                self.f["2D/xy"].create_group(c)
                self.f["2D/xz"].create_group(c)
                self.f["2D/yz"].create_group(c)
                
            # Create groups for intermediate checkpoint solutions
            dim = str(self.dim)+"D"
            self.f[dim].create_group("checkpoint")
            self.f[dim].create_group("oldcheckpoint")            
            self.f[dim+"/checkpoint"].create_group("U")
            self.f[dim+"/checkpoint"].create_group("P")
            self.f[dim+"/oldcheckpoint"].create_group("U")
            self.f[dim+"/oldcheckpoint"].create_group("P")
            self.f.attrs.create("dt", config.dt)
            self.f.attrs.create("N", self.N)    
            self.f.attrs.create("L", config.L)    
            self.f["2D/yz"].attrs.create("i", config.write_yz_slice[0])
            self.f["2D/xy"].attrs.create("j", config.write_xy_slice[0])
            self.f["2D/xz"].attrs.create("k", config.write_xz_slice[0])
            
            if len(self.mesh) > 0:
                self.f["3D"].create_group("mesh")
            for key,val in self.mesh.iteritems():
                self.f["3D/mesh/"].create_dataset(key, shape=(len(val),), dtype=self.dtype)
                self.f["3D/mesh/"+key][:] = val
            
        def check_if_write(self, tstep):
            if config.write_result % tstep == 0:
                return True
            elif config.checkpoint % tstep == 0:
                return True 
            elif config.write_xy_slice[1] % tstep == 0:
                return True
            elif config.write_yz_slice[1] % tstep == 0:
                return True
            elif config.write_xz_slice[1] % tstep == 0:
                return True
            else:
                return False
            
        def checkpoint(self, U, P, U0):
            if self.f is None: self.init_h5file() 
            else:
                self.f = h5py.File(self.fname, driver="mpio", comm=self.FFT.comm)
            
            if config.decomposition in ("slab", "pencil"):
                shape = [3] + list(self.N)
                if not "0" in self.f["3D/checkpoint/U"].keys():
                    self.f["3D/checkpoint/U"].create_dataset("0", shape=shape, dtype=self.dtype)
                    self.f["3D/checkpoint/U"].create_dataset("1", shape=shape, dtype=self.dtype)
                    self.f["3D/checkpoint/P"].create_dataset("1", shape=self.N, dtype=self.dtype)
                    self.f["3D/oldcheckpoint/U"].create_dataset("0", shape=shape, dtype=self.dtype)
                    self.f["3D/oldcheckpoint/U"].create_dataset("1", shape=shape, dtype=self.dtype)
                    self.f["3D/oldcheckpoint/P"].create_dataset("1", shape=self.N, dtype=self.dtype)

            else:
                shape = [2] + self.N
                if not "0" in self.f["2D/checkpoint/U"].keys():
                    self.f["2D/checkpoint/U"].create_dataset("0", shape=shape, dtype=self.dtype)
                    self.f["2D/checkpoint/U"].create_dataset("1", shape=shape, dtype=self.dtype)
                    self.f["2D/checkpoint/P"].create_dataset("1", shape=self.N, dtype=self.dtype)
                    self.f["2D/oldcheckpoint/U"].create_dataset("0", shape=shape, dtype=self.dtype)
                    self.f["2D/oldcheckpoint/U"].create_dataset("1", shape=shape, dtype=self.dtype)
                    self.f["2D/oldcheckpoint/P"].create_dataset("1", shape=self.N, dtype=self.dtype)
                
            # Backup previous solution
            s = FFT.real_local_slice()
            self.f["3D/oldcheckpoint/U/0"][:, s]  = self.f["3D/checkpoint/U/0"][:, s]
            self.f["3D/oldcheckpoint/U/1"][:, s] = self.f["3D/checkpoint/U/1"][:, s]
            self.f["3D/oldcheckpoint/P/1"][s] = self.f["3D/checkpoint/P/1"][s]
            
            # Get new values
            self.f["3D/checkpoint/U/0"][:, s] = U0
            self.f["3D/checkpoint/U/1"][:, s] = U
            self.f["3D/checkpoint/P/1"][s] = P
            self.f.close()
            
        def write(self, tstep):
            if self.f is None: self.init_h5file() 
            else:
                self.f = h5py.File(self.fname, driver="mpio", comm=self.FFT.comm)
                
            N = self.N
            dim = str(self.dim)+"D"
            s = self.FFT.real_local_slice()
            if tstep % config.write_result == 0:                
                for comp, val in self.components.iteritems():
                    self.f[dim+"/"+comp].create_dataset(str(tstep), shape=N, dtype=self.dtype)
                    self.f[dim+"/%s/%d"%(comp,tstep)][s] = val

            # Write slices
            if tstep % config.write_yz_slice[1] == 0:
                i = config.write_yz_slice[0]
                for comp in self.components:
                    self.f["2D/yz/"+comp].create_dataset(str(tstep), shape=(N[1], N[2]), dtype=self.dtype)

                sx = s[0]
                if i >= sx.start and i < sx.stop:
                    for comp, val in self.components.iteritems():
                        self.f["2D/yz/%s/%d"%(comp, tstep)][s[1], s[2]] = val[i-sx.start]

            if tstep % config.write_xz_slice[1] == 0:
                j = config.write_xz_slice[0]                
                for comp in self.components:
                    self.f["2D/xz/"+comp].create_dataset(str(tstep), shape=(N[0], N[2]), dtype=self.dtype)
                
                if config.decomposition == 'slab':
                    for comp, val in self.components.iteritems():
                        self.f["2D/xz/%s/%d"%(comp,tstep)][s[0], s[2]] = val[:, j, :]
                        
                elif config.decomposition == 'pencil':
                    sy = s[1]
                    if j >= sy.start and j < sy.stop:
                        for comp, val in self.components.iteritems():
                            self.f["2D/xz/%s/%d"%(comp,tstep)][s[0], s[2]] = val[:, j-sy.start, :]

            if tstep % config.write_xy_slice[1] == 0:
                k = config.write_xy_slice[0]                
                for comp in self.components:
                    self.f["2D/xy/"+comp].create_dataset(str(tstep), shape=(N[0], N[1]), dtype=self.dtype)
                
                for comp, val in self.components.iteritems():
                    self.f["2D/xy/%s/%d"%(comp,tstep)][s[0], s[1]] = val[:, :, k]

            self.f.close()
            
        def close(self):
            if self.f: self.f.close()
                            
except:
    class HDF5Writer(object):
        def __init__(self, FFT, dtype, comps, filename="U.h5", mesh={}, fromfile=None):
            if FFT.comm.Get_rank() == 0:
                print Warning("Need to install h5py to allow storing results")
        
        def write(self, tstep):
            pass
        
        def check_if_write(self, tstep):
            return False
        
        def close(self):
            del self
