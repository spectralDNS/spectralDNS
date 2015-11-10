__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-19"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""Wrap call to hdf5 to allow running without installing h5py
"""
from cbcdns import config
from numpy.linalg import norm

__all__ = ['HDF5Writer']

try:
    import h5py
    class HDF5Writer(object):
    
        def __init__(self, comm, dtype, comps, filename="U.h5", mesh={}):
            self.comm = comm
            self.components = comps
            self.fname = filename
            self.dtype = dtype
            self.f = None
            self.N = 2**config.M
            self.rank = self.comm.Get_rank()
            self.dim = len(comps[comps.keys()[0]].shape)
            self.mesh = mesh
            num_processes = self.comm.Get_size()
            if config.decomposition == "pencil":
                commxz = comm.Split(self.rank/config.P1)
                commxy = comm.Split(self.rank%config.P1)    
                xzrank = commxz.Get_rank() # Local rank in xz-plane
                xyrank = commxy.Get_rank() # Local rank in xy-plane 
                P2 = num_processes / config.P1
                N1 = self.N/config.P1
                N2 = self.N/P2
                self.x1 = slice(xzrank * N1[0], (xzrank+1) * N1[0], 1)
                self.x2 = slice(xyrank * N2[1], (xyrank+1) * N2[1], 1)
            
        def init_h5file(self):
            self.f = h5py.File(self.fname, "w", driver="mpio", comm=self.comm)            
            self.f.create_group("3D")
            self.f.create_group("2D")    # For slices in 3D geometries
            for c in self.components:
                self.f["2D"].create_group(c)
                self.f["3D"].create_group(c)
                
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
            self.f["2D"].attrs.create("i", config.write_yz_slice[0])
            if len(self.mesh) > 0:
                self.f["3D"].create_group("mesh")
            for key,val in self.mesh.iteritems():
                self.f["3D/mesh/"].create_dataset(key, shape=(len(val),), dtype=self.dtype)
                self.f["3D/mesh/"+key][:] = val
            
        def checkpoint(self, U, P, U0):
            if self.f is None: self.init_h5file() 
            else:
                self.f = h5py.File(self.fname, driver="mpio", comm=self.comm)
            
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

            if config.decomposition == 'slab':
                
                Np = self.N / self.comm.Get_size()
                
                # Backup previous solution
                s = slice(self.rank*Np[0], (self.rank+1)*Np[0], 1)
                self.f["3D/oldcheckpoint/U/0"][:, s]  = self.f["3D/checkpoint/U/0"][:, s]
                self.f["3D/oldcheckpoint/U/1"][:, s] = self.f["3D/checkpoint/U/1"][:, s]
                self.f["3D/oldcheckpoint/P/1"][s] = self.f["3D/checkpoint/P/1"][s]
                
                # Get new values
                self.f["3D/checkpoint/U/0"][:, s] = U0
                self.f["3D/checkpoint/U/1"][:, s] = U
                self.f["3D/checkpoint/P/1"][s] = P

            elif config.decomposition == 'pencil':
                
                x1, x2 = self.x1, self.x2
                # Backup previous solution
                self.f["3D/oldcheckpoint/U/0"][:, x1, x2] = self.f["3D/checkpoint/U/0"][:, x1, x2]
                self.f["3D/oldcheckpoint/U/1"][:, x1, x2] = self.f["3D/checkpoint/U/1"][:, x1, x2]
                self.f["3D/oldcheckpoint/P/1"][:, x1, x2] = self.f["3D/checkpoint/P/1"][:, x1, x2]
                # Get new values
                self.f["3D/checkpoint/U/0"][:, x1, x2] = U0
                self.f["3D/checkpoint/U/1"][:, x1, x2] = U
                self.f["3D/checkpoint/P/1"][x1, x2] = P
                
            elif config.decomposition == 'line':
                
                Np =  N / self.comm.Get_size()                
                # Backup previous solution
                s = slice(self.rank*Np[0], (self.rank+1)*Np[0], 1)
                self.f["2D/oldcheckpoint/U/0"][:, ] = self.f["2D/checkpoint/U/0"][:, s]
                self.f["2D/oldcheckpoint/U/1"][:, s] = self.f["2D/checkpoint/U/1"][:, s]
                self.f["2D/oldcheckpoint/P/1"][:, s] = self.f["2D/checkpoint/P/1"][:, s]
                # Get new values
                self.f["2D/checkpoint/U/0"][:, s] = U0
                self.f["2D/checkpoint/U/1"][:, s] = U
                self.f["2D/checkpoint/P/1"][s] = P
            self.f.close()
            
        def write(self, tstep):
            if self.f is None: self.init_h5file() 
            else:
                self.f = h5py.File(self.fname, driver="mpio", comm=self.comm)
                
            N = self.N
            
            if tstep % config.write_result == 0 and config.decomposition == 'slab':
                
                Np = N / self.comm.Get_size()                
                for comp, val in self.components.iteritems():
                    self.f["3D/"+comp].create_dataset(str(tstep), shape=N, dtype=self.dtype)
                    self.f["3D/%s/%d"%(comp,tstep)][self.rank*Np[0]:(self.rank+1)*Np[0]] = val

            elif tstep % config.write_result == 0 and config.decomposition == 'pencil':
                
                x1, x2 = self.x1, self.x2
                for comp, val in self.components.iteritems():
                    self.f["3D/"+comp].create_dataset(str(tstep), shape=N, dtype=self.dtype)
                    self.f["3D/%s/%d"%(comp, tstep)][x1, x2, :] = val
                    
            elif tstep % config.write_result == 0 and config.decomposition == 'line':
                
                Np =  N / self.comm.Get_size()                
                for comp, val in self.components.iteritems():
                    self.f["2D/"+comp].create_dataset(str(tstep), shape=(N[0], N[1]), dtype=self.dtype)
                    self.f["2D/%s/%d"%(comp,tstep)][self.rank*Np[0]:(self.rank+1)*Np[0]] = val                
                    
            if tstep % config.write_yz_slice[1] == 0 and config.decomposition == 'slab':
                i = config.write_yz_slice[0]
                Np = N / self.comm.Get_size()     
                for comp in self.components:
                    self.f["2D/"+comp].create_dataset(str(tstep), shape=(N[1], N[2]), dtype=self.dtype)
                                    
                if i >= self.rank*Np[0] and i < (self.rank+1)*Np[0]:
                    for comp, val in self.components.iteritems():                
                        self.f["2D/%s/%d"%(comp, tstep)][:] = val[i-self.rank*Np[0]]

            elif tstep % config.write_yz_slice[1] == 0 and config.decomposition == 'pencil':
                i = config.write_yz_slice[0]
                x1, x2 = self.x1, self.x2
                for comp in self.components:
                    self.f["2D/"+comp].create_dataset(str(tstep), shape=(N[1], N[2]), dtype=self.dtype)
                    for comp, val in self.components.iteritems():                                    
                        self.f["2D/%s/%d"%(comp, tstep)][x1, x2] = val[:, :, i]
            self.f.close()
            
        def close(self):
            if self.f: self.f.close()
                            
except:
    class HDF5Writer(object):
        def __init__(self, comm, N, dtype, comps, filename="U.h5"):
            if comm.Get_rank() == 0:
                print Warning("Need to install h5py to allow storing results")
        
        def write(self, tstep):
            pass
        
        def close(self):
            del self
