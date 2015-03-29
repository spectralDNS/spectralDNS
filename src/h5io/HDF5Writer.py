__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-19"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""Wrap call to hdf5 to allow running without installing h5py
"""
__all__ = ['HDF5Writer']

try:
    import h5py
    class HDF5Writer(object):
    
        def __init__(self, comm, dt, N, params, dtype, filename="U.h5"):
            self.comm = comm
            self.components = components = ["U", "V", "W", "P"]
            if "eta" in params: components += ["Bx", "By", "Bz"]
            self.fname = filename
            self.params = params
            self.dtype = dtype
            self.dt = dt
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
            self.f["2D"].attrs.create("i", self.params['write_yz_slice'][0])            
            
        def write(self, U, P, tstep):
            if not self.f: self.init_h5file() 
            
            if tstep % self.params['write_result'] == 0 and self.params['decomposition'] == 'slab':
                rank = self.comm.Get_rank()
                N = self.f.attrs["N"]
                assert N == P.shape[-1]
                Np =  N / self.comm.Get_size()
                
                for comp in self.components:
                    self.f["3D/"+comp].create_dataset(str(tstep), shape=(N, N, N), dtype=self.dtype)
                                    
                self.f["3D/U/%d"%tstep][rank*Np:(rank+1)*Np] = U[0]
                self.f["3D/V/%d"%tstep][rank*Np:(rank+1)*Np] = U[1]
                self.f["3D/W/%d"%tstep][rank*Np:(rank+1)*Np] = U[2]
                self.f["3D/P/%d"%tstep][rank*Np:(rank+1)*Np] = P
                if len(self.components) == 7:
                    self.f["3D/Bx/%d"%tstep][rank*Np:(rank+1)*Np] = U[3]
                    self.f["3D/By/%d"%tstep][rank*Np:(rank+1)*Np] = U[4]
                    self.f["3D/Bz/%d"%tstep][rank*Np:(rank+1)*Np] = U[5]

            elif tstep % self.params['write_result'] == 0 and self.params['decomposition'] == 'pencil':
                N = self.f.attrs["N"]
                
                for comp in self.components:
                    self.f["3D/"+comp].create_dataset(str(tstep), shape=(N, N, N), dtype=self.dtype)
                                    
                x1, x2 = self.x1, self.x2
                self.f["3D/U/%d"%tstep][x1, x2, :] = U[0]
                self.f["3D/V/%d"%tstep][x1, x2, :] = U[1]
                self.f["3D/W/%d"%tstep][x1, x2, :] = U[2]
                self.f["3D/P/%d"%tstep][x1, x2, :] = P
                if len(self.components) == 7:
                    self.f["3D/Bx/%d"%tstep][x1, x2, :] = U[3]
                    self.f["3D/By/%d"%tstep][x1, x2, :] = U[4]
                    self.f["3D/Bz/%d"%tstep][x1, x2, :] = U[5]
                    
            if tstep % self.params['write_yz_slice'][1] == 0 and self.params['decomposition'] == 'slab':
                i = self.params['write_yz_slice'][0]
                rank = self.comm.Get_rank()
                N = self.f.attrs["N"]
                assert N == P.shape[-1]
                Np =  N / self.comm.Get_size()     
                for comp in self.components:
                    self.f["2D/"+comp].create_dataset(str(tstep), shape=(N, N), dtype=self.dtype)
                                    
                if i >= rank*Np and i < (rank+1)*Np:
                    self.f["2D/U/%d"%tstep][:] = U[0, i-rank*Np]
                    self.f["2D/V/%d"%tstep][:] = U[1, i-rank*Np]
                    self.f["2D/W/%d"%tstep][:] = U[2, i-rank*Np]
                    self.f["2D/P/%d"%tstep][:] = P[i-rank*Np]
                    if len(self.components) == 7:
                        self.f["2D/Bx/%d"%tstep][:] = U[3, i-rank*Np]
                        self.f["2D/By/%d"%tstep][:] = U[4, i-rank*Np]
                        self.f["2D/Bz/%d"%tstep][:] = U[5, i-rank*Np]

            elif tstep % self.params['write_yz_slice'][1] == 0 and self.params['decomposition'] == 'pencil':
                i = self.params['write_yz_slice'][0]
                N = self.f.attrs["N"]
                for comp in self.components:
                    self.f["2D/"+comp].create_dataset(str(tstep), shape=(N, N), dtype=self.dtype)
                                    
                x1, x2 = self.x1, self.x2
                self.f["2D/U/%d"%tstep][x1, x2] = U[0, :, :, i]
                self.f["2D/V/%d"%tstep][x1, x2] = U[1, :, :, i]
                self.f["2D/W/%d"%tstep][x1, x2] = U[2, :, :, i]
                self.f["2D/P/%d"%tstep][x1, x2] = P[:, :, i]
                if len(self.components) == 7:
                    self.f["2D/Bx/%d"%tstep][x1, x2] = U[3, :, :, i]
                    self.f["2D/By/%d"%tstep][x1, x2] = U[4, :, :, i]
                    self.f["2D/Bz/%d"%tstep][x1, x2] = U[5, :, :, i]
                            
        def close(self):
            if self.f: self.f.close()
                            
except:
    class HDF5Writer(object):
        def __init__(self, comm, dt, N, params, filename="U.h5"):
            if comm.Get_rank() == 0:
                print Warning("Need to install h5py to allow storing results")
        
        def write(self, U, P, tstep):
            pass
        
        def close(self):
            del self
