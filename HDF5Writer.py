__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-19"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""Wrap call to hdf5 to allow running without installing h5py
"""
HDF5Writer = None
try:
    import h5py
    class HDF5Writer(object):
    
        def __init__(self, comm, dt, N, filename="U.h5"):
            self.f = h5py.File("U.h5", "w", driver="mpio", comm=comm)
            self.comm = comm
            self.f.create_group("U")
            self.f.create_group("V")
            self.f.create_group("W")
            self.f.create_group("P")
            self.f.attrs.create("dt", dt)
            self.f.attrs.create("N", N)
            
        def write(self, U, P, tstep):
            rank = self.comm.Get_rank()
            N = self.f.attrs["N"]
            assert N == P.shape[-1]
            Np =  N / self.comm.Get_size()     
            self.f.create_dataset("/U/%d"%tstep, shape=(N, N, N), dtype="float")
            self.f.create_dataset("/V/%d"%tstep, shape=(N, N, N), dtype="float")
            self.f.create_dataset("/W/%d"%tstep, shape=(N, N, N), dtype="float")
            self.f.create_dataset("/P/%d"%tstep, shape=(N, N, N), dtype="float")
            self.f["/U/%d"%tstep][rank*Np:(rank+1)*Np] = U[0]
            self.f["/V/%d"%tstep][rank*Np:(rank+1)*Np] = U[1]
            self.f["/W/%d"%tstep][rank*Np:(rank+1)*Np] = U[2]
            self.f["/P/%d"%tstep][rank*Np:(rank+1)*Np] = P
            
        def close(self):
            self.f.close()

except:
    print Warning("Install h5py to allow storing results")
