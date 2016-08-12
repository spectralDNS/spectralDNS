__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-19"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""Wrap call to hdf5 to allow running without installing h5py
"""
from numpy.linalg import norm

__all__ = ['HDF5Writer']

try:
    import h5py
    class HDF5Writer(object):

        def __init__(self, FFT, dtype, comps, chkpoint={}, filename="U.h5", mesh={}):
            self.FFT = FFT
            self.components = comps
            self.chkpoint = chkpoint
            self.fname = filename
            self.dtype = dtype
            self.f = None
            self.N = FFT.N
            self.dim = len(comps[comps.keys()[0]].shape)
            self.mesh = mesh
            self.updatestep = 0

        def init_h5file(self, params):
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
            for key in self.chkpoint['current']:
                self.f[dim+"/checkpoint"].create_group(key)

            self.f.attrs.create("dt", params.dt)
            self.f.attrs.create("N", self.N)    
            self.f.attrs.create("L", params.L)    
            if params.has_key('write_yz_slice'):
                self.f["2D/yz"].attrs.create("i", params.write_yz_slice[0])
                self.f["2D/xy"].attrs.create("j", params.write_xy_slice[0])
                self.f["2D/xz"].attrs.create("k", params.write_xz_slice[0])
            
            if len(self.mesh) > 0:
                self.f["3D"].create_group("mesh")
            for key, val in self.mesh.iteritems():
                self.f["3D/mesh/"].create_dataset(key, shape=(len(val),), dtype=self.dtype)
                self.f["3D/mesh/"+key][:] = val

        def update(self, **kw):
            self.update_components(**kw)
            params = kw['params']
            if self.check_if_write(params):
                self.write(params)

            if params.tstep % params.checkpoint == 0:
                self.checkpoint(params)

        def check_if_write(self, params):
            if params.tstep % params.write_result == 0:
                return True

            if params.has_key('write_xy_slice'):
                if (params.tstep % params.write_xy_slice[1] == 0 or 
                    params.tstep % params.write_yz_slice[1] == 0 or
                    params.tstep % params.write_xz_slice[1] == 0):
                    return True
            return False

        def checkpoint(self, params):
            if self.f is None: self.init_h5file(params) 
            else:
                self.f = h5py.File(self.fname, driver="mpio", comm=self.FFT.comm)

            dim = str(self.dim)+"D"
            keys = self.chkpoint['current'].keys()
            # Create datasets first time around
            if not "1" in self.f["{}/checkpoint/{}".format(dim, keys[0])].keys():
                for key, val in self.chkpoint['current'].iteritems():
                    self.f["{}/checkpoint/{}".format(dim, key)].create_dataset("1", shape=self.N, dtype=val.dtype)
                for key, val in self.chkpoint['previous'].iteritems():
                    self.f["{}/checkpoint/{}".format(dim, key)].create_dataset("0", shape=self.N, dtype=val.dtype)

            s = self.FFT.real_local_slice()
            # Get new values
            if dim == "2D":
                for key, val in self.chkpoint['current'].iteritems():
                    if len(s) == len(val.shape):
                        self.f["2D/checkpoint/{}/1".format(key)][s] = val
                    else:
                        self.f["2D/checkpoint/{}/1".format(key)][:, s[0], s[1]] = val
                for key, val in self.chkpoint['previous'].iteritems():
                    if len(s) == len(val.shape):
                        self.f["2D/checkpoint/{}/0".format(key)][s] = val
                    else:
                        self.f["2D/checkpoint/{}/0".format(key)][:, s[0], s[1]] = val

            else:
                for key, val in self.chkpoint['current'].iteritems():
                    if len(s) == len(val.shape):
                        self.f["3D/checkpoint/{}/1".format(key)][s] = val
                    else:
                        self.f["3D/checkpoint/{}/1".format(key)][:, s[0], s[1], s[2]] = val
                for key, val in self.chkpoint['previous'].iteritems():
                    if len(s) == len(val.shape):
                        self.f["3D/checkpoint/{}/0".format(key)][s] = val
                    else:
                        self.f["3D/checkpoint/{}/0".format(key)][:, s[0], s[1], s[2]] = val
            self.f.close()

        def write(self, params):
            if self.f is None: 
                self.init_h5file(params) 
            else:
                self.f = h5py.File(self.fname, driver="mpio", comm=self.FFT.comm)

            N = self.N
            dim = str(self.dim)+"D"
            s = self.FFT.real_local_slice()
            if params.tstep % params.write_result == 0:                
                for comp, val in self.components.iteritems():
                    self.f[dim+"/"+comp].create_dataset(str(params.tstep), shape=N, dtype=self.dtype)
                    self.f[dim+"/%s/%d"%(comp, params.tstep)][s] = val

            # Write slices
            if params.has_key('write_yz_slice'):
                if params.tstep % params.write_yz_slice[1] == 0:
                    i = params.write_yz_slice[0]
                    for comp in self.components:
                        self.f["2D/yz/"+comp].create_dataset(str(params.tstep), shape=(N[1], N[2]), dtype=self.dtype)

                    sx = s[0]
                    if i >= sx.start and i < sx.stop:
                        for comp, val in self.components.iteritems():
                            self.f["2D/yz/%s/%d"%(comp, params.tstep)][s[1], s[2]] = val[i-sx.start]

                if params.tstep % params.write_xz_slice[1] == 0:
                    j = params.write_xz_slice[0]                
                    for comp in self.components:
                        self.f["2D/xz/"+comp].create_dataset(str(params.tstep), shape=(N[0], N[2]), dtype=self.dtype)

                    if params.decomposition == 'slab':
                        for comp, val in self.components.iteritems():
                            self.f["2D/xz/%s/%d"%(comp, params.tstep)][s[0], s[2]] = val[:, j, :]

                    elif params.decomposition == 'pencil':
                        sy = s[1]
                        if j >= sy.start and j < sy.stop:
                            for comp, val in self.components.iteritems():
                                self.f["2D/xz/%s/%d"%(comp, params.tstep)][s[0], s[2]] = val[:, j-sy.start, :]

                if params.tstep % params.write_xy_slice[1] == 0:
                    k = params.write_xy_slice[0]                
                    for comp in self.components:
                        self.f["2D/xy/"+comp].create_dataset(str(params.tstep), shape=(N[0], N[1]), dtype=self.dtype)
                    
                    for comp, val in self.components.iteritems():
                        self.f["2D/xy/%s/%d"%(comp, params.tstep)][s[0], s[1]] = val[:, :, k]

            self.f.close()

        def update_components(self, **kw):
            pass

        def close(self):
            if self.f: self.f.close()

except:
    class HDF5Writer(object):
        def __init__(self, FFT, dtype, comps, filename="U.h5", mesh={}):
            if FFT.comm.Get_rank() == 0:
                print Warning("Need to install h5py to allow storing results")

        def write(self, params):
            pass

        def check_if_write(self, params):
            return False

        def checkpoint(self, *args):
            pass

        def close(self):
            del self
