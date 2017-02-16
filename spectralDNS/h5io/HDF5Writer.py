__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-19"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""Wrap call to hdf5 to allow running without installing h5py
"""
from numpy import all
from numpy.linalg import norm
import warnings
import six

__all__ = ['HDF5Writer']

try:
    import h5py
    class HDF5Writer(object):

        def __init__(self, comps, chkpoint={}, filename="U.h5", mesh={}):
            self.components = comps
            self.chkpoint = chkpoint
            self.fname = filename
            self.f = None
            self.dim = len(comps[list(comps.keys())[0]].shape)
            self.mesh = mesh

        def _init_h5file(self, params, FFT, **kw):
            comm = FFT.comm
            self.f = h5py.File(self.fname, "w", driver="mpio", comm=comm)
            self.f.create_group("3D")
            self.f.create_group("2D")
            if len(self.mesh) > 0:
                self.f.create_group("mesh")
                for key, val in six.iteritems(self.mesh):
                    self.f["/mesh/"].create_dataset(key, shape=(len(val),), dtype=val.dtype)
                    self.f["/mesh/"+key][:] = val

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
            self.f.attrs.create("N", params.N)
            self.f.attrs.create("L", params.L)
            self.f.attrs.create("Sk", 0.0)
            if 'write_yz_slice' in params:
                self.f["2D/yz"].attrs.create("i", params.write_yz_slice[0])
                self.f["2D/xy"].attrs.create("j", params.write_xy_slice[0])
                self.f["2D/xz"].attrs.create("k", params.write_xz_slice[0])

        def update(self, params, **kw):
            if (self.check_if_write(params) or
                  params.tstep % params.checkpoint == 0):
                self.update_components(**kw)

            if self.check_if_write(params):
                self._write(params, **kw)

            if params.tstep % params.checkpoint == 0:
                self._checkpoint(params, **kw)

        def check_if_write(self, params):
            if params.tstep % params.write_result == 0:
                return True

            if 'write_xy_slice' in params:
                if (params.tstep % params.write_xy_slice[1] == 0 or
                    params.tstep % params.write_yz_slice[1] == 0 or
                    params.tstep % params.write_xz_slice[1] == 0):
                    return True
            return False

        def _checkpoint(self, params, **kw):
            FFT = kw.pop('FFT', kw.get('FST'))
            if self.f is None:
                self._init_h5file(params, FFT, **kw)
            else:
                self.f = h5py.File(self.fname, driver="mpio", comm=FFT.comm)
                if not all(self.f.attrs['N'] == params.N):
                    self._init_h5file(params, FFT, **kw)

            dim = str(self.dim)+"D"
            keys = list(self.chkpoint['current'].keys())
            # Create datasets first time around
            create_new_dataset = False
            for key in keys:
                kk = "{}/checkpoint/{}".format(dim, key)
                if not "1" in list(self.f[kk].keys()):
                    create_new_dataset = True
            FFT.comm.barrier()
            if create_new_dataset:
                for key, val in six.iteritems(self.chkpoint['current']):
                    shape = params.N if len(val.shape) == self.dim else (val.shape[0],)+tuple(params.N)
                    self.f["{}/checkpoint/{}".format(dim, key)].create_dataset("1", shape=shape, dtype=val.dtype)
                for key, val in six.iteritems(self.chkpoint['previous']):
                    shape = params.N if len(val.shape) == self.dim else (val.shape[0],)+tuple(params.N)
                    self.f["{}/checkpoint/{}".format(dim, key)].create_dataset("0", shape=shape, dtype=val.dtype)

            s = FFT.real_local_slice()
            # Get new values
            if dim == "2D":
                for key, val in six.iteritems(self.chkpoint['current']):
                    if len(s) == len(val.shape):
                        self.f["2D/checkpoint/{}/1".format(key)][s] = val
                    else:
                        self.f["2D/checkpoint/{}/1".format(key)][:, s[0], s[1]] = val
                for key, val in six.iteritems(self.chkpoint['previous']):
                    if len(s) == len(val.shape):
                        self.f["2D/checkpoint/{}/0".format(key)][s] = val
                    else:
                        self.f["2D/checkpoint/{}/0".format(key)][:, s[0], s[1]] = val

            else:
                for key, val in six.iteritems(self.chkpoint['current']):
                    if len(s) == len(val.shape):
                        self.f["3D/checkpoint/{}/1".format(key)][s] = val
                    else:
                        ss = (slice(None),)+s
                        self.f["3D/checkpoint/{}/1".format(key)][ss] = val

                for key, val in six.iteritems(self.chkpoint['previous']):
                    if len(s) == len(val.shape):
                        self.f["3D/checkpoint/{}/0".format(key)][s] = val
                    else:
                        ss = (slice(None),)+s
                        self.f["3D/checkpoint/{}/0".format(key)][ss] = val

            # For channel solver with dynamic pressure
            if 'Sk' in kw:
                z0 = kw['Sk'][1,0,0,0].real
                z0 = FFT.comm.bcast(z0)
                self.f.attrs["Sk"] = z0
            self.f.close()

        def _write(self, params, **kw):
            FFT = kw.pop('FFT', kw.get('FST'))
            if self.f is None:
                self._init_h5file(params, FFT, **kw)
            else:
                self.f = h5py.File(self.fname, driver="mpio", comm=FFT.comm)
                if not all(self.f.attrs['N'] == params.N):
                    self._init_h5file(params, FFT, **kw)

            dim = str(self.dim)+"D"
            N = params.N
            s = FFT.real_local_slice()
            if params.tstep % params.write_result == 0:
                for comp, val in six.iteritems(self.components):
                    self.f[dim+"/"+comp].create_dataset(str(params.tstep), shape=N, dtype=val.dtype)
                    self.f[dim+"/%s/%d"%(comp, params.tstep)][s] = val

            # Write slices
            if 'write_yz_slice' in params:
                if params.tstep % params.write_yz_slice[1] == 0:
                    i = params.write_yz_slice[0]
                    for comp in self.components:
                        self.f["2D/yz/"+comp].create_dataset(str(params.tstep), shape=(N[1], N[2]), dtype=FFT.float)

                    sx = s[0]
                    if i >= sx.start and i < sx.stop:
                        for comp, val in six.iteritems(self.components):
                            self.f["2D/yz/%s/%d"%(comp, params.tstep)][s[1], s[2]] = val[i-sx.start]

                if params.tstep % params.write_xz_slice[1] == 0:
                    j = params.write_xz_slice[0]
                    for comp in self.components:
                        self.f["2D/xz/"+comp].create_dataset(str(params.tstep), shape=(N[0], N[2]), dtype=FFT.float)

                    if params.decomposition == 'slab':
                        for comp, val in six.iteritems(self.components):
                            self.f["2D/xz/%s/%d"%(comp, params.tstep)][s[0], s[2]] = val[:, j, :]

                    elif params.decomposition == 'pencil':
                        sy = s[1]
                        if j >= sy.start and j < sy.stop:
                            for comp, val in six.iteritems(self.components):
                                self.f["2D/xz/%s/%d"%(comp, params.tstep)][s[0], s[2]] = val[:, j-sy.start, :]

                if params.tstep % params.write_xy_slice[1] == 0:
                    k = params.write_xy_slice[0]
                    for comp in self.components:
                        self.f["2D/xy/"+comp].create_dataset(str(params.tstep), shape=(N[0], N[1]), dtype=FFT.float)

                    for comp, val in six.iteritems(self.components):
                        self.f["2D/xy/%s/%d"%(comp, params.tstep)][s[0], s[1]] = val[:, :, k]

            # For channel solver with dynamic pressure
            if 'Sk' in kw:
                z0 = kw['Sk'][1,0,0,0].real
                z0 = FFT.comm.bcast(z0)
                self.f.attrs["Sk"] = z0

            self.f.close()

        def update_components(self, **kw):
            pass

        def close(self):
            if self.f: self.f.close()

except:
    class HDF5Writer(object):
        def __init__(self, comps, chkpoint={}, filename="U.h5", mesh={}):
            warnings.warn("Need to install h5py to allow storing results")

        def check_if_write(self, params):
            return False

        def _checkpoint(self, params, **context):
            pass

        def _write(self, params, **context):
            pass

        def close(self):
            del self
