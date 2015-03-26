__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-19"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""Wrap call to hdf5 to allow running without installing h5py
"""
__all__ = ['HDF5Writer']
from numpy import pi, float32

xdmffile = """<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.1">
  <Domain>
    <Grid Name="Structured Grid" GridType="Collection" CollectionType="Temporal">
"""
timeattr = """      <Time TimeType="List"><DataItem Format="XML" Dimensions="{1}"> {0} </DataItem></Time>"""
attribute3D = """
        <Attribute Name="{0}" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{5}" Dimensions="{1} {1} {1}">
            {2}:/3D/{3}/{4}
          </DataItem>
        </Attribute>"""

attribute2D = """
        <Attribute Name="{0}_2D" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{5}" Dimensions="{1} {1}">
            {2}:/2D/{3}/{4}
          </DataItem>
        </Attribute>"""

try:
    import h5py
    import copy
    class HDF5Writer(object):
    
        def __init__(self, comm, dt, N, params, dtype, filename="U.h5"):
            self.f = h5py.File(filename, "w", driver="mpio", comm=comm)
            self.comm = comm
            self.components = components = ["U", "V", "W", "P"]
            if "eta" in params: components += ["Bx", "By", "Bz"]
            self.f.create_group("3D")
            self.f.create_group("2D")
            for c in components:
                self.f["3D"].create_group(c)
                self.f["2D"].create_group(c)
            self.f.attrs.create("dt", dt)
            self.f.attrs.create("N", N)    
            self.fname = filename
            self.params = params
            self.f["2D"].attrs.create("i", params['write_yz_slice'][0])
            self.dtype = dtype
            
        def write(self, U, P, tstep):
            
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
                self.f["3D/U/%d"%tstep][x1, :, x2] = U[0]
                self.f["3D/V/%d"%tstep][x1, :, x2] = U[1]
                self.f["3D/W/%d"%tstep][x1, :, x2] = U[2]
                self.f["3D/P/%d"%tstep][x1, :, x2] = P
                if len(self.components) == 7:
                    self.f["3D/Bx/%d"%tstep][x1, :, x2] = U[3]
                    self.f["3D/By/%d"%tstep][x1, :, x2] = U[4]
                    self.f["3D/Bz/%d"%tstep][x1, :, x2] = U[5]
                    
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
                if i >= x1.start and i < x1.stop:
                    self.f["2D/U/%d"%tstep][:, x2] = U[0, i-x1.start]
                    self.f["2D/V/%d"%tstep][:, x2] = U[1, i-x1.start]
                    self.f["2D/W/%d"%tstep][:, x2] = U[2, i-x1.start]
                    self.f["2D/P/%d"%tstep][:, x2] = P[i-x1.start]
                    if len(self.components) == 7:
                        self.f["2D/Bx/%d"%tstep][:, x2] = U[3, i-x1.start]
                        self.f["2D/By/%d"%tstep][:, x2] = U[4, i-x1.start]
                        self.f["2D/Bz/%d"%tstep][:, x2] = U[5, i-x1.start]
                            
        def close(self):
            self.f.close()
            
        def generate_xdmf(self):
            if self.comm.Get_rank() == 0:
                xf3d = copy.copy(xdmffile)
                timesteps = self.f["3D/U"].keys()
                N = self.f.attrs["N"]
                tt = ""
                for i in timesteps:
                    tt += "%s " %i
                
                xf3d += timeattr.format(tt, len(timesteps))
            
                for tstep in timesteps:
                    xf3d += """
      <Grid GridType="Uniform">
        <Geometry Type="ORIGIN_DXDYDZ">
          <DataItem DataType="UInt" Dimensions="3" Format="XML" Precision="4">0 0 0</DataItem>
          <DataItem DataType="Float" Dimensions="3" Format="XML" Precision="4">{0} {0} {0}</DataItem>
        </Geometry>""".format(2*pi)

                    xf3d += """
        <Topology Dimensions="{0} {0} {0}" Type="3DCoRectMesh"/>""".format(N)
                    prec = 4 if self.dtype is float32 else 8
                    for comp in self.f["3D"]:
                        xf3d += attribute3D.format(comp, N, self.fname, comp, tstep, prec)
                    xf3d += """  
      </Grid>
"""
                xf3d += """    
    </Grid>
  </Domain>
</Xdmf>  
"""
                xf = open(self.fname[:-2]+"xdmf", "w")
                xf.write(xf3d)
                xf.close()
                
                if len(self.f["2D/U"].keys()) == 0:
                    return
                
                xf2d = copy.copy(xdmffile)
                timesteps = self.f["2D/U"].keys()
                N = self.f.attrs["N"]
                tt = ""
                for i in timesteps:
                    tt += "%s " %i
                
                xf2d += timeattr.format(tt, len(timesteps))
            
                for tstep in timesteps:
                    xf2d += """
      <Grid GridType="Uniform">
        <Geometry Type="ORIGIN_DXDY">
          <DataItem DataType="UInt" Dimensions="2" Format="XML" Precision="4">0 0</DataItem>
          <DataItem DataType="Float" Dimensions="2" Format="XML" Precision="4">{0} {0}</DataItem>
        </Geometry>""".format(2*pi)

                    xf2d += """
        <Topology Dimensions="{0} {0}" Type="2DCoRectMesh"/>""".format(N)
                    prec = 4 if self.dtype is float32 else 8
                    for comp in self.f["2D"]:
                        xf2d += attribute2D.format(comp, N, self.fname, comp, tstep, prec)
                    xf2d += """  
      </Grid>
"""
                xf2d += """  
    </Grid>
  </Domain>
</Xdmf>"""
                xf = open(self.fname[:-3]+"_2D.xdmf", "w")
                xf.write(xf2d)
                xf.close()

except:
    class HDF5Writer(object):
        def __init__(self, comm, dt, N, params, filename="U.h5"):
            if comm.Get_rank() == 0:
                print Warning("Need to install h5py to allow storing results")
        
        def write(self, U, P, tstep):
            pass
        
        def close(self):
            del self
            
        def generate_xdmf(self):
            pass
