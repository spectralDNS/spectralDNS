__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-19"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""Wrap call to hdf5 to allow running without installing h5py
"""
__all__ = ['HDF5Writer', 'generate_xdmf']
from numpy import pi

xdmffile = """<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.1">
  <Domain>
    <Grid Name="Structured Grid">
"""
attribute3D = """
      <Attribute Name="{0}" Center="Node">
        <DataItem Format="HDF" NumberType="Float" Precision="4" Dimensions="{1} {1} {1}">
         {2}:/3D/{3}/{4}
        </DataItem>
      </Attribute>"""

attribute2D = """
      <Attribute Name="{0}" Center="Node">
        <DataItem Format="HDF" NumberType="Float" Precision="4" Dimensions="1 {1} {1}">
         {2}:/2D/{3}/{4}
        </DataItem>
      </Attribute>"""

try:
    import h5py
    class HDF5Writer(object):
    
        def __init__(self, comm, dt, N, params, filename="U.h5"):
            self.f = h5py.File("U.h5", "w", driver="mpio", comm=comm)
            self.comm = comm
            self.f.create_group("3D")
            self.f["3D"].create_group("U")
            self.f["3D"].create_group("V")
            self.f["3D"].create_group("W")
            self.f["3D"].create_group("P")
            self.f.create_group("2D")
            self.f["2D"].create_group("U")
            self.f["2D"].create_group("V")
            self.f["2D"].create_group("W")
            self.f["2D"].create_group("P")
            self.f.attrs.create("dt", dt)
            self.f.attrs.create("N", N)    
            self.fname = filename
            self.params = params
            self.f["2D"].attrs.create("i", params['write_yz_slice'][0])
            
        def write(self, U, P, tstep):
            
            if tstep % self.params['write_result'] == 0:
                rank = self.comm.Get_rank()
                N = self.f.attrs["N"]
                assert N == P.shape[-1]
                Np =  N / self.comm.Get_size()
                
                for comp in ["U", "V", "W", "P"]:
                    self.f["3D/"+comp].create_dataset(str(tstep), shape=(N, N, N), dtype="float")
                                    
                self.f["3D/U/%d"%tstep][rank*Np:(rank+1)*Np] = U[0]
                self.f["3D/V/%d"%tstep][rank*Np:(rank+1)*Np] = U[1]
                self.f["3D/W/%d"%tstep][rank*Np:(rank+1)*Np] = U[2]
                self.f["3D/P/%d"%tstep][rank*Np:(rank+1)*Np] = P
                
            if tstep % self.params['write_yz_slice'][1] == 0:
                i = self.params['write_yz_slice'][0]
                rank = self.comm.Get_rank()
                N = self.f.attrs["N"]
                assert N == P.shape[-1]
                Np =  N / self.comm.Get_size()     
                for comp in ["U", "V", "W", "P"]:
                    self.f["2D/"+comp].create_dataset(str(tstep), shape=(1, N, N), dtype="float")
                                    
                if i >= rank*Np and i < (rank+1)*Np:
                    self.f["2D/U/%d"%tstep][:] = U[0, i-rank*Np]
                    self.f["2D/V/%d"%tstep][:] = U[1, i-rank*Np]
                    self.f["2D/W/%d"%tstep][:] = U[2, i-rank*Np]
                    self.f["2D/P/%d"%tstep][:] = P[i-rank*Np]
                            
        def close(self):
            self.f.close()
            
        def generate_xdmf(self):
            global xdmffile
            N = self.f.attrs["N"]
            xdmffile += """      <Geometry Type="ORIGIN_DXDYDZ">
        <DataItem DataType="UInt" Dimensions="3" Format="XML" Precision="4">0 0 0</DataItem>
        <DataItem DataType="Float" Dimensions="3" Format="XML" Precision="4">{0} {0} {0}</DataItem>
      </Geometry>""".format(2*pi)

            xdmffile += """
      <Topology Dimensions="{0} {0} {0}" Type="3DCoRectMesh"/>""".format(N)
    
            for comp in self.f["3D"]:
                for tstep, dset in self.f["3D/"+comp].iteritems():
                    xdmffile += attribute3D.format(comp, N, self.fname, comp, tstep)
                                                
            xdmffile += """  
    </Grid>"""
            xdmffile += """
    <Grid Name="Structured Grid 3D">
      <Geometry Type="ORIGIN_DXDYDZ">
        <DataItem DataType="UInt" Dimensions="3" Format="XML" Precision="4">{1} 0 0</DataItem>
        <DataItem DataType="Float" Dimensions="3" Format="XML" Precision="4">{1} {0} {0}</DataItem>
      </Geometry>""".format(2*pi, self.params['write_yz_slice'][0]*2*pi/self.f.attrs['N'])

            xdmffile += """
      <Topology Dimensions="1 {0} {0}" Type="3DCoRectMesh"/>""".format(N)

            for comp in self.f["2D"]:
                for tstep, dset in self.f["2D/"+comp].iteritems():
                    xdmffile += attribute2D.format(comp, N, self.fname, comp, tstep)
                        
            xdmffile += """  
    </Grid>
  </Domain>
</Xdmf>"""
            xf = open(self.fname[:-2]+"xdmf", "w")
            xf.write(xdmffile)
            xf.close()

except:
    class HDF5Writer(object):
        def __init__(self, comm, dt, N, filename="U.h5"):
            if comm.Get_rank() == 0:
                print Warning("Need to install h5py to allow storing results")
        
        def write(self):
            pass
        
        def close(self):
            del self
            
        def generate_xdmf(self):
            pass
