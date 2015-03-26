import h5py
import copy
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

def generate_xdmf(h5filename):
    f = h5py.File(h5filename)
    xf3d = copy.copy(xdmffile)
    timesteps = f["3D/U"].keys()
    N = f.attrs["N"]
    tt = ""
    for i in timesteps:
        tt += "%s " %i
    
    xf3d += timeattr.format(tt, len(timesteps))
    
    dtype = f["3D/U"].values()[0].dtype

    for tstep in timesteps:
        xf3d += """
      <Grid GridType="Uniform">
        <Geometry Type="ORIGIN_DXDYDZ">
          <DataItem DataType="UInt" Dimensions="3" Format="XML" Precision="4">0 0 0</DataItem>
          <DataItem DataType="Float" Dimensions="3" Format="XML" Precision="4">{0} {0} {0}</DataItem>
        </Geometry>""".format(2*pi)

        xf3d += """
        <Topology Dimensions="{0} {0} {0}" Type="3DCoRectMesh"/>""".format(N)
        prec = 4 if dtype == float32 else 8
        for comp in f["3D"]:
            xf3d += attribute3D.format(comp, N, h5filename, comp, tstep, prec)
        xf3d += """  
      </Grid>
"""
    xf3d += """    
    </Grid>
  </Domain>
</Xdmf>  
"""
    f.attrs.create("xdmf_3d", xf3d)
    xf = open(h5filename[:-2]+"xdmf", "w")
    xf.write(xf3d)
    xf.close()            
    if len(f["2D/U"].keys()) == 0:
        return
    
    xf2d = copy.copy(xdmffile)
    timesteps = f["2D/U"].keys()
    N = f.attrs["N"]
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
        prec = 4 if dtype is float32 else 8
        for comp in f["2D"]:
            xf2d += attribute2D.format(comp, N, h5filename, comp, tstep, prec)
        xf2d += """  
      </Grid>
"""
    xf2d += """  
    </Grid>
  </Domain>
</Xdmf>"""
    f.attrs.create("xdmf_2d", xf2d)
    xf2 = open(h5filename[:-2]+"xdmf_2D", "w")
    xf2.write(xf2d)
    xf2.close()        
    
if __name__ == "__main__":
    import sys    
    generate_xdmf(sys.argv[-1])
