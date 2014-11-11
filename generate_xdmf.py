import h5py
from numpy import pi

xdmffile = """<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.1">
  <Domain>
    <Grid Name="Structured Grid">
"""
attribute = """
      <Attribute Name="{0}" Center="Node">
        <DataItem Format="HDF" NumberType="Float" Precision="4" Dimensions="{1} {1} {1}">
         {2}:/{3}/{4}
        </DataItem>
      </Attribute>"""

def generate_xdmf(h5filename):
    global xdmffile
    f = h5py.File(h5filename, "r")
    N = f.attrs["N"]
    xdmffile += """      <Geometry Type="ORIGIN_DXDYDZ">
        <DataItem DataType="UInt" Dimensions="3" Format="XML" Precision="4">0 0 0</DataItem>
        <DataItem DataType="Float" Dimensions="3" Format="XML" Precision="4">{0} {0} {0}</DataItem>
      </Geometry>""".format(2*pi)

    xdmffile += """
      <Topology Dimensions="{0} {0} {0}" Type="3DCoRectMesh"/>""".format(N)
    
    for group in f:
        for tstep, dset in f[group].iteritems():
            xdmffile += attribute.format(group, N, h5filename, group, tstep)

    xdmffile += """  
    </Grid>
  </Domain>
</Xdmf>"""
    xf = open(h5filename[:-2]+"xdmf", "w")
    xf.write(xdmffile)
    xf.close()
    
if __name__ == "__main__":
    import sys
    generate_xdmf(sys.argv[-1])
    