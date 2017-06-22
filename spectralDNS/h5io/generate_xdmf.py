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
          <DataItem Format="HDF" NumberType="Float" Precision="{7}" Dimensions="{1} {2} {3}">
            {4}:/3D/{5}/{6}
          </DataItem>
        </Attribute>"""

attribute2D = """
        <Attribute Name="{0}_2D" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{6}" Dimensions="{1} {2}">
            {3}:/2D/{4}/{5}
          </DataItem>
        </Attribute>"""

attribute2Dslice = """
        <Attribute Name="{0}_{7}" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{6}" Dimensions="{1} {2}">
            {3}:/2D/{7}/{4}/{5}
          </DataItem>
        </Attribute>"""

isotropic = """
        <Geometry Type="ORIGIN_DXDYDZ">
          <DataItem DataType="UInt" Dimensions="3" Format="XML" Precision="4">0 0 0</DataItem>
          <DataItem DataType="Float" Dimensions="3" Format="XML" Precision="4">{0} {1} {2}</DataItem>
        </Geometry>"""

channel =  """
        <Geometry Type="VXVYVZ">
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{3}">
           {4}:/mesh/z
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{2}">
           {4}:/mesh/y
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{1}">
           {4}:/mesh/x
          </DataItem>
        </Geometry>"""

isotropic2D = """
        <Geometry Type="ORIGIN_DXDY">
          <DataItem DataType="UInt" Dimensions="2" Format="XML" Precision="4">0 0</DataItem>
          <DataItem DataType="Float" Dimensions="2" Format="XML" Precision="4">{0} {1}</DataItem>
        </Geometry>"""

channel2D =  """
        <Geometry Type="VXVY">
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{1}">
           {3}:/mesh/{4}
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{2}">
           {3}:/mesh/{5}
          </DataItem>
        </Geometry>"""

def generate_xdmf(h5filename):
    f = h5py.File(h5filename)
    comps = list(f["3D"].keys())
    for i in ("checkpoint", "oldcheckpoint", "mesh"):
        try:
            popped = comps.remove(i)
        except:
            pass

    N = f.attrs["N"]
    L = f.attrs["L"]
    if len(f.attrs.get('N')) == 3:
        xf3d = copy.copy(xdmffile)
        timesteps = list(f["/".join(("3D", comps[0]))].keys())
        tt = ""
        for i in timesteps:
            tt += "%s " %i

        xf3d += timeattr.format(tt, len(timesteps))

        #from IPython import embed; embed()
        dtype = f["/".join(("3D", comps[0]))].get(timesteps[0]).dtype
        prec = 4 if dtype is float32 else 8

        for tstep in timesteps:
            xf3d += """
      <Grid GridType="Uniform">"""

            if "mesh" in f["/3D"].keys():
                xf3d += channel.format(prec, N[0], N[1], N[2], h5filename)
                xf3d += """
        <Topology Dimensions="{0} {1} {2}" Type="3DRectMesh"/>""".format(*N)
            else:
                xf3d += isotropic.format(L[0]/N[0], L[1]/N[1], L[2]/N[2])
                xf3d += """
        <Topology Dimensions="{0} {1} {2}" Type="3DCoRectMesh"/>""".format(*N)

            prec = 4 if dtype == float32 else 8
            for comp in comps:
                xf3d += attribute3D.format(comp, N[0], N[1], N[2], h5filename, comp, tstep, prec)
            xf3d += """
      </Grid>
"""
        xf3d += """
    </Grid>
  </Domain>
</Xdmf>
"""
        #f.attrs.create("xdmf_3d", xf3d)
        xf = open(h5filename[:-2]+"xdmf", "w")
        xf.write(xf3d)
        xf.close()

    # Return if no 2D data
    names = []
    f["/2D"].visit(names.append)
    if not any([isinstance(f["/2D/"+x], h5py.Dataset) for x in names]):
        return

    comps = f["/2D"].keys()
    [comps.remove(x) for x in ('xy', 'xz', 'yz')]

    if len(f["/".join(("2D", comps[0]))]) > 0:
        xf2d = copy.copy(xdmffile)
        timesteps = f["/".join(("2D", comps[0]))].keys()
        dtype = f["/".join(("2D", comps[0]))].values()[0].dtype
        prec = 4 if dtype is float32 else 8

        tt = ""
        for i in timesteps:
            tt += "%s " %i

        xf2d += timeattr.format(tt, len(timesteps))

        for tstep in timesteps:
            xf2d += """
      <Grid GridType="Uniform">"""

            if "mesh" in f.keys():
                xf2d += channel2D.format(prec, N[0], N[1], h5filename, 'x', 'y')
                xf2d += """
        <Topology Dimensions="{0} {1}" Type="2DRectMesh"/>""".format(N[0], N[1])
            else:
                xf2d += isotropic2D.format(L[0]/N[0], L[1]/N[1])
                xf2d += """
        <Topology Dimensions="{0} {1}" Type="2DCoRectMesh"/>""".format(N[0], N[1])

            for comp in comps:
                xf2d += attribute2D.format(comp, N[0], N[1], h5filename, comp, tstep, prec)
            xf2d += """
      </Grid>
"""
        xf2d += """
    </Grid>
  </Domain>
</Xdmf>"""
        xf2 = open(h5filename[:-3]+"_2D.xdmf", "w")
        xf2.write(xf2d)
        xf2.close()

    if len(f["/".join(("2D/yz", comps[0]))]) > 0:
        xf2d = copy.copy(xdmffile)
        timesteps = f["/".join(("2D/yz", comps[0]))].keys()
        dtype = f["/".join(("2D/yz", comps[0]))].values()[0].dtype
        prec = 4 if dtype is float32 else 8
        tt = ""
        for i in timesteps:
            tt += "%s " %i

        xf2d += timeattr.format(tt, len(timesteps))

        for tstep in timesteps:
            xf2d += """
      <Grid GridType="Uniform">"""

            if "mesh" in f.keys():
                xf2d += channel2D.format(prec, N[1], N[2], h5filename, 'y', 'z')
                xf2d += """
        <Topology Dimensions="{0} {1}" Type="2DRectMesh"/>""".format(N[1], N[2])
            else:
                xf2d += isotropic2D.format(L[1]/N[1], L[2]/N[2])
                xf2d += """
        <Topology Dimensions="{0} {1}" Type="2DCoRectMesh"/>""".format(N[1], N[2])

            if len(f["/".join(("2D/yz", comps[0]))]) > 0:
                for comp in f["2D/yz"]:
                    xf2d += attribute2Dslice.format(comp, N[1], N[2], h5filename, comp, tstep, prec, 'yz')
            xf2d += """
      </Grid>
"""
        xf2d += """
    </Grid>
  </Domain>
</Xdmf>"""
        xf2 = open(h5filename[:-3]+"_yz.xdmf", "w")
        xf2.write(xf2d)
        xf2.close()

    if len(f["/".join(("2D/xz", comps[0]))]) > 0:
        xf2d = copy.copy(xdmffile)
        timesteps = f["/".join(("2D/xz", comps[0]))].keys()
        dtype = f["/".join(("2D/xz", comps[0]))].values()[0].dtype
        prec = 4 if dtype is float32 else 8
        tt = ""
        for i in timesteps:
            tt += "%s " %i

        xf2d += timeattr.format(tt, len(timesteps))

        for tstep in timesteps:
            xf2d += """
      <Grid GridType="Uniform">"""

            if "mesh" in f.keys():
                xf2d += channel2D.format(prec, N[0], N[2], h5filename, 'x', 'z')
                xf2d += """
        <Topology Dimensions="{0} {1}" Type="2DRectMesh"/>""".format(N[0], N[2])
            else:
                xf2d += isotropic2D.format(L[0]/N[0], L[2]/N[2])
                xf2d += """
        <Topology Dimensions="{0} {1}" Type="2DCoRectMesh"/>""".format(N[0], N[2])

            if len(f["/".join(("2D/xz", comps[0]))]) > 0:
                for comp in f["2D/xz"]:
                    xf2d += attribute2Dslice.format(comp, N[0], N[2], h5filename, comp, tstep, prec, 'xz')
            xf2d += """
      </Grid>
"""
        xf2d += """
    </Grid>
  </Domain>
</Xdmf>"""
        xf2 = open(h5filename[:-3]+"_xz.xdmf", "w")
        xf2.write(xf2d)
        xf2.close()

    if len(f["/".join(("2D/xy", comps[0]))]) > 0:
        xf2d = copy.copy(xdmffile)
        timesteps = f["/".join(("2D/xy", comps[0]))].keys()
        dtype = f["/".join(("2D/xy", comps[0]))].values()[0].dtype
        prec = 4 if dtype is float32 else 8
        tt = ""
        for i in timesteps:
            tt += "%s " %i

        xf2d += timeattr.format(tt, len(timesteps))

        for tstep in timesteps:
            xf2d += """
      <Grid GridType="Uniform">"""

            if "mesh" in f.keys():
                xf2d += channel2D.format(prec, N[0], N[1], h5filename, 'x', 'y')
                xf2d += """
        <Topology Dimensions="{0} {1}" Type="2DRectMesh"/>""".format(N[0], N[1])
            else:
                xf2d += isotropic2D.format(L[0]/N[0], L[1]/N[1])
                xf2d += """
        <Topology Dimensions="{0} {1}" Type="2DCoRectMesh"/>""".format(N[0], N[1])

            if len(f["/".join(("2D/xy", comps[0]))]) > 0:
                for comp in f["2D/xy"]:
                    xf2d += attribute2Dslice.format(comp, N[0], N[1], h5filename, comp, tstep, prec, 'xy')
            xf2d += """
      </Grid>
"""
        xf2d += """
    </Grid>
  </Domain>
</Xdmf>"""
        xf2 = open(h5filename[:-3]+"_xy."+"xdmf", "w")
        xf2.write(xf2d)
        xf2.close()

if __name__ == "__main__":
    import sys
    generate_xdmf(sys.argv[-1])
