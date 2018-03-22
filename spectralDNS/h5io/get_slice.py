#pylint: disable=bare-except,len-as-condition

import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD

def get_slice(h5filename, tsteps, newfilename=None):
    f = h5py.File(h5filename)
    N = f.attrs["N"]
    comps = f["2D"].keys()
    try:
        dtype = f["/".join(("2D", comps[0]))].values()[0].dtype
    except:
        dtype = f["/".join(("3D", comps[0]))].values()[0].dtype

    assert isinstance(tsteps, tuple)
    if newfilename:
        assert newfilename[-3:] == ".h5"
        fnew = h5py.File(newfilename, "w", driver="mpio", comm=comm)
        fnew.create_group("2D")
        fnew.create_group("3D")
        fnew.attrs.create("N", N)
        for comp in comps:
            fnew["3D"].create_group(comp)
            fnew["2D"].create_group(comp)
        for tstep in tsteps:
            tstep = str(tstep)
            if len(f["/".join(("3D", comps[0]))]) > 0:
                for comp in comps:
                    fnew["3D/"+comp].create_dataset(tstep, shape=(N, N, N), dtype=dtype)
                    fnew["3D/"+comp+"/"+tstep][:] = f["3D/"+comp+"/"+tstep]

            if len(f["/".join(("2D", comps[0]))]) > 0:
                for comp in comps:
                    fnew["2D/"+comp].create_dataset(tstep, shape=(N, N), dtype=dtype)
                    fnew["2D/"+comp+"/"+tstep][:] = f["2D/"+comp+"/"+tstep]
        fnew.close()

    else:

        for tstep in tsteps:
            tstep = str(tstep)
            fnew = h5py.File(h5filename[:-3]+"_"+tstep+".h5", "w", driver="mpio", comm=comm)
            fnew.create_group("2D")
            fnew.create_group("3D")
            fnew.attrs.create("N", N)
            for comp in comps:
                fnew["3D"].create_group(comp)
                fnew["2D"].create_group(comp)
            if len(f["/".join(("3D", comps[0]))]) > 0:
                for comp in comps:
                    fnew["3D/"+comp].create_dataset(tstep, shape=(N, N, N), dtype=dtype)
                    fnew["3D/"+comp+"/"+tstep][:] = f["3D/"+comp+"/"+tstep]

            if len(f["/".join(("2D", comps[0]))]) > 0:
                for comp in comps:
                    fnew["2D/"+comp].create_dataset(tstep, shape=(N, N), dtype=dtype)
                    fnew["2D/"+comp+"/"+tstep][:] = f["2D/"+comp+"/"+tstep]
            fnew.close()

    f.close()

if __name__ == "__main__":
    import sys
    assert len(sys.argv) in (3, 4)
    if len(sys.argv) == 4:
        get_slice(sys.argv[-3], sys.argv[-2], sys.argv[-1])
    else:
        get_slice(sys.argv[-2], sys.argv[-1])
