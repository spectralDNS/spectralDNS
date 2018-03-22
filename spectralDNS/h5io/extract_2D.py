import os
import h5py

def extract_2D(h5filename):
    os.system("h5copy -v -i {0}.h5 -s '/2D' -o {0}_2D.h5 -d '/2D'".format(h5filename[:-3]))
    f0 = h5py.File(h5filename, "r")
    f1 = h5py.File(h5filename[:-3]+"_2D.h5")
    f1.create_group("3D")
    f1["3D"].create_group("U")
    f1["3D"].create_group("P")
    f1.attrs.create("dt", f0.attrs["dt"])
    f1.attrs.create("N", f0.attrs["N"])
    f1.attrs.create("L", f0.attrs["L"])
    f1.close()
    f0.close()

if __name__ == "__main__":
    import sys
    extract_2D(sys.argv[-1])
