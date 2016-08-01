from mpiFFT4py.slab import R2C as FFT

### FIXME Need to move SlabShen_R2C in its own module, out of spectralDNS
from spectralDNS.mesh.channel import SlabShen_R2C as FST
from spectralDNS.shen.shentransform import ShenDirichletBasis, ShenBiharmonicBasis
from mpi4py import MPI
import numpy as np
import h5py

def refine(infile, mesh):
    """Refine 3D solution
    """
    assert mesh in ("channel", "triplyperiodic")
    comm = MPI.COMM_WORLD
    filename, ending = infile.split(".")
    fin = h5py.File(infile, driver="mpio", comm=comm)    
    fout = h5py.File(filename+"_refined.h5", "w", driver="mpio", comm=comm)
    N = fin.attrs["N"]
    N1 = N.copy()
    if mesh == "channel":
        N1[1:] *= 2
        FFT0 = FST(N, fin.attrs["L"], MPI, padsize=2)
        SB = ShenBiharmonicBasis("GL")
        ST = ShenDirichletBasis("GL")
        
    elif mesh == "triplyperiodic":
        N1 *= 2        
        FFT0 = FFT(N, fin.attrs["L"], MPI, "double", padsize=2)

    shape = (3, N1[0], N1[1], N1[2])
    fout.create_group("3D")
    fout["3D"].create_group("checkpoint")
    fout["3D/checkpoint"].create_group("U")
    fout.attrs.create("dt", fin.attrs["dt"])
    fout.attrs.create("N", N1) 
    fout.attrs.create("L", fin.attrs["L"])
    fout["3D/checkpoint/U"].create_dataset("0", shape=shape, dtype=FFT0.float)
    fout["3D/checkpoint/U"].create_dataset("1", shape=shape, dtype=FFT0.float)

    assert "checkpoint" in fin["3D"]
    rank = comm.Get_rank()
    U0 = np.empty((3,)+FFT0.real_shape(), dtype=FFT0.float)
    s = FFT0.real_local_slice()
    s1 = FFT0.real_local_slice(padsize=2)
    
    U0[:] = fin["3D/checkpoint/U/0"][:, s[0], s[1], s[2]]
    U0_hat = np.empty((3,)+FFT0.complex_shape(), FFT0.complex)
    U0_pad = np.empty((3,)+FFT0.real_shape_padded(), dtype=FFT0.float)
    if mesh == "triplyperiodic":
        for i in range(3):
            U0_hat[i] = FFT0.fftn(U0[i], U0_hat[i])
            
        for i in range(3):
            U0_pad[i] = FFT0.ifftn(U0_hat[i], U0_pad[i], dealias="3/2-rule") # Name is 3/2-rule, but padsize is 2

    else:
        U0_hat[0] = FFT0.fst(U0[0], U0_hat[0], SB)
        for i in range(1,3):
            U0_hat[i] = FFT0.fst(U0[i], U0_hat[i], ST)
        
        U0_pad[0] = FFT0.ifst(U0_hat[0], U0_pad[0], SB, dealias="3/2-rule")
        for i in range(1,3):
            U0_pad[i] = FFT0.ifst(U0_hat[i], U0_pad[i], ST, dealias="3/2-rule") # Name is 3/2-rule, but padsize is 2

    # Get new values
    fout["3D/checkpoint/U/0"][:, s1[0], s1[1], s1[2]] = U0_pad[:]
    U0[:] = fin["3D/checkpoint/U/1"][:, s[0], s[1], s[2]]
    
    if mesh == "triplyperiodic":
        for i in range(3):
            U0_hat[i] = FFT0.fftn(U0[i], U0_hat[i])    
        for i in range(3):
            U0_pad[i] = FFT0.ifftn(U0_hat[i], U0_pad[i], dealias="3/2-rule")    
    else:
        U0_hat[0] = FFT0.fst(U0[0], U0_hat[0], SB)
        for i in range(1,3):
            U0_hat[i] = FFT0.fst(U0[i], U0_hat[i], ST)
        
        U0_pad[0] = FFT0.ifst(U0_hat[0], U0_pad[0], SB, dealias="3/2-rule")
        for i in range(1,3):
            U0_pad[i] = FFT0.ifst(U0_hat[i], U0_pad[i], ST, dealias="3/2-rule") # Name is 3/2-rule, but padsize is 2

    fout["3D/checkpoint/U/1"][:, s1[0], s1[1], s1[2]] = U0_pad[:]
    fout.close()
    fin.close()
    
if __name__=="__main__":
    import sys
    refine(sys.argv[-2], sys.argv[-1])