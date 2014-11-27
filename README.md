spectralDNS
=======
spectralDNS is a classical pseudo-spectral direct Navier-Stokes solver for triply periodic domains. The only unique feature is that it's written entirely in python using numpy, mpi4py and pyfftw and, stripping away unnecessary pre- and post-processing steps, the solver is approximately 100 lines long, including the MPI. The code scales very well weakly on the Abel supercomputer at the University of Oslo, at least up to 512 cores. For a cube of size 512**3, the code runs at 1.5 seconds per time step.
<p align="center">
    <img src="https://raw.github.com/wiki/mikaem/spectralDNS/figs/weak_scaling_avg.png" width="600" height="400" alt="Weak scaling on Abel cluster"/>
</p>
<p align="center">
    Weak scaling on the Abel cluster.
</p>


