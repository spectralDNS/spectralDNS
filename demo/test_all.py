import pytest
import subprocess

def test_single_DNS():
    subprocess.call("mpirun -np 1 python TG.py --solver NS --compute_energy 1000", shell=True)

def test_single_DNS2():
    subprocess.call("mpirun -np 1 python TG.py --solver NS --optimization cython --compute_energy 1000", shell=True)

def test_single_DNS3():
    subprocess.call("mpirun -np 1 python TG.py --solver NS --M 6 6 6 --compute_energy 1000", shell=True)

def test_single_DNS4():
    subprocess.call("mpirun -np 1 python TG.py --solver NS --M 6 5 4 --L 6*pi 4*pi 2*pi --compute_energy 1000", shell=True)

def test_single_DNS2D():
    subprocess.call("mpirun -np 1 python TG2D.py --solver NS2D --plot_result -1", shell=True)
    
def test_single_VV():
    subprocess.call("mpirun -np 1 python TG.py --solver VV --compute_energy 1000", shell=True)

def test_single_VV2():
    subprocess.call("mpirun -np 1 python TG.py --solver VV --optimization cython --compute_energy 1000", shell=True)

def test_mpi_slab_DNS():
    subprocess.call("mpirun -np 4 python TG.py --solver NS --compute_energy 1000", shell=True)

def test_mpi_pencil_DNS():
    subprocess.call("mpirun -np 4 python TG.py --solver NS --decomposition pencil --P1 2 --compute_energy 1000", shell=True)

def test_mpi_pencil_DNS2(): # Uneven number of processors in each direction
    subprocess.call("mpirun -np 8 python TG.py --solver NS --decomposition pencil --P1 2 --compute_energy 1000", shell=True)
    
def test_mpi_slab_VV():
    subprocess.call("mpirun -np 4 python TG.py --solver VV --compute_energy 1000", shell=True)

def test_mpi_pencil_VV():
    subprocess.call("mpirun -np 4 python TG.py --solver VV --decomposition pencil --P1 2 --compute_energy 1000", shell=True)
    
def test_single_MHD():
    subprocess.call("mpirun -np 1 python TGMHD.py --solver MHD --compute_energy 1000", shell=True)
    
