import pytest
import subprocess

def test_single_DNS():
    d = subprocess.check_output("mpirun -np 1 python TG.py --solver NS", shell=True)

def test_single_DNS2():
    d = subprocess.check_output("mpirun -np 1 python TG.py --solver NS --optimization cython", shell=True)

def test_single_DNS3():
    d = subprocess.check_output("mpirun -np 1 python TG.py --solver NS --M 6 6 6", shell=True)
    
def test_single_DNS4():
    d = subprocess.check_output("mpirun -np 1 python TG.py --solver NS --M 6 5 4 --L 6*pi 4*pi 2*pi", shell=True)

def test_single_DNS5():
    d = subprocess.check_output("mpirun -np 1 python TG.py --solver NS --M 4 4 4 --write_result 5", shell=True)

def test_single_DNS2D():
    d = subprocess.check_output("mpirun -np 1 python TG2D.py --solver NS2D", shell=True)
    
def test_single_VV():
    d = subprocess.check_output("mpirun -np 1 python TG.py --solver VV", shell=True)

def test_single_VV2():
    d = subprocess.check_output("mpirun -np 1 python TG.py --solver VV --optimization cython", shell=True)

def test_single_KMM():
    d = subprocess.check_output("mpirun -np 1 python OrrSommerfeld.py --solver KMM --optimization cython", shell=True)

def test_single_IPCS():
    d = subprocess.check_output("mpirun -np 1 python OrrSommerfeld.py --solver IPCS --optimization cython", shell=True)

def test_mpi_slab_DNS():
    d = subprocess.check_output("mpirun -np 4 python TG.py --solver NS", shell=True)

def test_mpi_pencil_DNS():
    d = subprocess.check_output("mpirun -np 4 python TG.py --solver NS --decomposition pencil --P1 2", shell=True)

def test_mpi_pencil_DNS2():
    d = subprocess.check_output("mpirun -np 8 python TG.py --solver NS --decomposition pencil --P1 2", shell=True)
    
def test_mpi_slab_VV():
    d = subprocess.check_output("mpirun -np 4 python TG.py --solver VV", shell=True)

def test_mpi_pencil_VV():
    d = subprocess.check_output("mpirun -np 4 python TG.py --solver VV --decomposition pencil --P1 2", shell=True)
    
def test_single_MHD():
    d = subprocess.check_output("mpirun -np 1 python TGMHD.py --solver MHD", shell=True)
    
def test_mpi_slab_MHD():
    d = subprocess.check_output("mpirun -np 4 python TGMHD.py --solver MHD", shell=True)
    
def test_mpi_pencil_MHD():
    d = subprocess.check_output("mpirun -np 4 python TGMHD.py --solver MHD --decomposition pencil --P1 2", shell=True)
    
def test_mpi_slab_DNS2D():
    d = subprocess.check_output("mpirun -np 4 python TG2D.py --solver NS2D", shell=True)

def test_mpi_KMM():
    d = subprocess.check_output("mpirun -np 4 python OrrSommerfeld.py --solver KMM --optimization cython", shell=True)

def test_mpi_IPCS():
    d = subprocess.check_output("mpirun -np 4 python OrrSommerfeld.py --solver IPCS --optimization cython", shell=True)
