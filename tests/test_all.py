import pytest
import subprocess

def test_single_channel():
    d = subprocess.check_output("mpirun -np 1 python OrrSommerfeld.py", shell=True)

def test_single_MHD():
    d = subprocess.check_output("mpirun -np 1 python TGMHD.py MHD", shell=True)
    
def test_mpi_slab_MHD():
    d = subprocess.check_output("mpirun -np 4 python TGMHD.py MHD", shell=True)
    
def test_mpi_pencil_MHD():
    d = subprocess.check_output("mpirun -np 4 python TGMHD.py --decomposition pencil MHD", shell=True)
    
def test_mpi_channel():
    d = subprocess.check_output("mpirun -np 4 python OrrSommerfeld.py", shell=True)
