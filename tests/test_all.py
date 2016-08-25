import pytest
import subprocess

def test_single_DNS():
    d = subprocess.check_output("mpirun -np 1 python TG.py", shell=True)

def test_single_DNS_M654_write():
    d = subprocess.check_output("mpirun -np 1 python TG.py --M 6 5 4 --L 6*pi 4*pi 2*pi", shell=True)

def test_single_DNS_integrators():
    d = subprocess.check_output("mpirun -np 1 python TG_integrators.py", shell=True)

def test_single_NS2D():
    d = subprocess.check_output("mpirun -np 1 python TG2D.py NS2D", shell=True)
    
def test_single_channel():
    d = subprocess.check_output("mpirun -np 1 python OrrSommerfeld.py", shell=True)

def test_mpi_slab_DNS():
    d = subprocess.check_output("mpirun -np 4 python TG.py", shell=True)

def test_mpi_slab_DNS_M654_write():
    d = subprocess.check_output("mpirun -np 4 python TG.py --M 6 5 4 --L 6*pi 4*pi 2*pi", shell=True)

def test_mpi_DNS2D():
    d = subprocess.check_output("mpirun -np 4 python TG2D.py NS2D", shell=True)

def test_mpi_pencil4_DNS():
    d = subprocess.check_output("mpirun -np 4 python TG.py --decomposition pencil", shell=True)

def test_mpi_pencil8_DNS():
    d = subprocess.check_output("mpirun -np 8 python TG.py --decomposition pencil", shell=True)

def test_single_MHD():
    d = subprocess.check_output("mpirun -np 1 python TGMHD.py MHD", shell=True)
    
def test_mpi_slab_MHD():
    d = subprocess.check_output("mpirun -np 4 python TGMHD.py MHD", shell=True)
    
def test_mpi_pencil_MHD():
    d = subprocess.check_output("mpirun -np 4 python TGMHD.py --decomposition pencil MHD", shell=True)
    
def test_mpi_channel():
    d = subprocess.check_output("mpirun -np 4 python OrrSommerfeld.py", shell=True)
