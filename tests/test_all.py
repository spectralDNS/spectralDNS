import pytest
import subprocess

def test_single_DNS():
    d = subprocess.check_output("mpirun -np 1 python TG.py NS", shell=True)

def test_single_DNS_M654_write():
    d = subprocess.check_output("mpirun -np 1 python TG.py --M 6 5 4 --L 6*pi 4*pi 2*pi NS", shell=True)

def test_single_DNS_integrators():
    d = subprocess.check_output("mpirun -np 1 python TG_integrators.py NS", shell=True)

def test_single_NS2D():
    d = subprocess.check_output("mpirun -np 1 python TG2D.py NS2D", shell=True)
    
def test_single_KMM():
    d = subprocess.check_output("mpirun -np 1 python OrrSommerfeld.py --optimization cython KMM", shell=True)

def test_single_KMM2():
    d = subprocess.check_output("mpirun -np 1 python OrrSommerfeld.py --optimization cython --dealias 3/2-rule KMM", shell=True)

def test_single_KMM3():
    d = subprocess.check_output("mpirun -np 1 python OrrSommerfeld.py --optimization cython --dealias 3/2-rule --dealias_cheb KMM", shell=True)

def test_single_IPCS():
    d = subprocess.check_output("mpirun -np 1 python OrrSommerfeld.py --optimization cython IPCS", shell=True)

def test_single_IPCSR():
    d = subprocess.check_output("mpirun -np 1 python OrrSommerfeld.py IPCSR", shell=True)

def test_mpi_slab_DNS():
    d = subprocess.check_output("mpirun -np 4 python TG.py NS", shell=True)

def test_mpi_slab_DNS_M654_write():
    d = subprocess.check_output("mpirun -np 4 python TG.py --M 6 5 4 --L 6*pi 4*pi 2*pi NS", shell=True)

def test_mpi_DNS2D():
    d = subprocess.check_output("mpirun -np 4 python TG2D.py NS2D", shell=True)

def test_mpi_pencil4_DNS():
    d = subprocess.check_output("mpirun -np 4 python TG.py --decomposition pencil NS", shell=True)

def test_mpi_pencil8_DNS():
    d = subprocess.check_output("mpirun -np 8 python TG.py --decomposition pencil NS", shell=True)

def test_single_MHD():
    d = subprocess.check_output("mpirun -np 1 python TGMHD.py MHD", shell=True)
    
def test_mpi_slab_MHD():
    d = subprocess.check_output("mpirun -np 4 python TGMHD.py MHD", shell=True)
    
def test_mpi_pencil_MHD():
    d = subprocess.check_output("mpirun -np 4 python TGMHD.py --decomposition pencil MHD", shell=True)
    
def test_mpi_KMM():
    d = subprocess.check_output("mpirun -np 4 python OrrSommerfeld.py --optimization cython --checkpoint 5 KMM", shell=True)

def test_mpi_KMM2():
    d = subprocess.check_output("mpirun -np 4 python OrrSommerfeld.py --optimization cython --checkpoint 5 --dealias 3/2-rule  KMM", shell=True)

def test_mpi_KMM3():
    d = subprocess.check_output("mpirun -np 4 python OrrSommerfeld.py --optimization cython --checkpoint 5 --dealias 3/2-rule --dealias_cheb KMM", shell=True)

def test_mpi_IPCS():
    d = subprocess.check_output("mpirun -np 4 python OrrSommerfeld.py --optimization cython --checkpoint 5 IPCS", shell=True)

def test_mpi_IPCSR():
    d = subprocess.check_output("mpirun -np 4 python OrrSommerfeld.py --optimization cython --checkpoint 5 IPCSR", shell=True)