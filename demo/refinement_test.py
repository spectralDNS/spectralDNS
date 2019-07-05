import re
import subprocess
from numpy import log, array

h = []
error = []
number = "([0-9]+.[0-9]+e[+-][0-9]+)"
#dt = 2*array([1./100., 1./120., 1./140., 1./160., 1./180, 1./200., 1./300. , 1./400.])
#dt = array([1./6., 1./8., 1./10., 1./12., 1./20., 1./24.])
#dt = array([ 1./20., 1./30., 1./40., 1./50., 1./60.])
dt = array([1./10, 1./15., 1./20., 1./25., 1./30., 1./35., 1./40.])*10
#dt = array([ 1./35., 1./40.])
#dt = array([ 1./10., 1./12., 1./14., 1./16., 1./18.])
#dt = array([ 1./5., 1./6., 1./7., 1./8., 1./9.])
#dt = array([ 4./5., 2./3., 1./2., 2./5., 1./3., 1./4., 1./5., 1./6., 1./7., 1./8.])

#dt = array([0.0025, 0.005, 0.01, 0.02])

for i, t in enumerate(dt):
    output = subprocess.check_output("python OrrSommerfeld.py --refinement_test True --M 7 3 2 --dt %s --T %s --compute_energy 1 --plot_step -1 --convection Vortex --eps 1e-7 --optimization cython --Dquad GC --Bquad GC IPCSRK3"%(str(t), str(10.0)), shell=True)
    match = re.search("Computed error = "+number+" "+number, output.decode("utf-8"))
    err, h1 = [float(j) for j in match.groups(0)]
    error.append(err)
    h.append(h1)
    if i == 0:
        print("Error          hmin           r       ")
        print("%2.8e %2.8e %2.8f"%(error[-1], h[-1], 0))
    if i > 0:
        print("%2.8e %2.8e %2.8f"%(error[-1], h[-1], log(error[-1]/error[-2])/log(h[-1]/h[-2])))
