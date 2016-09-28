import subprocess
from numpy import log, array
import sys
import re

h = []
error = []
number= "([0-9]+.[0-9]+e[+-][0-9]+)"
#dt = 2*array([1./100., 1./120., 1./140., 1./160., 1./180, 1./200., 1./300. , 1./400.])
#dt = array([1./6., 1./8., 1./10., 1./12., 1./20., 1./24.])
#dt = array([ 1./20., 1./30., 1./40., 1./50., 1./60.])
dt = array([ 1./2., 1./3., 1./4., 1./5., 1./6.])
#dt = array([0.0025, 0.005, 0.01, 0.02])

for i in range(len(dt)):
    output = subprocess.check_output("python OrrSommerfeld.py --refinement_test True --M 7 5 1 --dt %s --T %s --compute_energy 1000 --plot_step -1 --convection Vortex KMM"%(str(dt[i]),str(4.0)), shell=True)
    match = re.search("Computed error = "+number+" "+number, output)
    err, h1 = [eval(j) for j in match.groups(0)]
    error.append(err)
    h.append(h1)
    if i == 0:
        print "Error          hmin           r       "
        print "%2.8e %2.8e %2.8f"%(error[-1], h[-1], 0)
    if i > 0:
        print "%2.8e %2.8e %2.8f"%(error[-1], h[-1], log(error[-1]/error[-2])/log(h[-1]/h[-2]))
