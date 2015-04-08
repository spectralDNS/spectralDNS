"""
Import from the problem 
"""
import config

d = {2: "TwoD", 3: "ThreeD"}
exec "from {0}.{1} import *".format(d[config.dimensions], config.problem)
