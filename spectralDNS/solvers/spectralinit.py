__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import sys, cProfile
from spectralDNS.utilities import *
from spectralDNS.h5io import *

# Import problem specific methods and solver methods specific to either slab or pencil decomposition
from spectralDNS.maths import *

#TODO:Figure out what the lines below do
"""
def set_source(Source, **kwargs):
    Source[:] = 0
    return Source
    """

