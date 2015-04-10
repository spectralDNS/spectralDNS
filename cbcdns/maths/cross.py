__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-07"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..optimization import optimizer

__all__ = ['cross1', 'cross2', 'cross3']

@optimizer
def cross1(c, a, b):
    """Regular c = a x b, where type(a) = type(b) is real"""
    #c[:] = cross(a, b, axisa=0, axisb=0, axisc=0) # Very slow
    c[0] = a[1]*b[2]-a[2]*b[1]
    c[1] = a[2]*b[0]-a[0]*b[2]
    c[2] = a[0]*b[1]-a[1]*b[0]
    return c

@optimizer    
def cross2(c, a, b):
    """ c = 1j*(a x b), where type(a) is int and type(b) is complex"""
    c = cross1(c, a, b)
    c *= 1j
    return c

@optimizer    
def cross3(c, a, b):
    """ c = 1j*(a x b), where type(a) is float and type(b) is complex.
    
    Differs from cross2 only in strongly typed, optimized implementations
    
    """
    c = cross1(c, a, b)
    c *= 1j
    return c
