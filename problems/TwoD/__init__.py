__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-03-07"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

parameters = {
    'M': 6,
    'temporal': 'RK4',
    'plot_result': -1,         # Show an image every..
    'nu': 0.01,
    'dt': 0.01,
    'T': 1.0,
    'problem': 'TaylorGreen',
    'debug': False
}

def initialize(U, **kwargs):
    return U

def update(**kwargs):
    pass

def regression_test(**kwargs):
    pass
