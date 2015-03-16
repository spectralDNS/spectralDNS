__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-03-10"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from numpy import *
from scipy.weave import converters, ext_tools
import scipy.weave.inline_tools as inline_tools
import scipy.weave.c_spec as c_spec
import sys

def weave_module(precision):
    """Compute most of righ hand side using optimized C++ code
    """
    float, complex = {"single": (float32, complex64),
                      "double": (float64, complex128)}[precision]
    
    nu = float(1.0)
    dU = empty((3, 3, 3, 3), dtype=complex)
    U_hat = empty((3, 3, 3, 3), dtype=complex)
    P_hat = empty((3, 3, 3), dtype=complex)
    K = empty((3, 3, 3, 3), dtype=int)
    K2 = empty((3, 3, 3), dtype=float)
    K_over_K2 = empty((3, 3, 3, 3), dtype=float)
    dealias = empty((3, 3, 3), dtype=bool)
 
    numeric_type = c_spec.num_to_c_types[K2.dtype.char]
    code = """
%s n = nu;
int N1 = NdU[1];
int N2 = NdU[2];
int N3 = NdU[3];
for (int i=0;i<N1;i++){
  for (int j=0;j<N2;j++){
    for (int k=0;k<N3;k++)
    {  
      dU(0,i,j,k) *= dealias(i,j,k);
      dU(1,i,j,k) *= dealias(i,j,k);
      dU(2,i,j,k) *= dealias(i,j,k);
    }
  }
}
for (int i=0;i<N1;i++){
  for (int j=0;j<N2;j++){
    for (int k=0;k<N3;k++)
    { // Some casting required to compile weave
      %s z = n*K2(i,j,k); 
      %s k0 = K(0,i,j,k);
      %s k1 = K(1,i,j,k);
      %s k2 = K(2,i,j,k);
      P_hat(i,j,k) = dU(0,i,j,k)*K_over_K2(0,i,j,k)+dU(1,i,j,k)*K_over_K2(1,i,j,k)+dU(2,i,j,k)*K_over_K2(2,i,j,k);
      dU(0,i,j,k) -= (P_hat(i,j,k)*k0 + U_hat(0,i,j,k)*z);
      dU(1,i,j,k) -= (P_hat(i,j,k)*k1 + U_hat(1,i,j,k)*z);
      dU(2,i,j,k) -= (P_hat(i,j,k)*k2 + U_hat(2,i,j,k)*z);  
    }
  }
}
"""%(numeric_type, numeric_type, numeric_type, numeric_type, numeric_type)

    mod = ext_tools.ext_module("weave_"+precision)
    fun = ext_tools.ext_function("weaverhs", code, ['dU', 'U_hat', 'K2', 'K', 'P_hat', 'K_over_K2', 'dealias', 'nu'],
                                 type_converters=converters.blitz)
    mod.add_function(fun)
    
    a = empty((3, 3, 3, 3), dtype=float)
    b = empty((3, 3, 3, 3), dtype=float)
    c = empty((3, 3, 3, 3), dtype=float)
    numeric_type = c_spec.num_to_c_types[a.dtype.char]
    code = """
int N1 = Na[1];
int N2 = Na[2];
int N3 = Na[3];
%s a0, a1, a2, b0, b1, b2;
for (int i=0;i<N1;i++){
  for (int j=0;j<N2;j++){
    for (int k=0;k<N3;k++)
    {  
      a0 = a(0,i,j,k);a1 = a(1,i,j,k);a2 = a(2,i,j,k);
      b0 = b(0,i,j,k);b1 = b(1,i,j,k);b2 = b(2,i,j,k);
      c(0,i,j,k) = a1*b2 - a2*b1;
      c(1,i,j,k) = a2*b0 - a0*b2;
      c(2,i,j,k) = a0*b1 - a1*b0;
    }
  }
}
""" %numeric_type
    fun2 = ext_tools.ext_function("weavecross", code, ['a', 'b', 'c'],
                                  type_converters=converters.blitz)
    mod.add_function(fun2)
    
    a = empty((3, 3, 3, 3), dtype=complex)
    b = empty((3, 3, 3, 3), dtype=int)  
    c = empty((3, 3, 3, 3), dtype=complex)

    numeric_type_a = c_spec.num_to_c_types[a.dtype.char]
    numeric_type_b = c_spec.num_to_c_types[b.dtype.char]
    code = """
int N1 = Na[1];
int N2 = Na[2];
int N3 = Na[3];
%s a0, a1, a2;
%s b0, b1, b2;
for (int i=0;i<N1;i++){
  for (int j=0;j<N2;j++){
    for (int k=0;k<N3;k++)
    {  
      a0 = a(0,i,j,k);a1 = a(1,i,j,k);a2 = a(2,i,j,k);
      b0 = b(0,i,j,k);b1 = b(1,i,j,k);b2 = b(2,i,j,k);

      c(0,i,j,k) = %s (a1.imag()*b2 - a2.imag()*b1, -(a1.real()*b2 - a2.real()*b1));
      c(1,i,j,k) = %s (a2.imag()*b0 - a0.imag()*b2, -(a2.real()*b0 - a0.real()*b2));
      c(2,i,j,k) = %s (a0.imag()*b1 - a1.imag()*b0, -(a0.real()*b1 - a1.real()*b0));
      
    }
  }
}
""" %(numeric_type_a, numeric_type_b, numeric_type_a, numeric_type_a,numeric_type_a)
    fun3 = ext_tools.ext_function("weavecrossi", code, ['a', 'b', 'c'],
                                  type_converters=converters.blitz)
    mod.add_function(fun3)
    mod.compile(extra_compile_args=['-O3', '-ffast-math'], verbose=2)

import weave_single, weave_double
def weaverhs(dU, U_hat, K2, K, P_hat, K_over_K2, dealias, nu, precision="double"):
    if precision == "single":
        weave_single.weaverhs(dU, U_hat, K2, K, P_hat, K_over_K2, dealias, nu)
    else:
        weave_double.weaverhs(dU, U_hat, K2, K, P_hat, K_over_K2, dealias, nu)
        
def weavecross(a, b, c, precision="double"):
    if precision == "single":
        weave_single.weavecross(a, b, c)
    else:
        weave_double.weavecross(a, b, c)
        
def weavecrossi(a, b, c, precision="double"):
    if precision == "single":
        weave_single.weavecrossi(a, b, c)
    else:
        weave_double.weavecrossi(a, b, c)
        
if __name__=="__main__":
    weave_module(sys.argv[-1])
    