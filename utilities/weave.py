__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-03-10"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from scipy.weave import converters
import scipy.weave.inline_tools as inline_tools
    
def weave_rhs(dU, nu, P_hat, U_hat, K2, K, K_over_K2, dealias):
    """Compute most of righ hand side using optimized C++ code
    """
    code = """
double n = nu;
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
      double z = n*K2(i,j,k); 
      double k0 = K(0,i,j,k);
      double k1 = K(1,i,j,k);
      double k2 = K(2,i,j,k);
      P_hat(i,j,k) = dU(0,i,j,k)*K_over_K2(0,i,j,k)+dU(1,i,j,k)*K_over_K2(1,i,j,k)+dU(2,i,j,k)*K_over_K2(2,i,j,k);
      dU(0,i,j,k) -= (P_hat(i,j,k)*k0 + U_hat(0,i,j,k)*z);
      dU(1,i,j,k) -= (P_hat(i,j,k)*k1 + U_hat(1,i,j,k)*z);
      dU(2,i,j,k) -= (P_hat(i,j,k)*k2 + U_hat(2,i,j,k)*z);
    }
  }
}
"""
    inline_tools.inline(code, ['dU', 'nu', 'P_hat', 'U_hat', 'K2', 'K', 'K_over_K2', 'dealias'],
                 type_converters=converters.blitz, verbose=2, auto_downcast=0,
                 extra_compile_args=['-O3', '-ffast-math'])
    return dU
