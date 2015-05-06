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
    
    float, complex = {"single": (float32, complex64),
                      "double": (float64, complex128)}[precision]
    
    # Just declare some variables with correct type. The size and values are arbitrary at this point
    nu = float(1.0)
    dU = empty((3, 3, 3, 3), dtype=complex)
    U_hat = empty((3, 3, 3, 3), dtype=complex)
    P_hat = empty((3, 3, 3), dtype=complex)
    K = empty((3, 3, 3, 3), dtype=float)
    K2 = empty((3, 3, 3), dtype=float)
    K_over_K2 = empty((3, 3, 3, 3), dtype=float)
    dealias = empty((3, 3, 3), dtype=uint8)
 
    numeric_type = c_spec.num_to_c_types[K_over_K2.dtype.char]    
    code0 = """
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
}"""
    mod = ext_tools.ext_module("weave_"+precision)    
    fun0 = ext_tools.ext_function("dealias_rhs", code0, ['dU', 'dealias'],
                                 type_converters=converters.blitz)    
    mod.add_function(fun0)
        
    code = """
%s n = nu;
int N1 = NdU[1];
int N2 = NdU[2];
int N3 = NdU[3];
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
    
    fun = ext_tools.ext_function("add_pressure_diffusion", 
                                 code, ['dU', 'U_hat', 'K2', 'K', 'P_hat', 'K_over_K2', 'nu'],
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
    fun2 = ext_tools.ext_function("cross1", code, ['c', 'a', 'b'],
                                  type_converters=converters.blitz)
    mod.add_function(fun2)
    
    a = empty((3, 3, 3, 3), dtype=int)
    b = empty((3, 3, 3, 3), dtype=complex)  
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

      c(0,i,j,k) = %s (-(a1*b2.imag() - a2*b1.imag()), a1*b2.real() - a2*b1.real());
      c(1,i,j,k) = %s (-(a2*b0.imag() - a0*b2.imag()), a2*b0.real() - a0*b2.real());
      c(2,i,j,k) = %s (-(a0*b1.imag() - a1*b0.imag()), a0*b1.real() - a1*b0.real());
      
    }
  }
}
""" %(numeric_type_a, numeric_type_b, numeric_type_b, numeric_type_b,numeric_type_b)
    fun3 = ext_tools.ext_function("cross2a", code, ['c', 'a', 'b'],
                                  type_converters=converters.blitz)
    mod.add_function(fun3)
    
    a = empty((3, 3, 3, 3), dtype=float)
    numeric_type_a = c_spec.num_to_c_types[a.dtype.char]
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

      c(0,i,j,k) = %s (-(a1*b2.imag() - a2*b1.imag()), a1*b2.real() - a2*b1.real());
      c(1,i,j,k) = %s (-(a2*b0.imag() - a0*b2.imag()), a2*b0.real() - a0*b2.real());
      c(2,i,j,k) = %s (-(a0*b1.imag() - a1*b0.imag()), a0*b1.real() - a1*b0.real());
      
    }
  }
}
""" %(numeric_type_a, numeric_type_b, numeric_type_b, numeric_type_b,numeric_type_b)
    fun3b = ext_tools.ext_function("cross2b", code, ['c', 'a', 'b'],
                                  type_converters=converters.blitz)
    mod.add_function(fun3b)        

    Uc_hatT = empty((4, 4, 4), dtype=complex)
    U_mpi   = empty((2, 4, 2, 4), dtype=complex)
    num_processes = int(1)
    N1 = N2 = P1 = P2 = Np = Nf = int(1)

    code = """
int kk;
int n0 = NUc_hatT[0];
int n1 = NU_mpi[2];
int n2 = NU_mpi[3];    
for (int i=0; i<num_processes; i++){
  for (int j=0; j<n0; j++){
    for (int k=i*n1; k<(i+1)*n1; k++){
      kk = k-i*n1;
      for (int l=0; l<n2; l++){      
        Uc_hatT(j, k, l) = U_mpi(i, j, kk, l);
      }
    }
  }
}
"""
    fun4 = ext_tools.ext_function("transpose_Uc", code, ['Uc_hatT', 'U_mpi', 'num_processes'],
                                  type_converters=converters.blitz)
    mod.add_function(fun4)
    
    code = """
int kk;
int n0 = NU_mpi[1];
int n1 = NU_mpi[2];
int n2 = NU_mpi[3];    
for (int i=0; i<num_processes; i++){
  for (int j=0; j<n0; j++){
    for (int k=i*n1; k<(i+1)*n1; k++){
      kk = k-i*n1;
      for (int l=0; l<n2; l++){      
        U_mpi(i, j, kk, l) = Uc_hatT(j, k, l);
      }
    }
  }
}
"""
    fun5 = ext_tools.ext_function("transpose_Umpi", code, ['U_mpi', 'Uc_hatT', 'num_processes'],
                                  type_converters=converters.blitz)
    mod.add_function(fun5)

    Uc_hat_x = empty((4, 4, 4), dtype=complex)
    Uc_hat_y = empty((4, 4, 4), dtype=complex)
    Uc_hat_z = empty((4, 4, 4), dtype=complex)
    Uc_hat_xr = empty((4, 4, 4), dtype=complex)
    

    code = """
int i0, kk;
int n0 = NUc_hat_z[0];
int n1 = NUc_hat_z[1];
int n2 = NUc_hat_x[2];
for (int i=0; i<P1; i++){
  for (int j=i*n0; j<(i+1)*n0; j++){
    i0 = j-i*n0;
    for (int k=0; k<n1; k++){
      for (int l=0; l<n2; l++){      
        Uc_hat_x(j, k, l) = Uc_hat_z(i0, k, l+i*n2);
      }
    }
  }
}
"""
    fun6 = ext_tools.ext_function("transform_Uc_xz", code, ['Uc_hat_x', 'Uc_hat_z', 'P1'],
                                  type_converters=converters.blitz)
    mod.add_function(fun6)
    
    code = """
int i0, k0;
int n0 = NUc_hat_y[0];
int n1 = NUc_hat_xr[1];
int n2 = NUc_hat_xr[2];    
for (int i=0; i<P2; i++){
  for (int j=i*n0; j<(i+1)*n0; j++){
    i0 = j-i*n0;
    for (int k=0; k<n1; k++){
      k0 = k+i*n1;
      for (int l=0; l<n2; l++){      
        Uc_hat_y(i0, k0, l) = Uc_hat_xr(j, k, l);
      }
    }
  }
}
"""
    fun7 = ext_tools.ext_function("transform_Uc_yx", code, ['Uc_hat_y', 'Uc_hat_xr', 'P2'],
                                  type_converters=converters.blitz)
    mod.add_function(fun7)

    code = """
int i0;
int n0 = NUc_hat_y[0];
int n1 = NUc_hat_x[1];
int n2 = NUc_hat_x[2];
for (int i=0; i<P2; i++){
  for (int j=i*n0; j<(i+1)*n0; j++){
    i0 = j-i*n0;
    for (int k=0; k<n1; k++){
      for (int l=0; l<n2; l++){      
        Uc_hat_x(j, k, l) = Uc_hat_y(i0, k+i*n1, l);
      }
    }
  }
}
"""
    fun8 = ext_tools.ext_function("transform_Uc_xy", code, ['Uc_hat_x', 'Uc_hat_y', 'P2'],
                                  type_converters=converters.blitz)
    mod.add_function(fun8)

    code = """
int i0;
int n0 = NUc_hat_z[0];
int n1 = NUc_hat_xr[1];
int n2 = NUc_hat_xr[2];
for (int i=0; i<P1; i++){
  for (int j=i*n0; j<(i+1)*n0; j++){
    i0 = j-i*n0;
    for (int k=0; k<n1; k++){
      for (int l=0; l<n2; l++){      
        Uc_hat_z(i0, k, l+i*n2) = Uc_hat_xr(j, k, l);
      }
    }
  }
}
"""
    fun9 = ext_tools.ext_function("transform_Uc_zx", code, ['Uc_hat_z', 'Uc_hat_xr', 'P1'],
                                  type_converters=converters.blitz)
    mod.add_function(fun9)
    
    return mod
        
if __name__=="__main__":
    mod = weave_module(sys.argv[-1])
    