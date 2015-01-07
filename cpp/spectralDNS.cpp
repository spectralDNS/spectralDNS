#include "fftw3-mpi.h"
#include <math.h>
#include <iostream> 
#include <iomanip>
#include <complex>
#include <vector>

using namespace std;

// mpic++ -std=c++11 -O3 spectralDNS.cpp -o spectralDNS

int main( int argc, char *argv[] )
{
  int rank, num_processes, M, N, Np, Nf;
  double wtime, nu, T, dt, L, dx;
  double t0, t1;
  MPI::Init ( argc, argv );
  fftw_mpi_init();
  
  t0 = MPI::Wtime();
  
  num_processes = MPI::COMM_WORLD.Get_size ( );
  rank = MPI::COMM_WORLD.Get_rank ( );
  double pi = 3.141592653589793238;
  ptrdiff_t alloc_local, local_n0, local_0_start, local_n1, local_1_start, i, j, k;

  nu = 0.000625;
  T = 0.1;
  dt = 0.01;
  M = 7;
  N = pow(2, M);
  L = 2*pi;
  Np = N / num_processes;
  dx = L / N;
  std::cout << std::scientific << std::setprecision(16);
  
  vector<double> a {1./6., 1./3., 1./3., 1./6.};
  vector<double> b {0.5, 0.5, 1.0};
  int tot = N*N*N;
  Nf = N/2+1;
  alloc_local = fftw_mpi_local_size_3d_transposed(N, N, Nf, MPI::COMM_WORLD,
                                        &local_n0, &local_0_start,
                                        &local_n1, &local_1_start);
  
  vector<double> U(2*alloc_local);
  vector<double> V(2*alloc_local);
  vector<double> W(2*alloc_local);
  vector<double> U_tmp(2*alloc_local);
  vector<double> V_tmp(2*alloc_local);
  vector<double> W_tmp(2*alloc_local);
  vector<double> CU(2*alloc_local);
  vector<double> CV(2*alloc_local);
  vector<double> CW(2*alloc_local);  
  vector<complex<double> > U_hat(alloc_local);
  vector<complex<double> > V_hat(alloc_local);
  vector<complex<double> > W_hat(alloc_local);
  vector<double> P(2*alloc_local);
  vector<complex<double> > P_hat(alloc_local);
  vector<complex<double> > U_hat0(alloc_local);
  vector<complex<double> > V_hat0(alloc_local);
  vector<complex<double> > W_hat0(alloc_local);
  vector<complex<double> > U_hat1(alloc_local);
  vector<complex<double> > V_hat1(alloc_local);
  vector<complex<double> > W_hat1(alloc_local);
  vector<complex<double> > dU(alloc_local);
  vector<complex<double> > dV(alloc_local);
  vector<complex<double> > dW(alloc_local);  
  vector<complex<double> > curlX(alloc_local);
  vector<complex<double> > curlY(alloc_local);
  vector<complex<double> > curlZ(alloc_local);
  
  vector<double> kx(N);
  vector<double> kz(Nf);
  for (int i=0; i<N/2; i++)
  {
      kx[i] = i;
      kz[i] = i;
  }
  kz[N/2] = N/2;
  for (int i=-N/2; i<0; i++)
      kx[i+N] = i;

  //fftw_plan plan_backward;
  fftw_plan rfftn, irfftn;
  rfftn = fftw_mpi_plan_dft_r2c_3d(N, N, N, U.data(), reinterpret_cast<fftw_complex*>(U_hat.data()), 
                                   MPI::COMM_WORLD, FFTW_MPI_TRANSPOSED_OUT);
  irfftn = fftw_mpi_plan_dft_c2r_3d(N, N, N, reinterpret_cast<fftw_complex*>(U_hat.data()),  U.data(), 
                                   MPI::COMM_WORLD, FFTW_MPI_TRANSPOSED_IN);  
      
  for (int i=0; i<local_n0; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
      {
        int z = (i*N+j)*2*Nf+k;
        U[z] = sin(dx*(i+local_0_start))*cos(dx*j)*cos(dx*k);
        V[z] = -cos(dx*(i+local_0_start))*sin(dx*j)*cos(dx*k);
        W[z] = 0.0;
      }
    
  fftw_mpi_execute_dft_r2c( rfftn, U.data(), reinterpret_cast<fftw_complex*>(U_hat.data()));
  fftw_mpi_execute_dft_r2c( rfftn, V.data(), reinterpret_cast<fftw_complex*>(V_hat.data()));
  fftw_mpi_execute_dft_r2c( rfftn, W.data(), reinterpret_cast<fftw_complex*>(W_hat.data()));
      
  double kmax = 2./3.*(N/2+1);  
  complex<double> one(0, 1); 
  double t=0.0;
  int tstep = 0;
  while (t < T-1e-8)
  {
     t += dt;
     tstep++;
     for (int i=0; i<local_n1; i++)
       for (int j=0; j<N; j++)
         for (int k=0; k<Nf; k++)
         {
             int z = (i*N+j)*Nf+k;
             U_hat0[z] = U_hat[z];
             V_hat0[z] = V_hat[z];
             W_hat0[z] = W_hat[z];
             U_hat1[z] = U_hat[z];
             V_hat1[z] = V_hat[z];
             W_hat1[z] = W_hat[z];
        }
        
     for (int rk=0; rk<4; rk++)
     {
        if (rk > 0)
        {
            fftw_mpi_execute_dft_c2r(irfftn, reinterpret_cast<fftw_complex*>(U_hat.data()), U.data());
            fftw_mpi_execute_dft_c2r(irfftn, reinterpret_cast<fftw_complex*>(V_hat.data()), V.data());
            fftw_mpi_execute_dft_c2r(irfftn, reinterpret_cast<fftw_complex*>(W_hat.data()), W.data());
            for (int k=0; k<U.size(); k++)
            {
                U[k] /= tot;
                V[k] /= tot;
                W[k] /= tot;
            }
        }
        // Compute curl
        for (int i=0; i<local_n1; i++)
          for (int j=0; j<N; j++)
            for (int k=0; k<Nf; k++)
            {
                int z = (i*N+j)*Nf+k;
                curlZ[z] = one*(kx[i+local_1_start]*V_hat[z]-kx[j]*U_hat[z]);
                curlY[z] = one*(kz[k]*U_hat[z]-kx[i+local_1_start]*W_hat[z]);
                curlX[z] = one*(kx[j]*W_hat[z]-kz[k]*V_hat[z]);
            }                        
        fftw_mpi_execute_dft_c2r(irfftn, reinterpret_cast<fftw_complex*>(curlX.data()), CU.data());
        fftw_mpi_execute_dft_c2r(irfftn, reinterpret_cast<fftw_complex*>(curlY.data()), CV.data());
        fftw_mpi_execute_dft_c2r(irfftn, reinterpret_cast<fftw_complex*>(curlZ.data()), CW.data());
        for (int k=0; k<CU.size(); k++)
        {
            CU[k] /= tot;
            CV[k] /= tot;
            CW[k] /= tot;
        }
        
        // Cross
        for (int i=0; i<local_n0; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++)
                {
                  int z = (i*N+j)*2*Nf+k;
                  U_tmp[z] = V[z]*CW[z]-W[z]*CV[z];
                  V_tmp[z] = W[z]*CU[z]-U[z]*CW[z];
                  W_tmp[z] = U[z]*CV[z]-V[z]*CU[z];      
                }
                
        fftw_mpi_execute_dft_r2c( rfftn, U_tmp.data(), reinterpret_cast<fftw_complex*>(dU.data()));
        fftw_mpi_execute_dft_r2c( rfftn, V_tmp.data(), reinterpret_cast<fftw_complex*>(dV.data()));
        fftw_mpi_execute_dft_r2c( rfftn, W_tmp.data(), reinterpret_cast<fftw_complex*>(dW.data()));
        
        for (int i=0; i<local_n1; i++)
          for (int j=0; j<N; j++)
            for (int k=0; k<Nf; k++)
            {
              int z = (i*N+j)*Nf+k;
              int zero_or_one = (abs(kx[i+local_1_start])<kmax)*(abs(kx[j])<kmax)*(abs(kx[k])<kmax) == true ? 1 : 0;
              dU[z] *= (zero_or_one*dt);
              dV[z] *= (zero_or_one*dt);
              dW[z] *= (zero_or_one*dt);             
            }
        // 
        for (int i=0; i<local_n1; i++)
          for (int j=0; j<N; j++)
            for (int k=0; k<Nf; k++)
            {
                int z = (i*N+j)*Nf+k;
                double kk = kx[i+local_1_start]*kx[i+local_1_start] + kx[j]*kx[j] + kx[k]*kx[k];
                kk = kk > 0 ? kk : 1;
                P_hat[z] = (dU[z]*kx[i+local_1_start] + dV[z]*kx[j] + dW[z]*kz[k])/kk;
                dU[z] -= (P_hat[z]*kx[i+local_1_start] + nu*dt*kk*U_hat[z]);
                dV[z] -= (P_hat[z]*kx[j] + nu*dt*kk*V_hat[z]);
                dW[z] -= (P_hat[z]*kz[k] + nu*dt*kk*W_hat[z]);
            }
            
        if (rk < 3)
        {
          for (int i=0; i<local_n1; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<Nf; k++)
                {
                   int z = (i*N+j)*Nf+k;
                   U_hat[z] = U_hat0[z] + b[rk]*dU[z];
                   V_hat[z] = V_hat0[z] + b[rk]*dV[z];
                   W_hat[z] = W_hat0[z] + b[rk]*dW[z];
                }
            
        }
        for (int i=0; i<local_n1; i++)
          for (int j=0; j<N; j++)
            for (int k=0; k<Nf; k++)
            {
                int z = (i*N+j)*Nf+k;
                U_hat1[z] += a[rk]*dU[z];
                V_hat1[z] += a[rk]*dV[z];
                W_hat1[z] += a[rk]*dW[z];
            }        
     }     
    for (int i=0; i<local_n1; i++)
      for (int j=0; j<N; j++)
        for (int k=0; k<Nf; k++)
        {
            int z = (i*N+j)*Nf+k;
            U_hat[z] = U_hat1[z];
            V_hat[z] = V_hat1[z];
            W_hat[z] = W_hat1[z];
        }     
}

  vector<double> s(1), ss(1);
  s[0] = 0.0;
  for (int i=0; i<local_n0; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
      {
        int z = (i*N+j)*2*Nf+k;
        s[0] += (U[z]*U[z] + V[z]*V[z] + W[z]*W[z]);
      }
  s[0] *= (0.5*dx*dx*dx/L/L/L);

  MPI::COMM_WORLD.Reduce(s.data(), ss.data(), 1, MPI::DOUBLE, MPI::SUM, 0);  
  std::cout << " k = " << ss[0] << std::endl;

  std::cout << MPI::Wtime()-t0 << std::endl;
  fftw_destroy_plan(rfftn);
  fftw_destroy_plan(irfftn);
  
//
//  Terminate MPI.
//
  MPI::Finalize ( );
}
