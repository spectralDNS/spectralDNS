#include "fftw3-mpi.h"
#include <math.h>
#include <iostream> 
#include <iomanip>
#include <complex>
#include <vector>

using namespace std;

//mpic++ -std=c++11 -O3 spectralDNS.cpp -o spectralDNS -lfftw3_mpi -lfftw3
//mpixlcxx_r -qsmp -O3 spectralDNS.cpp -o spectralDNS $FFTW3_INC $FFTW3_LIB

// To change from float to double replace all fftw_ with fftw_ and vice versa. + change linked libraries.
typedef double precision;

int main( int argc, char *argv[] )
{
  int rank, num_processes, M, N, Np, Nf;
  double wtime, L, dx;
  precision nu, dt, T;
  double t0, t1, fastest_time, slowest_time, start_time;
  MPI::Init ( argc, argv );
  fftw_mpi_init();
    
  num_processes = MPI::COMM_WORLD.Get_size();
  rank = MPI::COMM_WORLD.Get_rank();
  precision pi = 3.141592653589793238;
  ptrdiff_t alloc_local, local_n0, local_0_start, local_n1, local_1_start, i, j, k;
  vector<double> s_in(1), s_out(1), vs_in(2), vs_out(2);  

  nu = 0.000625;
  T = 0.1;
  dt = 0.01;
  M = 7;  
  if ( argc > 1 ) {
    M = atoi( argv[1] );
  }
  
//   N = pow(static_cast<int>(2), M); // Not accepted by Shaheen
  N = 1;
  for (int i=0; i<M;i++)
      N *= 2;
  L = 2*pi;
  Np = N / num_processes;
  dx = L / N;
  std::cout << std::scientific << std::setprecision(16);
  if (rank==0)
      std::cout << "N = " << N << std::endl;
  vector<precision> a(4);
  a[0] = 1./6.; a[1] = 1./3.; a[2] = 1./3.; a[3] = 1./6.;
  vector<precision> b(3);
  b[0] = 0.5; b[1] = 0.5; b[2] = 1.0;
  int tot = N*N*N;
  Nf = N/2+1;
  alloc_local = fftw_mpi_local_size_3d_transposed(N, N, Nf, MPI::COMM_WORLD,
                                        &local_n0, &local_0_start,
                                        &local_n1, &local_1_start);
  
  vector<precision> U(2*alloc_local);
  vector<precision> V(2*alloc_local);
  vector<precision> W(2*alloc_local);
  vector<precision> U_tmp(2*alloc_local);
  vector<precision> V_tmp(2*alloc_local);
  vector<precision> W_tmp(2*alloc_local);
  vector<precision> CU(2*alloc_local);
  vector<precision> CV(2*alloc_local);
  vector<precision> CW(2*alloc_local);  
  vector<precision> P(2*alloc_local);
  vector<int> dealias(2*alloc_local);
  vector<precision> kk(2*alloc_local);
  vector<complex<precision> > U_hat(alloc_local);
  vector<complex<precision> > V_hat(alloc_local);
  vector<complex<precision> > W_hat(alloc_local);
  vector<complex<precision> > P_hat(alloc_local);
  vector<complex<precision> > U_hat0(alloc_local);
  vector<complex<precision> > V_hat0(alloc_local);
  vector<complex<precision> > W_hat0(alloc_local);
  vector<complex<precision> > U_hat1(alloc_local);
  vector<complex<precision> > V_hat1(alloc_local);
  vector<complex<precision> > W_hat1(alloc_local);
  vector<complex<precision> > dU(alloc_local);
  vector<complex<precision> > dV(alloc_local);
  vector<complex<precision> > dW(alloc_local);  
  vector<complex<precision> > curlX(alloc_local);
  vector<complex<precision> > curlY(alloc_local);
  vector<complex<precision> > curlZ(alloc_local);
  
  // Starting time
  MPI::COMM_WORLD.Barrier();
  t0 = MPI::Wtime();
  start_time = t0;

  vector<precision> kx(N);
  vector<precision> kz(Nf);
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
        const int z = (i*N+j)*2*Nf+k;
        U[z] = sin(dx*(i+local_0_start))*cos(dx*j)*cos(dx*k);
        V[z] = -cos(dx*(i+local_0_start))*sin(dx*j)*cos(dx*k);
        W[z] = 0.0;
      }
    
  fftw_mpi_execute_dft_r2c( rfftn, U.data(), reinterpret_cast<fftw_complex*>(U_hat.data()));
  fftw_mpi_execute_dft_r2c( rfftn, V.data(), reinterpret_cast<fftw_complex*>(V_hat.data()));
  fftw_mpi_execute_dft_r2c( rfftn, W.data(), reinterpret_cast<fftw_complex*>(W_hat.data()));
    
  precision kmax = 2./3.*(N/2+1);  
  for (int i=0; i<local_n1; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<Nf; k++)
      {
        const int z = (i*N+j)*Nf+k;             
        dealias[z] = (abs(kx[i+local_1_start])<kmax)*(abs(kx[j])<kmax)*(abs(kx[k])<kmax) == true ? 1 : 0;      
      }
        
  for (int i=0; i<local_n1; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<Nf; k++)
      {
        const int z = (i*N+j)*Nf+k;
        int m = kx[i+local_1_start]*kx[i+local_1_start] + kx[j]*kx[j] + kx[k]*kx[k];
        kk[z] = m > 0 ? m : 1;
      }       
  
  complex<precision> one(0, 1); 
  double t=0.0;
  int tstep = 0;
  fastest_time = 1e8;
  slowest_time = 0;
  while (t < T-1e-8)
  {
     t += dt;
     tstep++;
     for (int i=0; i<local_n1; i++)
       for (int j=0; j<N; j++)
         for (int k=0; k<Nf; k++)
         {
             const int z = (i*N+j)*Nf+k;
             U_hat0[z] = U_hat[z];
             V_hat0[z] = V_hat[z];
             W_hat0[z] = W_hat[z];
             U_hat1[z] = U_hat[z];
             V_hat1[z] = V_hat[z];
             W_hat1[z] = W_hat[z];
        }
//      std::copy(U_hat.begin(), U_hat.end(), U_hat0.begin());
//      std::copy(V_hat.begin(), V_hat.end(), V_hat0.begin());
//      std::copy(W_hat.begin(), W_hat.end(), W_hat0.begin());
//      std::copy(U_hat.begin(), U_hat.end(), U_hat1.begin());
//      std::copy(U_hat.begin(), U_hat.end(), U_hat1.begin());
//      std::copy(U_hat.begin(), U_hat.end(), U_hat1.begin());
     
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
               const int z = (i*N+j)*Nf+k;
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
              const int z = (i*N+j)*2*Nf+k;
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
              const int z = (i*N+j)*Nf+k;
              dU[z] *= (dealias[z]*dt);
              dV[z] *= (dealias[z]*dt);
              dW[z] *= (dealias[z]*dt);
            }
        // 
        for (int i=0; i<local_n1; i++)
          for (int j=0; j<N; j++)
            for (int k=0; k<Nf; k++)
            {
                const int z = (i*N+j)*Nf+k;
                P_hat[z] = (dU[z]*kx[i+local_1_start] + dV[z]*kx[j] + dW[z]*kz[k])/kk[z];
                dU[z] -= (P_hat[z]*kx[i+local_1_start] + nu*dt*kk[z]*U_hat[z]);
                dV[z] -= (P_hat[z]*kx[j] + nu*dt*kk[z]*V_hat[z]);
                dW[z] -= (P_hat[z]*kz[k] + nu*dt*kk[z]*W_hat[z]);
            }
            
        if (rk < 3)
        {
          for (int i=0; i<local_n1; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<Nf; k++)
                {
                   const int z = (i*N+j)*Nf+k;
                   U_hat[z] = U_hat0[z] + b[rk]*dU[z];
                   V_hat[z] = V_hat0[z] + b[rk]*dV[z];
                   W_hat[z] = W_hat0[z] + b[rk]*dW[z];
                }
            
        }
        for (int i=0; i<local_n1; i++)
          for (int j=0; j<N; j++)
            for (int k=0; k<Nf; k++)
            {
                const int z = (i*N+j)*Nf+k;
                U_hat1[z] += a[rk]*dU[z];
                V_hat1[z] += a[rk]*dV[z];
                W_hat1[z] += a[rk]*dW[z];
            }        
     }     
    for (int i=0; i<local_n1; i++)
      for (int j=0; j<N; j++)
        for (int k=0; k<Nf; k++)
        {
            const int z = (i*N+j)*Nf+k;
            U_hat[z] = U_hat1[z];
            V_hat[z] = V_hat1[z];
            W_hat[z] = W_hat1[z];
        }     
        
    if (tstep % 2 == 0)
    {
        s_in[0] = 0.0;
        for (int i=0; i<local_n0; i++)
            for (int j=0; j<N; j++)
            for (int k=0; k<N; k++)
            {
                int z = (i*N+j)*2*Nf+k;
                s_in[0] += (U[z]*U[z] + V[z]*V[z] + W[z]*W[z]);
            }
        s_in[0] *= (0.5*dx*dx*dx/L/L/L);

        MPI::COMM_WORLD.Reduce(s_in.data(), s_out.data(), 1, MPI::DOUBLE, MPI::SUM, 0);  
        if (rank==0)
        std::cout << " k = " << s_out[0] << std::endl;
    }        
        
    t1 = MPI::Wtime();
    if (tstep > 1)
    {
        fastest_time = min(t1-t0, fastest_time);
        slowest_time = max(t1-t0, slowest_time);
    }   
    t0 = t1;
}

  s_in[0] = 0.0;
  for (int i=0; i<local_n0; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
      {
        int z = (i*N+j)*2*Nf+k;
        s_in[0] += (U[z]*U[z] + V[z]*V[z] + W[z]*W[z]);
      }
  s_in[0] *= (0.5*dx*dx*dx/L/L/L);

  MPI::COMM_WORLD.Reduce(s_in.data(), s_out.data(), 1, MPI::DOUBLE, MPI::SUM, 0);  
  if (rank==0)
    std::cout << " k = " << s_out[0] << std::endl;

  MPI::COMM_WORLD.Barrier();
  t1 = MPI::Wtime();
  if (rank == 0)  
    std::cout << "Time = " << t1 - start_time  << std::endl;
  
  fftw_destroy_plan(rfftn);
  fftw_destroy_plan(irfftn);
  vs_in[0] = fastest_time;
  vs_in[1] = slowest_time;
  MPI::COMM_WORLD.Reduce(vs_in.data(), vs_out.data(), 2, MPI::DOUBLE, MPI::MIN, 0);
  if (rank==0)
      std::cout << "Fastest = " << vs_out[0] << ", " << vs_out[1] << std::endl; 
  MPI::COMM_WORLD.Reduce(vs_in.data(), vs_out.data(), 2, MPI::DOUBLE, MPI::MAX, 0);
  if (rank==0)
      std::cout << "Slowest = " << vs_out[0] << ", " << vs_out[1] << std::endl; 
  
  MPI::Finalize ( );
}
