#include <assert.h>
#include <stdio.h>
#include "covariance_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* covariance.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "covariance.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)n;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(cov,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, cov[i][j]);
    }
  POLYBENCH_DUMP_END("cov");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_covariance(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;

  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  if (m >= 1) {
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

    double *dev_cov;
    double *dev_data;
    double *dev_mean;
    
    cudaCheckReturn(cudaMalloc((void **) &dev_cov, (m) * (1200) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_data, (n) * (1200) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_mean, (m) * sizeof(double)));
    
    cudaCheckReturn(cudaMemcpy(dev_cov, cov, (m) * (1200) * sizeof(double), cudaMemcpyHostToDevice));
    if (n >= 1)
      cudaCheckReturn(cudaMemcpy(dev_data, data, (n) * (1200) * sizeof(double), cudaMemcpyHostToDevice));
    {
      dim3 k0_dimBlock(32);
      dim3 k0_dimGrid(ppcg_min(32768, (m + 31) / 32));
      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_data, dev_mean, m, n);
      cudaCheckKernel();
    }
    
    {
      dim3 k1_dimBlock(16, 32);
      dim3 k1_dimGrid(ppcg_min(256, (m + 31) / 32), ppcg_min(256, (m + 31) / 32));
      kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_cov, m, n);
      cudaCheckKernel();
    }
    
    {
      dim3 k2_dimBlock(32);
      dim3 k2_dimGrid(ppcg_min(32768, (m + 31) / 32));
      kernel2 <<<k2_dimGrid, k2_dimBlock>>> (float_n, dev_mean, m, n);
      cudaCheckKernel();
    }
    
    if (n >= 1)
      {
        dim3 k3_dimBlock(16, 32);
        dim3 k3_dimGrid(ppcg_min(256, (m + 31) / 32), ppcg_min(256, (n + 31) / 32));
        kernel3 <<<k3_dimGrid, k3_dimBlock>>> (dev_data, dev_mean, m, n);
        cudaCheckKernel();
      }
      
    {
      dim3 k4_dimBlock(16, 32);
      dim3 k4_dimGrid(ppcg_min(256, (m + 31) / 32), ppcg_min(256, (m + 31) / 32));
      kernel4 <<<k4_dimGrid, k4_dimBlock>>> (dev_cov, dev_data, float_n, m, n);
      cudaCheckKernel();
    }
    
    cudaCheckReturn(cudaMemcpy(cov, dev_cov, (m) * (1200) * sizeof(double), cudaMemcpyDeviceToHost));
    if (n >= 1)
      cudaCheckReturn(cudaMemcpy(data, dev_data, (n) * (1200) * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(mean, dev_mean, (m) * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaFree(dev_cov));
    cudaCheckReturn(cudaFree(dev_data));
    cudaCheckReturn(cudaFree(dev_mean));
  }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(cov,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);


  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_covariance (m, n, float_n,
		     POLYBENCH_ARRAY(data),
		     POLYBENCH_ARRAY(cov),
		     POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(cov)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(cov);
  POLYBENCH_FREE_ARRAY(mean);

  return 0;
}
