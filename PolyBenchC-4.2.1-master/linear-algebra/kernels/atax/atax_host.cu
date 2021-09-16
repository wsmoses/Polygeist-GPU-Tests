#include <assert.h>
#include <stdio.h>
#include "atax_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* atax.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_atax(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  if (n >= 1 || m >= 1) {
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

    double *dev_A;
    double *dev_tmp;
    double *dev_x;
    double *dev_y;
    
    cudaCheckReturn(cudaMalloc((void **) &dev_A, (m) * (2100) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_tmp, (m) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_x, (n) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_y, (n) * sizeof(double)));
    
    if (m >= 1 && n >= 1) {
      cudaCheckReturn(cudaMemcpy(dev_A, A, (m) * (2100) * sizeof(double), cudaMemcpyHostToDevice));
      cudaCheckReturn(cudaMemcpy(dev_x, x, (n) * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (m >= 1)
      {
        dim3 k0_dimBlock(32);
        dim3 k0_dimGrid(ppcg_min(32768, (m + 31) / 32));
        kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, dev_tmp, dev_x, m, n);
        cudaCheckKernel();
      }
      
    if (n >= 1)
      {
        dim3 k1_dimBlock(32);
        dim3 k1_dimGrid(ppcg_min(32768, (n + 31) / 32));
        kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_A, dev_tmp, dev_y, m, n);
        cudaCheckKernel();
      }
      
    if (m >= 1)
      cudaCheckReturn(cudaMemcpy(tmp, dev_tmp, (m) * sizeof(double), cudaMemcpyDeviceToHost));
    if (n >= 1)
      cudaCheckReturn(cudaMemcpy(y, dev_y, (n) * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaFree(dev_A));
    cudaCheckReturn(cudaFree(dev_tmp));
    cudaCheckReturn(cudaFree(dev_x));
    cudaCheckReturn(cudaFree(dev_y));
  }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_atax (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(x),
	       POLYBENCH_ARRAY(y),
	       POLYBENCH_ARRAY(tmp));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}
