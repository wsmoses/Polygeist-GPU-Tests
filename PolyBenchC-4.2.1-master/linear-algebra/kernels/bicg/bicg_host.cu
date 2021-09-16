#include <assert.h>
#include <stdio.h>
#include "bicg_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* bicg.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "bicg.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (DATA_TYPE)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (DATA_TYPE)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, s[i]);
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, q[i]);
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_bicg(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  if (m >= 1 || n >= 1) {
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
    double *dev_p;
    double *dev_q;
    double *dev_r;
    double *dev_s;
    
    cudaCheckReturn(cudaMalloc((void **) &dev_A, (n) * (1900) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_p, (m) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_q, (n) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_r, (n) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_s, (m) * sizeof(double)));
    
    if (n >= 1 && m >= 1) {
      cudaCheckReturn(cudaMemcpy(dev_A, A, (n) * (1900) * sizeof(double), cudaMemcpyHostToDevice));
      cudaCheckReturn(cudaMemcpy(dev_p, p, (m) * sizeof(double), cudaMemcpyHostToDevice));
      cudaCheckReturn(cudaMemcpy(dev_r, r, (n) * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (m >= 1)
      {
        dim3 k0_dimBlock(32);
        dim3 k0_dimGrid(ppcg_min(32768, (m + 31) / 32));
        kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, dev_r, dev_s, n, m);
        cudaCheckKernel();
      }
      
    if (n >= 1) {
      {
        dim3 k1_dimBlock(32);
        dim3 k1_dimGrid(ppcg_min(32768, (n + 31) / 32));
        kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_A, dev_p, dev_q, n, m);
        cudaCheckKernel();
      }
      
      cudaCheckReturn(cudaMemcpy(q, dev_q, (n) * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (m >= 1)
      cudaCheckReturn(cudaMemcpy(s, dev_s, (m) * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaFree(dev_A));
    cudaCheckReturn(cudaFree(dev_p));
    cudaCheckReturn(cudaFree(dev_q));
    cudaCheckReturn(cudaFree(dev_r));
    cudaCheckReturn(cudaFree(dev_s));
  }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(r),
	      POLYBENCH_ARRAY(p));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_bicg (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(s),
	       POLYBENCH_ARRAY(q),
	       POLYBENCH_ARRAY(p),
	       POLYBENCH_ARRAY(r));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

  return 0;
}
