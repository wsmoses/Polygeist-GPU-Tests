#include <assert.h>
#include <stdio.h>
#include "gemver_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver(int n,
		   DATA_TYPE alpha,
		   DATA_TYPE beta,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(u1,N,n),
		   DATA_TYPE POLYBENCH_1D(v1,N,n),
		   DATA_TYPE POLYBENCH_1D(u2,N,n),
		   DATA_TYPE POLYBENCH_1D(v2,N,n),
		   DATA_TYPE POLYBENCH_1D(w,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n),
		   DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  if (n >= 1) {
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
    double *dev_u1;
    double *dev_u2;
    double *dev_v1;
    double *dev_v2;
    double *dev_w;
    double *dev_x;
    double *dev_y;
    double *dev_z;
    
    cudaCheckReturn(cudaMalloc((void **) &dev_A, (n) * (2000) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_u1, (n) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_u2, (n) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_v1, (n) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_v2, (n) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_w, (n) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_x, (n) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_y, (n) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_z, (n) * sizeof(double)));
    
    cudaCheckReturn(cudaMemcpy(dev_A, A, (n) * (2000) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_u1, u1, (n) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_u2, u2, (n) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_v1, v1, (n) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_v2, v2, (n) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_w, w, (n) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_x, x, (n) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_y, y, (n) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_z, z, (n) * sizeof(double), cudaMemcpyHostToDevice));
    {
      dim3 k0_dimBlock(16, 32);
      dim3 k0_dimGrid(ppcg_min(256, (n + 31) / 32), ppcg_min(256, (n + 31) / 32));
      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, dev_u1, dev_u2, dev_v1, dev_v2, n);
      cudaCheckKernel();
    }
    
    {
      dim3 k1_dimBlock(32);
      dim3 k1_dimGrid(ppcg_min(32768, (n + 31) / 32));
      kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_A, beta, dev_x, dev_y, dev_z, n);
      cudaCheckKernel();
    }
    
    {
      dim3 k2_dimBlock(32);
      dim3 k2_dimGrid(ppcg_min(32768, (n + 31) / 32));
      kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_A, alpha, dev_w, dev_x, n);
      cudaCheckKernel();
    }
    
    cudaCheckReturn(cudaMemcpy(A, dev_A, (n) * (2000) * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(w, dev_w, (n) * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(x, dev_x, (n) * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaFree(dev_A));
    cudaCheckReturn(cudaFree(dev_u1));
    cudaCheckReturn(cudaFree(dev_u2));
    cudaCheckReturn(cudaFree(dev_v1));
    cudaCheckReturn(cudaFree(dev_v2));
    cudaCheckReturn(cudaFree(dev_w));
    cudaCheckReturn(cudaFree(dev_x));
    cudaCheckReturn(cudaFree(dev_y));
    cudaCheckReturn(cudaFree(dev_z));
  }
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemver (n, alpha, beta,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(u1),
		 POLYBENCH_ARRAY(v1),
		 POLYBENCH_ARRAY(u2),
		 POLYBENCH_ARRAY(v2),
		 POLYBENCH_ARRAY(w),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y),
		 POLYBENCH_ARRAY(z));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);

  return 0;
}
