#include <assert.h>
#include <stdio.h>
#include "fdtd-2d_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* fdtd-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "fdtd-2d.h"


/* Array initialization. */
static
void init_array (int tmax,
		 int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int i, j;

  for (i = 0; i < tmax; i++)
    _fict_[i] = (DATA_TYPE) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
	ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
	ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
	hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex[i][j]);
    }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey[i][j]);
    }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz[i][j]);
    }
  POLYBENCH_DUMP_END("hz");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_fdtd_2d(int tmax,
		    int nx,
		    int ny,
		    DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int t, i, j;

  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  if (tmax >= 1 && ny >= 1) {
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

    double *dev__fict_;
    double *dev_ex;
    double *dev_ey;
    double *dev_hz;
    
    cudaCheckReturn(cudaMalloc((void **) &dev__fict_, (tmax) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_ex, (nx) * (1200) * sizeof(double)));
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    cudaCheckReturn(cudaMalloc((void **) &dev_ey, (ppcg_max(1, nx)) * (1200) * sizeof(double)));
    cudaCheckReturn(cudaMalloc((void **) &dev_hz, (nx) * (1200) * sizeof(double)));
    
    cudaCheckReturn(cudaMemcpy(dev__fict_, _fict_, (tmax) * sizeof(double), cudaMemcpyHostToDevice));
    if (nx >= 1 && ny >= 2)
      cudaCheckReturn(cudaMemcpy(dev_ex, ex, (nx) * (1200) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_ey, ey, (ppcg_max(1, nx)) * (1200) * sizeof(double), cudaMemcpyHostToDevice));
    if (nx >= 1 && nx + ny >= 3)
      cudaCheckReturn(cudaMemcpy(dev_hz, hz, (nx) * (1200) * sizeof(double), cudaMemcpyHostToDevice));
    for (int c0 = 0; c0 < tmax; c0 += 1) {
      if (nx >= 2)
        {
          dim3 k0_dimBlock(16, 32);
          dim3 k0_dimGrid(ppcg_min(256, (ny + 31) / 32), ppcg_min(256, (nx + 30) / 32));
          kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_ey, dev_hz, tmax, nx, ny, c0);
          cudaCheckKernel();
        }
        
      if (nx >= 1 && ny >= 2)
        {
          dim3 k1_dimBlock(16, 32);
          dim3 k1_dimGrid(ppcg_min(256, (ny + 30) / 32), ppcg_min(256, (nx + 31) / 32));
          kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_ex, dev_hz, tmax, nx, ny, c0);
          cudaCheckKernel();
        }
        
      {
        dim3 k2_dimBlock(16, 32);
        dim3 k2_dimGrid(ppcg_min(256, (ny + 31) / 32), nx >= 34 && ny >= 2 ? ppcg_min(256, (nx + 30) / 32) : 1);
        kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev__fict_, dev_ex, dev_ey, dev_hz, tmax, nx, ny, c0);
        cudaCheckKernel();
      }
      
    }
    if (nx >= 1 && ny >= 2)
      cudaCheckReturn(cudaMemcpy(ex, dev_ex, (nx) * (1200) * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(ey, dev_ey, (ppcg_max(1, nx)) * (1200) * sizeof(double), cudaMemcpyDeviceToHost));
    if (nx >= 1 && nx + ny >= 3)
      cudaCheckReturn(cudaMemcpy(hz, dev_hz, (nx) * (1200) * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaFree(dev__fict_));
    cudaCheckReturn(cudaFree(dev_ex));
    cudaCheckReturn(cudaFree(dev_ey));
    cudaCheckReturn(cudaFree(dev_hz));
  }
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);

  /* Initialize array(s). */
  init_array (tmax, nx, ny,
	      POLYBENCH_ARRAY(ex),
	      POLYBENCH_ARRAY(ey),
	      POLYBENCH_ARRAY(hz),
	      POLYBENCH_ARRAY(_fict_));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_fdtd_2d (tmax, nx, ny,
		  POLYBENCH_ARRAY(ex),
		  POLYBENCH_ARRAY(ey),
		  POLYBENCH_ARRAY(hz),
		  POLYBENCH_ARRAY(_fict_));


  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(ex),
				    POLYBENCH_ARRAY(ey),
				    POLYBENCH_ARRAY(hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(ex);
  POLYBENCH_FREE_ARRAY(ey);
  POLYBENCH_FREE_ARRAY(hz);
  POLYBENCH_FREE_ARRAY(_fict_);

  return 0;
}
