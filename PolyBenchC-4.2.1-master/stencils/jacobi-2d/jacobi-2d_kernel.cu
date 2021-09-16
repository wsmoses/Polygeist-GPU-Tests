#include "jacobi-2d_kernel.hu"
__global__ void kernel0(double *A, double *B, int tsteps, int n, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 32 * b0; c1 < n - 2; c1 += 8192)
      if (n >= t0 + c1 + 3)
        for (int c2 = 32 * b1; c2 < n - 2; c2 += 8192)
          for (int c4 = t1; c4 <= ppcg_min(31, n - c2 - 3); c4 += 16)
            B[(t0 + c1 + 1) * 1300 + (c2 + c4 + 1)] = (0.20000000000000001 * ((((A[(t0 + c1 + 1) * 1300 + (c2 + c4 + 1)] + A[(t0 + c1 + 1) * 1300 + (c2 + c4)]) + A[(t0 + c1 + 1) * 1300 + (c2 + c4 + 2)]) + A[(t0 + c1 + 2) * 1300 + (c2 + c4 + 1)]) + A[(t0 + c1) * 1300 + (c2 + c4 + 1)]));
}
__global__ void kernel1(double *A, double *B, int tsteps, int n, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 32 * b0; c1 < n - 2; c1 += 8192)
      if (n >= t0 + c1 + 3)
        for (int c2 = 32 * b1; c2 < n - 2; c2 += 8192)
          for (int c4 = t1; c4 <= ppcg_min(31, n - c2 - 3); c4 += 16)
            A[(t0 + c1 + 1) * 1300 + (c2 + c4 + 1)] = (0.20000000000000001 * ((((B[(t0 + c1 + 1) * 1300 + (c2 + c4 + 1)] + B[(t0 + c1 + 1) * 1300 + (c2 + c4)]) + B[(t0 + c1 + 1) * 1300 + (c2 + c4 + 2)]) + B[(t0 + c1 + 2) * 1300 + (c2 + c4 + 1)]) + B[(t0 + c1) * 1300 + (c2 + c4 + 1)]));
}
