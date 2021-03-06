#include "heat-3d_kernel.hu"
__global__ void kernel0(double *A, double *B, int n, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.z, t1 = threadIdx.y, t2 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 32 * b0; c1 < n - 2; c1 += 8192)
      if (n >= t0 + c1 + 3)
        for (int c2 = 32 * b1; c2 < n - 2; c2 += 8192)
          for (int c3 = 0; c3 < n - 2; c3 += 32)
            for (int c5 = t1; c5 <= ppcg_min(31, n - c2 - 3); c5 += 4)
              for (int c6 = t2; c6 <= ppcg_min(31, n - c3 - 3); c6 += 4)
                B[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)] = ((((0.125 * ((A[((t0 + c1 + 2) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)] - (2. * A[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)])) + A[((t0 + c1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)])) + (0.125 * ((A[((t0 + c1 + 1) * 120 + (c2 + c5 + 2)) * 120 + (c3 + c6 + 1)] - (2. * A[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)])) + A[((t0 + c1 + 1) * 120 + (c2 + c5)) * 120 + (c3 + c6 + 1)]))) + (0.125 * ((A[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 2)] - (2. * A[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)])) + A[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6)]))) + A[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)]);
}
__global__ void kernel1(double *A, double *B, int n, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.z, t1 = threadIdx.y, t2 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 32 * b0; c1 < n - 2; c1 += 8192)
      if (n >= t0 + c1 + 3)
        for (int c2 = 32 * b1; c2 < n - 2; c2 += 8192)
          for (int c3 = 0; c3 < n - 2; c3 += 32)
            for (int c5 = t1; c5 <= ppcg_min(31, n - c2 - 3); c5 += 4)
              for (int c6 = t2; c6 <= ppcg_min(31, n - c3 - 3); c6 += 4)
                A[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)] = ((((0.125 * ((B[((t0 + c1 + 2) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)] - (2. * B[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)])) + B[((t0 + c1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)])) + (0.125 * ((B[((t0 + c1 + 1) * 120 + (c2 + c5 + 2)) * 120 + (c3 + c6 + 1)] - (2. * B[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)])) + B[((t0 + c1 + 1) * 120 + (c2 + c5)) * 120 + (c3 + c6 + 1)]))) + (0.125 * ((B[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 2)] - (2. * B[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)])) + B[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6)]))) + B[((t0 + c1 + 1) * 120 + (c2 + c5 + 1)) * 120 + (c3 + c6 + 1)]);
}
