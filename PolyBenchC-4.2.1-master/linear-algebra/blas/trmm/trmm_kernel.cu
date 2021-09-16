#include "trmm_kernel.hu"
__global__ void kernel0(double *A, double *B, int m, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ double shared_A[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 1048576)
      for (int c1 = 0; c1 < m - 1; c1 += 32)
        for (int c2 = 0; c2 <= ppcg_min(m - 2, c1 + 31); c2 += 32) {
          if (t0 + c2 <= 999)
            for (int c3 = 0; c3 <= ppcg_min(31, m - c1 - 2); c3 += 1)
              shared_A[c3][t0] = A[(c1 + c3 + 1) * 1000 + (t0 + c2)];
          __syncthreads();
          if (n >= t0 + c0 + 1)
            for (int c4 = 0; c4 <= ppcg_min(31, m - c1 - 2); c4 += 1)
              for (int c5 = 0; c5 <= ppcg_min(31, c1 - c2 + c4); c5 += 1)
                B[(c2 + c5) * 1200 + (t0 + c0)] += (shared_A[c4][c5] * B[(c1 + c4 + 1) * 1200 + (t0 + c0)]);
          __syncthreads();
        }
}
__global__ void kernel1(double *B, double alpha, int m, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < m; c0 += 8192)
      if (m >= t0 + c0 + 1)
        for (int c1 = 32 * b1; c1 < n; c1 += 8192)
          for (int c3 = t1; c3 <= ppcg_min(31, n - c1 - 1); c3 += 16)
            B[(t0 + c0) * 1200 + (c1 + c3)] = (alpha * B[(t0 + c0) * 1200 + (c1 + c3)]);
}
