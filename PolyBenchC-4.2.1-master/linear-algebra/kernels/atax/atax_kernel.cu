#include "atax_kernel.hu"
__global__ void kernel0(double *A, double *tmp, double *x, int m, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ double shared_A[32][32];
    double private_tmp[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < m; c0 += 1048576) {
      if (n >= 1)
        for (int c2 = 0; c2 <= ppcg_min(31, m - c0 - 1); c2 += 1)
          shared_A[c2][t0] = A[(c0 + c2) * 2100 + t0];
      __syncthreads();
      if (m >= t0 + c0 + 1) {
        private_tmp[0] = 0.;
        for (int c3 = 0; c3 <= ppcg_min(31, n - 1); c3 += 1)
          private_tmp[0] = (private_tmp[0] + (shared_A[t0][c3] * x[c3]));
      }
      __syncthreads();
      for (int c1 = 32; c1 < n; c1 += 32) {
        if (t0 + c1 <= 2099)
          for (int c2 = 0; c2 <= ppcg_min(31, m - c0 - 1); c2 += 1)
            shared_A[c2][t0] = A[(c0 + c2) * 2100 + (t0 + c1)];
        __syncthreads();
        if (m >= t0 + c0 + 1)
          for (int c3 = 0; c3 <= ppcg_min(31, n - c1 - 1); c3 += 1)
            private_tmp[0] = (private_tmp[0] + (shared_A[t0][c3] * x[c1 + c3]));
        __syncthreads();
      }
      if (m >= t0 + c0 + 1)
        tmp[t0 + c0] = private_tmp[0];
      __syncthreads();
    }
}
__global__ void kernel1(double *A, double *tmp, double *y, int m, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ double shared_tmp[32];
    double private_y[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 1048576) {
      for (int c1 = 0; c1 < m; c1 += 32) {
        if (m >= t0 + c1 + 1)
          shared_tmp[t0] = tmp[t0 + c1];
        __syncthreads();
        if (n >= t0 + c0 + 1 && c1 == 0)
          private_y[0] = 0;
        if (n >= t0 + c0 + 1)
          for (int c3 = 0; c3 <= ppcg_min(31, m - c1 - 1); c3 += 1)
            private_y[0] = (private_y[0] + (A[(c1 + c3) * 2100 + (t0 + c0)] * shared_tmp[c3]));
        __syncthreads();
      }
      if (m <= 0) {
        __syncthreads();
        if (n >= t0 + c0 + 1)
          private_y[0] = 0;
        __syncthreads();
      }
      if (n >= t0 + c0 + 1)
        y[t0 + c0] = private_y[0];
      __syncthreads();
    }
}
