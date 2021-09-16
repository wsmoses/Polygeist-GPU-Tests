#include "mvt_kernel.hu"
__global__ void kernel0(double *A, double *x2, double *y_2, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    double private_x2[1];
    __shared__ double shared_y_2[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 1048576) {
      if (n >= t0 + c0 + 1)
        private_x2[0] = x2[t0 + c0];
      for (int c1 = 0; c1 < n; c1 += 32) {
        if (n >= t0 + c1 + 1)
          shared_y_2[t0] = y_2[t0 + c1];
        __syncthreads();
        if (n >= t0 + c0 + 1)
          for (int c3 = 0; c3 <= ppcg_min(31, n - c1 - 1); c3 += 1)
            private_x2[0] = (private_x2[0] + (A[(c1 + c3) * 2000 + (t0 + c0)] * shared_y_2[c3]));
        __syncthreads();
      }
      if (n >= t0 + c0 + 1)
        x2[t0 + c0] = private_x2[0];
      __syncthreads();
    }
}
__global__ void kernel1(double *A, double *x1, double *y_1, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ double shared_A[32][32];
    double private_x1[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 1048576) {
      if (n >= t0 + c0 + 1)
        private_x1[0] = x1[t0 + c0];
      for (int c1 = 0; c1 < n; c1 += 32) {
        if (t0 + c1 <= 1999)
          for (int c2 = 0; c2 <= ppcg_min(31, n - c0 - 1); c2 += 1)
            shared_A[c2][t0] = A[(c0 + c2) * 2000 + (t0 + c1)];
        __syncthreads();
        if (n >= t0 + c0 + 1)
          for (int c3 = 0; c3 <= ppcg_min(31, n - c1 - 1); c3 += 1)
            private_x1[0] = (private_x1[0] + (shared_A[t0][c3] * y_1[c1 + c3]));
        __syncthreads();
      }
      if (n >= t0 + c0 + 1)
        x1[t0 + c0] = private_x1[0];
      __syncthreads();
    }
}
