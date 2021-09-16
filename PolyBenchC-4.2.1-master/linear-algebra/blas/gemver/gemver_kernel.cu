#include "gemver_kernel.hu"
__global__ void kernel0(double *A, double *u1, double *u2, double *v1, double *v2, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_u1[32];
    __shared__ double shared_u2[32];
    __shared__ double shared_v1[32];
    __shared__ double shared_v2[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 8192) {
      if (t0 == 0) {
        for (int c1 = t1; c1 <= ppcg_min(31, n - c0 - 1); c1 += 16)
          shared_u1[c1] = u1[c0 + c1];
        for (int c1 = t1; c1 <= ppcg_min(31, n - c0 - 1); c1 += 16)
          shared_u2[c1] = u2[c0 + c1];
      }
      __syncthreads();
      for (int c1 = 32 * b1; c1 < n; c1 += 8192) {
        if (t0 == 0) {
          for (int c2 = t1; c2 <= ppcg_min(31, n - c1 - 1); c2 += 16)
            shared_v1[c2] = v1[c1 + c2];
          for (int c2 = t1; c2 <= ppcg_min(31, n - c1 - 1); c2 += 16)
            shared_v2[c2] = v2[c1 + c2];
        }
        __syncthreads();
        if (n >= t0 + c0 + 1)
          for (int c3 = t1; c3 <= ppcg_min(31, n - c1 - 1); c3 += 16)
            A[(t0 + c0) * 2000 + (c1 + c3)] = ((A[(t0 + c0) * 2000 + (c1 + c3)] + (shared_u1[t0] * shared_v1[c3])) + (shared_u2[t0] * shared_v2[c3]));
        __syncthreads();
      }
      __syncthreads();
    }
}
__global__ void kernel1(double *A, double beta, double *x, double *y, double *z, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    double private_x[1];
    __shared__ double shared_y[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 1048576) {
      if (n >= t0 + c0 + 1)
        private_x[0] = x[t0 + c0];
      for (int c1 = 0; c1 <= n; c1 += 32) {
        if (n >= t0 + c1 + 1)
          shared_y[t0] = y[t0 + c1];
        __syncthreads();
        if (n >= t0 + c0 + 1) {
          for (int c3 = 0; c3 <= ppcg_min(31, n - c1 - 1); c3 += 1)
            private_x[0] = (private_x[0] + ((beta * A[(c1 + c3) * 2000 + (t0 + c0)]) * shared_y[c3]));
          if (c1 + 31 >= n)
            private_x[0] = (private_x[0] + z[t0 + c0]);
        }
        __syncthreads();
      }
      if (n >= t0 + c0 + 1)
        x[t0 + c0] = private_x[0];
      __syncthreads();
    }
}
__global__ void kernel2(double *A, double alpha, double *w, double *x, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ double shared_A[32][32];
    double private_w[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 1048576) {
      if (n >= t0 + c0 + 1)
        private_w[0] = w[t0 + c0];
      for (int c1 = 0; c1 < n; c1 += 32) {
        if (t0 + c1 <= 1999)
          for (int c2 = 0; c2 <= ppcg_min(31, n - c0 - 1); c2 += 1)
            shared_A[c2][t0] = A[(c0 + c2) * 2000 + (t0 + c1)];
        __syncthreads();
        if (n >= t0 + c0 + 1)
          for (int c3 = 0; c3 <= ppcg_min(31, n - c1 - 1); c3 += 1)
            private_w[0] = (private_w[0] + ((alpha * shared_A[t0][c3]) * x[c1 + c3]));
        __syncthreads();
      }
      if (n >= t0 + c0 + 1)
        w[t0 + c0] = private_w[0];
      __syncthreads();
    }
}
