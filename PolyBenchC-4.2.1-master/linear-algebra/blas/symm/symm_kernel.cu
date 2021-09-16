#include "symm_kernel.hu"
__global__ void kernel0(double *A, double *B, double *C, double alpha, double beta, int m, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_A_1[32][32];
    double private_temp2;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < m; c0 += 8192) {
      if (m >= 32 * b0 + t0 + 1 && c0 == 32 * b0)
        for (int c2 = t1; c2 <= ppcg_min(31, -32 * b0 + 999); c2 += 16)
          shared_A_1[t0][c2] = A[(32 * b0 + t0) * 1000 + (32 * b0 + c2)];
      __syncthreads();
      if (m >= t0 + c0 + 1)
        for (int c1 = 32 * b1; c1 < n; c1 += 8192)
          for (int c3 = t1; c3 <= ppcg_min(31, n - c1 - 1); c3 += 16) {
            private_temp2 = 0;
            for (int c4 = 0; c4 < t0 + c0; c4 += 1)
              private_temp2 += (B[c4 * 1200 + (c1 + c3)] * A[(t0 + c0) * 1000 + c4]);
            C[(t0 + c0) * 1200 + (c1 + c3)] = (((beta * C[(t0 + c0) * 1200 + (c1 + c3)]) + ((alpha * B[(t0 + c0) * 1200 + (c1 + c3)]) * shared_A_1[t0][t0])) + (alpha * private_temp2));
          }
      __syncthreads();
    }
}
__global__ void kernel1(double *A, double *B, double *C, double alpha, int m, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_A[32][32];
    double private_C[2][1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 8192)
      for (int c1 = 32 * b1; c1 < m - 1; c1 += 8192) {
        if (n >= 32 * b0 + t0 + 1 && 32 * b0 + t0 <= 1199 && c0 == 32 * b0 && m >= t1 + c1 + 2) {
          private_C[0][0] = C[(t1 + c1) * 1200 + (32 * b0 + t0)];
          if (m >= t1 + c1 + 18)
            private_C[1][0] = C[(t1 + c1 + 16) * 1200 + (32 * b0 + t0)];
        }
        for (int c2 = c1; c2 < m - 1; c2 += 32) {
          if (c1 == 32 * b1 && m >= t0 + c2 + 2)
            for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 999); c4 += 16)
              shared_A[t0][c4] = A[(t0 + c2 + 1) * 1000 + (32 * b1 + c4)];
          __syncthreads();
          if (n >= t0 + c0 + 1)
            for (int c3 = ppcg_max(0, t1 + c1 - c2); c3 <= ppcg_min(31, m - c2 - 2); c3 += 1) {
              private_C[0][0] += ((alpha * B[(c2 + c3 + 1) * 1200 + (t0 + c0)]) * shared_A[c3][t1]);
              if (c2 + c3 >= t1 + c1 + 16)
                private_C[1][0] += ((alpha * B[(c2 + c3 + 1) * 1200 + (t0 + c0)]) * shared_A[c3][t1 + 16]);
            }
          __syncthreads();
        }
        if (n >= 32 * b0 + t0 + 1 && 32 * b0 + t0 <= 1199 && c0 == 32 * b0 && m >= t1 + c1 + 2) {
          C[(t1 + c1) * 1200 + (32 * b0 + t0)] = private_C[0][0];
          if (m >= t1 + c1 + 18)
            C[(t1 + c1 + 16) * 1200 + (32 * b0 + t0)] = private_C[1][0];
        }
        __syncthreads();
      }
}
