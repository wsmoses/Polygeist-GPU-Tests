#include "syr2k_kernel.hu"
__global__ void kernel0(double *A, double *B, double *C, double alpha, double beta, int n, int m)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_A_0[32][32];
    double private_C[1][2];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * ((b0 - b1 + 256) % 256) + 32 * b1; c0 < n; c0 += 8192)
      for (int c1 = 32 * b1; c1 <= ppcg_min(n - 1, c0 + 31); c1 += 8192) {
        if (b1 <= 37 && n >= t0 + c0 + 1 && t0 + c0 >= 32 * b1 + t1 && c1 == 32 * b1) {
          private_C[0][0] = C[(t0 + c0) * 1200 + (32 * b1 + t1)];
          if (32 * b1 + t1 <= 1183 && t0 + c0 >= 32 * b1 + t1 + 16)
            private_C[0][1] = C[(t0 + c0) * 1200 + (32 * b1 + t1 + 16)];
        }
        for (int c2 = 0; c2 < m; c2 += 32) {
          if (n >= t0 + c1 + 1)
            for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 999); c4 += 16)
              shared_A_0[t0][c4] = A[(t0 + c1) * 1000 + (c2 + c4)];
          __syncthreads();
          if (n >= t0 + c0 + 1 && t0 + c0 >= t1 + c1 && c2 == 0) {
            private_C[0][0] *= beta;
            if (t0 + c0 >= t1 + c1 + 16)
              private_C[0][1] *= beta;
          }
          if (n >= t0 + c0 + 1 && t0 + c0 >= t1 + c1)
            for (int c3 = 0; c3 <= ppcg_min(31, m - c2 - 1); c3 += 1) {
              private_C[0][0] += (((shared_A_0[t1][c3] * alpha) * B[(t0 + c0) * 1000 + (c2 + c3)]) + ((B[(t1 + c1) * 1000 + (c2 + c3)] * alpha) * A[(t0 + c0) * 1000 + (c2 + c3)]));
              if (t0 + c0 >= t1 + c1 + 16)
                private_C[0][1] += (((shared_A_0[t1 + 16][c3] * alpha) * B[(t0 + c0) * 1000 + (c2 + c3)]) + ((B[(t1 + c1 + 16) * 1000 + (c2 + c3)] * alpha) * A[(t0 + c0) * 1000 + (c2 + c3)]));
            }
          __syncthreads();
        }
        if (m <= 0) {
          __syncthreads();
          if (n >= t0 + c0 + 1 && t0 + c0 >= t1 + c1) {
            private_C[0][0] *= beta;
            if (t0 + c0 >= t1 + c1 + 16)
              private_C[0][1] *= beta;
          }
          __syncthreads();
        }
        if (b1 <= 37 && n >= t0 + c0 + 1 && t0 + c0 >= 32 * b1 + t1 && c1 == 32 * b1) {
          C[(t0 + c0) * 1200 + (32 * b1 + t1)] = private_C[0][0];
          if (32 * b1 + t1 <= 1183 && t0 + c0 >= 32 * b1 + t1 + 16)
            C[(t0 + c0) * 1200 + (32 * b1 + t1 + 16)] = private_C[0][1];
        }
        __syncthreads();
      }
}
