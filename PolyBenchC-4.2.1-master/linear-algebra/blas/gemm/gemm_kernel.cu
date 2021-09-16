#include "gemm_kernel.hu"
__global__ void kernel0(double *A, double *B, double *C, double alpha, double beta, int ni, int nk, int nj)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_A[32][32];
    double private_C[1][2];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < ni; c0 += 8192)
      for (int c1 = 32 * b1; c1 < nj; c1 += 8192) {
        if (nj >= 32 * b1 + t1 + 1 && 32 * b1 + t1 <= 1099 && ni >= t0 + c0 + 1 && c1 == 32 * b1) {
          private_C[0][0] = C[(t0 + c0) * 1100 + (32 * b1 + t1)];
          if (nj >= 32 * b1 + t1 + 17 && 32 * b1 + t1 <= 1083)
            private_C[0][1] = C[(t0 + c0) * 1100 + (32 * b1 + t1 + 16)];
        }
        for (int c2 = 0; c2 < nk; c2 += 32) {
          if (ni >= t0 + c0 + 1)
            for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 1199); c4 += 16)
              shared_A[t0][c4] = A[(t0 + c0) * 1200 + (c2 + c4)];
          __syncthreads();
          if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1 && c2 == 0) {
            private_C[0][0] *= beta;
            if (nj >= t1 + c1 + 17)
              private_C[0][1] *= beta;
          }
          if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1)
            for (int c3 = 0; c3 <= ppcg_min(31, nk - c2 - 1); c3 += 1) {
              private_C[0][0] += ((alpha * shared_A[t0][c3]) * B[(c2 + c3) * 1100 + (t1 + c1)]);
              if (nj >= t1 + c1 + 17)
                private_C[0][1] += ((alpha * shared_A[t0][c3]) * B[(c2 + c3) * 1100 + (t1 + c1 + 16)]);
            }
          __syncthreads();
        }
        if (nk <= 0) {
          __syncthreads();
          if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1) {
            private_C[0][0] *= beta;
            if (nj >= t1 + c1 + 17)
              private_C[0][1] *= beta;
          }
          __syncthreads();
        }
        if (nj >= 32 * b1 + t1 + 1 && 32 * b1 + t1 <= 1099 && ni >= t0 + c0 + 1 && c1 == 32 * b1) {
          C[(t0 + c0) * 1100 + (32 * b1 + t1)] = private_C[0][0];
          if (nj >= 32 * b1 + t1 + 17 && 32 * b1 + t1 <= 1083)
            C[(t0 + c0) * 1100 + (32 * b1 + t1 + 16)] = private_C[0][1];
        }
        __syncthreads();
      }
}
