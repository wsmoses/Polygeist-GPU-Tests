#include "doitgen_kernel.hu"
__global__ void kernel0(double *A, double *C4, double *sum, int nr, int nq, int np, int c0, int c1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ double shared_A[1][1][32];
    double private_sum[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 32 * b0; c2 < np; c2 += 1048576) {
      for (int c3 = 0; c3 < np; c3 += 32) {
        if (c1 <= 139 && t0 + c3 <= 159)
          shared_A[0][0][t0] = A[(c0 * 140 + c1) * 160 + (t0 + c3)];
        __syncthreads();
        if (np >= t0 + c2 + 1 && c3 == 0)
          private_sum[0] = 0.;
        if (np >= t0 + c2 + 1)
          for (int c5 = 0; c5 <= ppcg_min(31, np - c3 - 1); c5 += 1)
            private_sum[0] += (shared_A[0][0][c5] * C4[(c3 + c5) * 160 + (t0 + c2)]);
        __syncthreads();
      }
      if (np >= t0 + c2 + 1)
        sum[t0 + c2] = private_sum[0];
      __syncthreads();
    }
}
__global__ void kernel1(double *A, double *sum, int nr, int nq, int np, int c0, int c1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c2 = 32 * b0; c2 < np; c2 += 1048576)
      if (np >= t0 + c2 + 1)
        A[(c0 * 140 + c1) * 160 + (t0 + c2)] = sum[t0 + c2];
}
