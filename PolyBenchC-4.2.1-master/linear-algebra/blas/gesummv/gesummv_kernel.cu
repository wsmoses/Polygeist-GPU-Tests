#include "gesummv_kernel.hu"
__global__ void kernel0(double *A, double *B, double alpha, double beta, double *tmp, double *x, double *y, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ double shared_A[32][32];
    double private_tmp[1];
    double private_y[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 1048576) {
      for (int c1 = 0; c1 <= n; c1 += 32) {
        if (t0 + c1 <= 1299)
          for (int c2 = 0; c2 <= ppcg_min(31, n - c0 - 1); c2 += 1)
            shared_A[c2][t0] = A[(c0 + c2) * 1300 + (t0 + c1)];
        __syncthreads();
        if (n >= t0 + c0 + 1 && c1 == 0) {
          private_y[0] = 0.;
          private_tmp[0] = 0.;
        }
        if (n >= t0 + c0 + 1) {
          for (int c3 = 0; c3 <= ppcg_min(31, n - c1 - 1); c3 += 1) {
            private_y[0] = ((B[(t0 + c0) * 1300 + (c1 + c3)] * x[c1 + c3]) + private_y[0]);
            private_tmp[0] = ((shared_A[t0][c3] * x[c1 + c3]) + private_tmp[0]);
          }
          if (c1 + 31 >= n)
            private_y[0] = ((alpha * private_tmp[0]) + (beta * private_y[0]));
        }
        __syncthreads();
      }
      if (n >= t0 + c0 + 1) {
        y[t0 + c0] = private_y[0];
        tmp[t0 + c0] = private_tmp[0];
      }
      __syncthreads();
    }
}
