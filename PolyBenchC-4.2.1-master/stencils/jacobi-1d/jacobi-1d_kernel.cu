#include "jacobi-1d_kernel.hu"
__global__ void kernel0(double *A, double *B, int tsteps, int n, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c1 = 32 * b0; c1 < n - 2; c1 += 1048576)
      if (n >= t0 + c1 + 3)
        B[t0 + c1 + 1] = (0.33333 * ((A[t0 + c1] + A[t0 + c1 + 1]) + A[t0 + c1 + 2]));
}
__global__ void kernel1(double *A, double *B, int tsteps, int n, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c1 = 32 * b0; c1 < n - 2; c1 += 1048576)
      if (n >= t0 + c1 + 3)
        A[t0 + c1 + 1] = (0.33333 * ((B[t0 + c1] + B[t0 + c1 + 1]) + B[t0 + c1 + 2]));
}
