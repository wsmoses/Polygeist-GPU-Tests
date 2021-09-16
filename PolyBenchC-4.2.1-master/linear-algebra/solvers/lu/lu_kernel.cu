#include "lu_kernel.hu"
__global__ void kernel0(double *A, int n, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ double shared_A_1[1][1];

    {
      if (t0 == 0 && c0 <= 1999)
        shared_A_1[0][0] = A[c0 * 2000 + c0];
      __syncthreads();
      for (int c1 = 32 * b0 + 1048576 * ((-32 * b0 + c0 + 1048544) / 1048576); c1 < n - 1; c1 += 1048576)
        if (n >= t0 + c1 + 2 && t0 + c1 >= c0)
          A[(t0 + c1 + 1) * 2000 + c0] /= shared_A_1[0][0];
    }
}
__global__ void kernel1(double *A, int n, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_A_1[32][1];
    __shared__ double shared_A_2[1][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c1 = 32 * b0 + 8192 * ((-32 * b0 + c0 + 8160) / 8192); c1 < 32 * ((b0 - b1 + 255) % 256) + n - 8161; c1 += 8192) {
      if (t1 == 0 && c0 <= 1999 && n >= t0 + c1 + 2)
        shared_A_1[t0][0] = A[(t0 + c1 + 1) * 2000 + c0];
      __syncthreads();
      for (int c2 = 32 * b1 + 8192 * ((-32 * b1 + c1 + 8160) / 8192); c2 < n - 1; c2 += 8192) {
        if (t0 == 0 && c1 == 32 * b0 && c2 == 32 * b1)
          for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 1998); c4 += 16)
            shared_A_2[0][c4] = A[c0 * 2000 + (32 * b1 + c4 + 1)];
        __syncthreads();
        if (t0 + c1 >= c0)
          for (int c4 = ppcg_max(t1, t1 + 16 * ppcg_fdiv_q(t0 - t1 + c1 - c2 - 1, 16) + 16); c4 <= ppcg_min(31, n - c2 - 2); c4 += 16)
            A[(t0 + c1 + 1) * 2000 + (c2 + c4 + 1)] -= (shared_A_1[t0][0] * shared_A_2[0][c4]);
        __syncthreads();
      }
      __syncthreads();
    }
}
__global__ void kernel2(double *A, int n, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_A_1[32][1];
    __shared__ double shared_A_2[1][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c1 = 32 * ((b0 - b1 + 256) % 256) - ((-32 * b1 + c0 + 8160) % 8192) + c0 + 8160; c1 < n - 2; c1 += 8192) {
      if (t1 == 0 && c0 <= 1999 && n >= t0 + c1 + 3)
        shared_A_1[t0][0] = A[(t0 + c1 + 2) * 2000 + c0];
      __syncthreads();
      for (int c2 = 32 * b1 + 8192 * ((-32 * b1 + c0 + 8160) / 8192); c2 <= ppcg_min(n - 3, c1 + 31); c2 += 8192) {
        if (t0 == 0 && c2 == 32 * b1)
          for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 1998); c4 += 16)
            shared_A_2[0][c4] = A[c0 * 2000 + (32 * b1 + c4 + 1)];
        __syncthreads();
        if (n >= t0 + c1 + 3)
          for (int c4 = ppcg_max(t1, t1 + 16 * ppcg_fdiv_q(-t1 + c0 - c2 - 1, 16) + 16); c4 <= ppcg_min(31, t0 + c1 - c2); c4 += 16)
            A[(t0 + c1 + 2) * 2000 + (c2 + c4 + 1)] -= (shared_A_1[t0][0] * shared_A_2[0][c4]);
        __syncthreads();
      }
      __syncthreads();
    }
}
