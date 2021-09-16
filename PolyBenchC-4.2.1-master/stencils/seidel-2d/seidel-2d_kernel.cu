#include "seidel-2d_kernel.hu"
__global__ void kernel0(double *A, int tsteps, int n, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c1 = ppcg_max(32 * b0, 32 * b0 + 8192 * ppcg_fdiv_q(-3 * n - 128 * b0 + c0 - 119, 32768) + 8192); c1 < ppcg_min(tsteps, -16 * b1 + (c0 + 1) / 4); c1 += 8192)
      if (tsteps >= t0 + c1 + 1)
        for (int c2 = ppcg_max(ppcg_max(32 * b1, 32 * b1 + 8192 * ppcg_fdiv_q(-4 * tsteps - n - 64 * b1 + c0 - 59, 16384) + 8192), 32 * b1 + 8192 * ppcg_fdiv_q(-n - 64 * b1 + c0 - 4 * c1 - 187, 16384) + 8192); c2 < ppcg_min(n - 2, -2 * c1 + (c0 + 1) / 2 - 1); c2 += 8192)
          for (int c4 = ppcg_max(t1, t1 + 16 * ppcg_fdiv_q(-n - 4 * t0 - 2 * t1 + c0 - 4 * c1 - 2 * c2 - 1, 32) + 16); c4 <= ppcg_min(ppcg_min(31, n - c2 - 3), -2 * t0 - 2 * c1 - c2 + (c0 + 1) / 2 - 2); c4 += 16)
            A[(c2 + c4 + 1) * 2000 + (-4 * t0 + c0 - 4 * c1 - 2 * c2 - 2 * c4 - 2)] = (((((((((A[(c2 + c4) * 2000 + (-4 * t0 + c0 - 4 * c1 - 2 * c2 - 2 * c4 - 3)] + A[(c2 + c4) * 2000 + (-4 * t0 + c0 - 4 * c1 - 2 * c2 - 2 * c4 - 2)]) + A[(c2 + c4) * 2000 + (-4 * t0 + c0 - 4 * c1 - 2 * c2 - 2 * c4 - 1)]) + A[(c2 + c4 + 1) * 2000 + (-4 * t0 + c0 - 4 * c1 - 2 * c2 - 2 * c4 - 3)]) + A[(c2 + c4 + 1) * 2000 + (-4 * t0 + c0 - 4 * c1 - 2 * c2 - 2 * c4 - 2)]) + A[(c2 + c4 + 1) * 2000 + (-4 * t0 + c0 - 4 * c1 - 2 * c2 - 2 * c4 - 1)]) + A[(c2 + c4 + 2) * 2000 + (-4 * t0 + c0 - 4 * c1 - 2 * c2 - 2 * c4 - 3)]) + A[(c2 + c4 + 2) * 2000 + (-4 * t0 + c0 - 4 * c1 - 2 * c2 - 2 * c4 - 2)]) + A[(c2 + c4 + 2) * 2000 + (-4 * t0 + c0 - 4 * c1 - 2 * c2 - 2 * c4 - 1)]) / 9.);
}
