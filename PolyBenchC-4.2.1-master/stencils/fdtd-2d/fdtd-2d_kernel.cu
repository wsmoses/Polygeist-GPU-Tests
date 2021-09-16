#include "fdtd-2d_kernel.hu"
__global__ void kernel0(double *ey, double *hz, int tmax, int nx, int ny, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 32 * b0; c1 < nx - 1; c1 += 8192)
      if (nx >= t0 + c1 + 2)
        for (int c2 = 32 * b1; c2 < ny; c2 += 8192)
          for (int c4 = t1; c4 <= ppcg_min(31, ny - c2 - 1); c4 += 16)
            ey[(t0 + c1 + 1) * 1200 + (c2 + c4)] = (ey[(t0 + c1 + 1) * 1200 + (c2 + c4)] - (0.5 * (hz[(t0 + c1 + 1) * 1200 + (c2 + c4)] - hz[(t0 + c1) * 1200 + (c2 + c4)])));
}
__global__ void kernel1(double *ex, double *hz, int tmax, int nx, int ny, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 32 * b0; c1 < nx; c1 += 8192)
      if (nx >= t0 + c1 + 1)
        for (int c2 = 32 * b1; c2 < ny - 1; c2 += 8192)
          for (int c4 = t1; c4 <= ppcg_min(31, ny - c2 - 2); c4 += 16)
            ex[(t0 + c1) * 1200 + (c2 + c4 + 1)] = (ex[(t0 + c1) * 1200 + (c2 + c4 + 1)] - (0.5 * (hz[(t0 + c1) * 1200 + (c2 + c4 + 1)] - hz[(t0 + c1) * 1200 + (c2 + c4)])));
}
__global__ void kernel2(double *_fict_, double *ex, double *ey, double *hz, int tmax, int nx, int ny, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared__fict_[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (t0 == 0 && t1 == 0)
        shared__fict_[0] = _fict_[c0];
      __syncthreads();
      if (ny >= 32 * b1 + 2)
        for (int c1 = 32 * b0; c1 < nx - 1; c1 += 8192)
          if (nx >= t0 + c1 + 2) {
            for (int c2 = 32 * b1; c2 < ny - 1; c2 += 8192)
              if (ny >= t1 + c2 + 1) {
                for (int c4 = t1; c4 <= ppcg_min(31, ny - c2 - 2); c4 += 16) {
                  if (b0 == 0 && t0 == 0 && c1 == 0)
                    ey[0 * 1200 + (c2 + c4)] = shared__fict_[0];
                  hz[(t0 + c1) * 1200 + (c2 + c4)] = (hz[(t0 + c1) * 1200 + (c2 + c4)] - (0.69999999999999996 * (((ex[(t0 + c1) * 1200 + (c2 + c4 + 1)] - ex[(t0 + c1) * 1200 + (c2 + c4)]) + ey[(t0 + c1 + 1) * 1200 + (c2 + c4)]) - ey[(t0 + c1) * 1200 + (c2 + c4)])));
                }
                if (b0 == 0 && t0 == 0 && c1 == 0 && c2 + 32 >= ny && (ny - t1 - 1) % 16 == 0)
                  ey[0 * 1200 + (ny - 1)] = shared__fict_[0];
              }
            if (b0 == 0 && t0 == 0 && t1 == 0 && c1 == 0 && (-ny + 32 * b1 + 1) % 8192 == 0)
              ey[0 * 1200 + (ny - 1)] = shared__fict_[0];
          }
      if (nx <= 1 && b0 == 0 && t0 == 0) {
        for (int c2 = 32 * b1; c2 < ny; c2 += 8192)
          for (int c4 = t1; c4 <= ppcg_min(31, ny - c2 - 1); c4 += 16)
            ey[0 * 1200 + (c2 + c4)] = shared__fict_[0];
      } else if (nx >= 2 && b0 == 0 && 32 * b1 + 1 == ny && t0 == 0 && t1 == 0) {
        ey[0 * 1200 + (ny - 1)] = shared__fict_[0];
      }
    }
}
