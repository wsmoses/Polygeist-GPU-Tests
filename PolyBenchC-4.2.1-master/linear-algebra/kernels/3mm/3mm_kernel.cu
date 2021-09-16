#include "3mm_kernel.hu"
__global__ void kernel0(double *C, double *D, double *F, int ni, int nl, int nj, int nm, int nk)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_C[32][32];
    double private_F[1][2];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < nj; c0 += 8192)
      for (int c1 = 32 * b1; c1 < nl; c1 += 8192) {
        for (int c2 = 0; c2 < nm; c2 += 32) {
          if (nj >= t0 + c0 + 1)
            for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 1199); c4 += 16)
              shared_C[t0][c4] = C[(t0 + c0) * 1200 + (c2 + c4)];
          __syncthreads();
          if (nj >= t0 + c0 + 1 && nl >= t1 + c1 + 1 && c2 == 0) {
            private_F[0][0] = 0.;
            if (nl >= t1 + c1 + 17)
              private_F[0][1] = 0.;
          }
          if (nj >= t0 + c0 + 1 && nl >= t1 + c1 + 1)
            for (int c3 = 0; c3 <= ppcg_min(31, nm - c2 - 1); c3 += 1) {
              private_F[0][0] += (shared_C[t0][c3] * D[(c2 + c3) * 1100 + (t1 + c1)]);
              if (nl >= t1 + c1 + 17)
                private_F[0][1] += (shared_C[t0][c3] * D[(c2 + c3) * 1100 + (t1 + c1 + 16)]);
            }
          __syncthreads();
        }
        if (nm <= 0) {
          __syncthreads();
          if (nj >= t0 + c0 + 1 && nl >= t1 + c1 + 1) {
            private_F[0][0] = 0.;
            if (nl >= t1 + c1 + 17)
              private_F[0][1] = 0.;
          }
          __syncthreads();
        }
        if (nl >= 32 * b1 + t1 + 1 && 32 * b1 + t1 <= 1099 && nj >= t0 + c0 + 1 && c1 == 32 * b1) {
          F[(t0 + c0) * 1100 + (32 * b1 + t1)] = private_F[0][0];
          if (nl >= 32 * b1 + t1 + 17 && 32 * b1 + t1 <= 1083)
            F[(t0 + c0) * 1100 + (32 * b1 + t1 + 16)] = private_F[0][1];
        }
        __syncthreads();
      }
}
__global__ void kernel1(double *A, double *B, double *E, int ni, int nl, int nj, int nm, int nk)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_A[32][32];
    double private_E[1][2];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < ni; c0 += 8192)
      for (int c1 = 32 * b1; c1 < nj; c1 += 8192) {
        for (int c2 = 0; c2 < nk; c2 += 32) {
          if (ni >= t0 + c0 + 1)
            for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 999); c4 += 16)
              shared_A[t0][c4] = A[(t0 + c0) * 1000 + (c2 + c4)];
          __syncthreads();
          if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1 && c2 == 0) {
            private_E[0][0] = 0.;
            if (nj >= t1 + c1 + 17)
              private_E[0][1] = 0.;
          }
          if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1)
            for (int c3 = 0; c3 <= ppcg_min(31, nk - c2 - 1); c3 += 1) {
              private_E[0][0] += (shared_A[t0][c3] * B[(c2 + c3) * 900 + (t1 + c1)]);
              if (nj >= t1 + c1 + 17)
                private_E[0][1] += (shared_A[t0][c3] * B[(c2 + c3) * 900 + (t1 + c1 + 16)]);
            }
          __syncthreads();
        }
        if (nk <= 0) {
          __syncthreads();
          if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1) {
            private_E[0][0] = 0.;
            if (nj >= t1 + c1 + 17)
              private_E[0][1] = 0.;
          }
          __syncthreads();
        }
        if (nj >= 32 * b1 + t1 + 1 && 32 * b1 + t1 <= 899 && ni >= t0 + c0 + 1 && c1 == 32 * b1) {
          E[(t0 + c0) * 900 + (32 * b1 + t1)] = private_E[0][0];
          if (nj >= 32 * b1 + t1 + 17 && 32 * b1 + t1 <= 883)
            E[(t0 + c0) * 900 + (32 * b1 + t1 + 16)] = private_E[0][1];
        }
        __syncthreads();
      }
}
__global__ void kernel2(double *E, double *F, double *G, int ni, int nl, int nj, int nm, int nk)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_E[32][32];
    double private_G[1][2];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < ni; c0 += 8192)
      for (int c1 = 32 * b1; c1 < nl; c1 += 8192) {
        if (nj >= 1 && ni >= t0 + c0 + 1)
          for (int c4 = t1; c4 <= 31; c4 += 16)
            shared_E[t0][c4] = E[(t0 + c0) * 900 + c4];
        __syncthreads();
        if (ni >= t0 + c0 + 1 && nl >= t1 + c1 + 1) {
          private_G[0][0] = 0.;
          if (nl >= t1 + c1 + 17)
            private_G[0][1] = 0.;
          for (int c3 = 0; c3 <= ppcg_min(31, nj - 1); c3 += 1) {
            private_G[0][0] += (shared_E[t0][c3] * F[c3 * 1100 + (t1 + c1)]);
            if (nl >= t1 + c1 + 17)
              private_G[0][1] += (shared_E[t0][c3] * F[c3 * 1100 + (t1 + c1 + 16)]);
          }
        }
        __syncthreads();
        for (int c2 = 32; c2 < nj; c2 += 32) {
          if (ni >= t0 + c0 + 1)
            for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 899); c4 += 16)
              shared_E[t0][c4] = E[(t0 + c0) * 900 + (c2 + c4)];
          __syncthreads();
          if (ni >= t0 + c0 + 1 && nl >= t1 + c1 + 1)
            for (int c3 = 0; c3 <= ppcg_min(31, nj - c2 - 1); c3 += 1) {
              private_G[0][0] += (shared_E[t0][c3] * F[(c2 + c3) * 1100 + (t1 + c1)]);
              if (nl >= t1 + c1 + 17)
                private_G[0][1] += (shared_E[t0][c3] * F[(c2 + c3) * 1100 + (t1 + c1 + 16)]);
            }
          __syncthreads();
        }
        if (nl >= 32 * b1 + t1 + 1 && 32 * b1 + t1 <= 1099 && ni >= t0 + c0 + 1 && c1 == 32 * b1) {
          G[(t0 + c0) * 1100 + (32 * b1 + t1)] = private_G[0][0];
          if (nl >= 32 * b1 + t1 + 17 && 32 * b1 + t1 <= 1083)
            G[(t0 + c0) * 1100 + (32 * b1 + t1 + 16)] = private_G[0][1];
        }
        __syncthreads();
      }
}
