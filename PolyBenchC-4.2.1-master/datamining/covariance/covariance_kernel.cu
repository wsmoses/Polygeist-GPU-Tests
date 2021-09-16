#include "covariance_kernel.hu"
__global__ void kernel0(double *data, double *mean, int m, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    double private_mean[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < m; c0 += 1048576) {
      if (m >= t0 + c0 + 1) {
        private_mean[0] = 0.;
        for (int c1 = 0; c1 < n; c1 += 32)
          for (int c3 = 0; c3 <= ppcg_min(31, n - c1 - 1); c3 += 1)
            private_mean[0] += data[(c1 + c3) * 1200 + (t0 + c0)];
        mean[t0 + c0] = private_mean[0];
      }
      __syncthreads();
    }
}
__global__ void kernel1(double *cov, int m, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c0 = 32 * b0; c0 < m; c0 += 8192)
      for (int c1 = 32 * b1 + 8192 * ((-32 * b1 + c0 + 8160) / 8192); c1 < m; c1 += 8192)
        for (int c3 = ppcg_max(t1, t1 + 16 * ppcg_fdiv_q(t0 - t1 + c0 - c1 - 1, 16) + 16); c3 <= ppcg_min(31, m - c1 - 1); c3 += 16)
          cov[(t0 + c0) * 1200 + (c1 + c3)] = 0.;
}
__global__ void kernel2(double float_n, double *mean, int m, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c0 = 32 * b0; c0 < m; c0 += 1048576)
      if (m >= t0 + c0 + 1)
        mean[t0 + c0] /= float_n;
}
__global__ void kernel3(double *data, double *mean, int m, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_mean[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 8192)
      for (int c1 = 32 * b1; c1 < m; c1 += 8192) {
        if (t0 == 0)
          for (int c2 = t1; c2 <= ppcg_min(31, m - c1 - 1); c2 += 16)
            shared_mean[c2] = mean[c1 + c2];
        __syncthreads();
        if (n >= t0 + c0 + 1)
          for (int c3 = t1; c3 <= ppcg_min(31, m - c1 - 1); c3 += 16)
            data[(t0 + c0) * 1200 + (c1 + c3)] -= shared_mean[c3];
        __syncthreads();
      }
}
__global__ void kernel4(double *cov, double *data, double float_n, int m, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ double shared_data_0[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c0 = 32 * b0; c0 < m; c0 += 8192)
      for (int c1 = 32 * b1 + 8192 * ((-32 * b1 + c0 + 8160) / 8192); c1 < m; c1 += 8192) {
        for (int c2 = 0; c2 < n; c2 += 32) {
          if (c0 == 32 * b0 && n >= t0 + c2 + 1)
            for (int c4 = t1; c4 <= ppcg_min(31, -32 * b0 + 1199); c4 += 16)
              shared_data_0[t0][c4] = data[(t0 + c2) * 1200 + (32 * b0 + c4)];
          __syncthreads();
          for (int c4 = ppcg_max(t1, t1 + 16 * ppcg_fdiv_q(t0 - t1 + c0 - c1 - 1, 16) + 16); c4 <= ppcg_min(31, m - c1 - 1); c4 += 16) {
            for (int c5 = 0; c5 <= ppcg_min(31, n - c2 - 1); c5 += 1)
              cov[(t0 + c0) * 1200 + (c1 + c4)] += (shared_data_0[c5][t0] * data[(c2 + c5) * 1200 + (c1 + c4)]);
            if (c2 + 31 >= n) {
              cov[(t0 + c0) * 1200 + (c1 + c4)] /= (float_n - 1.);
              cov[(c1 + c4) * 1200 + (t0 + c0)] = cov[(t0 + c0) * 1200 + (c1 + c4)];
            }
          }
          __syncthreads();
        }
        if (n >= 32 && n % 32 == 0) {
          __syncthreads();
          for (int c4 = ppcg_max(t1, t1 + 16 * ppcg_fdiv_q(t0 - t1 + c0 - c1 - 1, 16) + 16); c4 <= ppcg_min(31, m - c1 - 1); c4 += 16) {
            cov[(t0 + c0) * 1200 + (c1 + c4)] /= (float_n - 1.);
            cov[(c1 + c4) * 1200 + (t0 + c0)] = cov[(t0 + c0) * 1200 + (c1 + c4)];
          }
          __syncthreads();
        }
        if (n <= 0) {
          __syncthreads();
          for (int c4 = ppcg_max(t1, t1 + 16 * ppcg_fdiv_q(t0 - t1 + c0 - c1 - 1, 16) + 16); c4 <= ppcg_min(31, m - c1 - 1); c4 += 16) {
            cov[(t0 + c0) * 1200 + (c1 + c4)] /= (float_n - 1.);
            cov[(c1 + c4) * 1200 + (t0 + c0)] = cov[(t0 + c0) * 1200 + (c1 + c4)];
          }
          __syncthreads();
        }
      }
}
