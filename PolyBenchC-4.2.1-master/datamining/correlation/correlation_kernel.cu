#include "correlation_kernel.hu"
__global__ void kernel0(double *corr, int m, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c0 = 32 * b0; c0 < m - 1; c0 += 1048576)
      if (m >= t0 + c0 + 2)
        corr[(t0 + c0) * 1200 + (t0 + c0)] = 1.;
}
__global__ void kernel1(double *data, double *mean, int m, int n)
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
__global__ void kernel2(double *data, double float_n, double *mean, double *stddev, int m, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    double private_mean[1];
    double private_stddev[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < m; c0 += 1048576) {
      if (m >= t0 + c0 + 1) {
        private_mean[0] = mean[t0 + c0];
        private_mean[0] /= float_n;
        private_stddev[0] = 0.;
        for (int c1 = 0; c1 < n; c1 += 32)
          for (int c3 = 0; c3 <= ppcg_min(31, n - c1 - 1); c3 += 1)
            private_stddev[0] += ((data[(c1 + c3) * 1200 + (t0 + c0)] - private_mean[0]) * (data[(c1 + c3) * 1200 + (t0 + c0)] - private_mean[0]));
        stddev[t0 + c0] = private_stddev[0];
        mean[t0 + c0] = private_mean[0];
      }
      __syncthreads();
    }
}
__global__ void kernel3(double eps, double float_n, double *stddev, int m, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    double private_stddev[1];

    for (int c0 = 32 * b0; c0 < m; c0 += 1048576)
      if (m >= t0 + c0 + 1) {
        private_stddev[0] = stddev[t0 + c0];
        private_stddev[0] /= float_n;
        private_stddev[0] = sqrt(private_stddev[0]);
        private_stddev[0] = ((private_stddev[0] <= eps) ? 1. : private_stddev[0]);
        stddev[t0 + c0] = private_stddev[0];
      }
}
__global__ void kernel4(double *data, double float_n, double *mean, double *stddev, int m, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    double private_data[1][2];
    __shared__ double shared_mean[32];
    __shared__ double shared_stddev[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 8192)
      for (int c1 = 32 * b1; c1 < m; c1 += 8192) {
        if (b1 <= 37 && m >= 32 * b1 + t1 + 1 && n >= t0 + c0 + 1 && c1 == 32 * b1) {
          private_data[0][0] = data[(t0 + c0) * 1200 + (32 * b1 + t1)];
          if (m >= 32 * b1 + t1 + 17 && 32 * b1 + t1 <= 1183)
            private_data[0][1] = data[(t0 + c0) * 1200 + (32 * b1 + t1 + 16)];
        }
        if (t0 == 0) {
          for (int c2 = t1; c2 <= ppcg_min(31, m - c1 - 1); c2 += 16)
            shared_mean[c2] = mean[c1 + c2];
          for (int c2 = t1; c2 <= ppcg_min(31, m - c1 - 1); c2 += 16)
            shared_stddev[c2] = stddev[c1 + c2];
        }
        __syncthreads();
        if (n >= t0 + c0 + 1 && m >= t1 + c1 + 1) {
          private_data[0][0] -= shared_mean[t1];
          if (m >= t1 + c1 + 17)
            private_data[0][1] -= shared_mean[t1 + 16];
          private_data[0][0] /= (sqrt(float_n) * shared_stddev[t1]);
          if (m >= t1 + c1 + 17)
            private_data[0][1] /= (sqrt(float_n) * shared_stddev[t1 + 16]);
        }
        __syncthreads();
        if (b1 <= 37 && m >= 32 * b1 + t1 + 1 && n >= t0 + c0 + 1 && c1 == 32 * b1) {
          data[(t0 + c0) * 1200 + (32 * b1 + t1)] = private_data[0][0];
          if (m >= 32 * b1 + t1 + 17 && 32 * b1 + t1 <= 1183)
            data[(t0 + c0) * 1200 + (32 * b1 + t1 + 16)] = private_data[0][1];
        }
      }
}
__global__ void kernel5(double *corr, int m, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c0 = 32 * b0; c0 < m - 1; c0 += 8192)
      for (int c1 = 32 * b1 + 8192 * ((-32 * b1 + c0 + 8160) / 8192); c1 < m - 1; c1 += 8192)
        for (int c3 = ppcg_max(t1, t1 + 16 * ppcg_fdiv_q(t0 - t1 + c0 - c1 - 1, 16) + 16); c3 <= ppcg_min(31, m - c1 - 2); c3 += 16)
          corr[(t0 + c0) * 1200 + (c1 + c3 + 1)] = 0.;
}
__global__ void kernel6(double *corr, double *data, int m, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    double private_corr_0[1][2];
    __shared__ double shared_corr_1[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < m - 1; c0 += 8192)
      for (int c1 = 32 * b1 + 8192 * ((-32 * b1 + c0 + 8161) / 8192); c1 < m; c1 += 8192) {
        if (b1 <= 37 && m >= 32 * b0 + t0 + 2 && 32 * b0 + t0 <= 1198 && m >= 32 * b1 + t1 + 1 && 32 * b1 + t1 + 15 >= 32 * b0 + t0 && c0 == 32 * b0 && c1 == 32 * b1) {
          if (32 * b1 + t1 >= 32 * b0 + t0 + 1)
            private_corr_0[0][0] = corr[(32 * b0 + t0) * 1200 + (32 * b1 + t1)];
          if (m >= 32 * b1 + t1 + 17 && 32 * b1 + t1 <= 1183)
            private_corr_0[0][1] = corr[(32 * b0 + t0) * 1200 + (32 * b1 + t1 + 16)];
        }
        __syncthreads();
        if (m >= t0 + c0 + 2 && m >= t1 + c1 + 1 && t1 + c1 + 15 >= t0 + c0) {
          for (int c2 = 0; c2 < n; c2 += 32)
            for (int c3 = 0; c3 <= ppcg_min(31, n - c2 - 1); c3 += 1) {
              if (t1 + c1 >= t0 + c0 + 1)
                private_corr_0[0][0] += (data[(c2 + c3) * 1200 + (t0 + c0)] * data[(c2 + c3) * 1200 + (t1 + c1)]);
              if (m >= t1 + c1 + 17)
                private_corr_0[0][1] += (data[(c2 + c3) * 1200 + (t0 + c0)] * data[(c2 + c3) * 1200 + (t1 + c1 + 16)]);
            }
          if (t1 + c1 >= t0 + c0 + 1)
            shared_corr_1[t1][t0] = private_corr_0[0][0];
          if (m >= t1 + c1 + 17)
            shared_corr_1[t1 + 16][t0] = private_corr_0[0][1];
          if (n >= 1 && b1 <= 37 && 32 * b0 + t0 <= 1198 && c0 == 32 * b0 && c1 == 32 * b1) {
            if (32 * b1 + t1 >= 32 * b0 + t0 + 1)
              corr[(32 * b0 + t0) * 1200 + (32 * b1 + t1)] = private_corr_0[0][0];
            if (m >= 32 * b1 + t1 + 17 && 32 * b1 + t1 <= 1183)
              corr[(32 * b0 + t0) * 1200 + (32 * b1 + t1 + 16)] = private_corr_0[0][1];
          }
        }
        __syncthreads();
        if (m >= t0 + c1 + 1)
          for (int c3 = t1; c3 <= ppcg_min(31, t0 - c0 + c1 - 1); c3 += 16)
            corr[(t0 + c1) * 1200 + (c0 + c3)] = shared_corr_1[t0][c3];
      }
}
