// g++ Q2.cpp -o Q2 -std=c++17 -O3 -Wall && ./Q2

#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int N = 1, // batch
    H = 64,          // height_feature
    W = 64 * 64,     // width_feature
    C = 1,           // in_channels
    F = 512,         // out_channels
    K = 3,           // kernel_size
    S = 1,           // stride
    P = 0,           // padding

    H_R = 1 + (H + 2 * P - K) / S, // output height
    W_R = 1 + (W + 2 * P - K) / S; // output width

int x[N][C][H][W];                     // Input Feature Map
int x_pad[N][C][H + 2 * P][W + 2 * P]; // Pad images
int w[F][C][K][K];                     // Filter
int out[N][F][H_R][W_R];               // Output Feature Map
int out_groundtruth[N][F][H_R][W_R];   // Output groundtruth

int im_col[N][C * K * K][H_R * W_R];
int filter_col[F][C * K * K];
int mul[N][F][H_R * W_R];

void init()
{
  for (int i = 0; i < N; i++)
    for (int j = 0; j < C; j++)
      for (int k = 0; k < H; k++)
        for (int l = 0; l < W; l++)
        {
          x[i][j][k][l] = rand();
          x_pad[i][j][k + P][l + P] = x[i][j][k][l];
        }

  for (int i = 0; i < F; i++)
    for (int j = 0; j < C; j++)
      for (int k = 0; k < K; k++)
        for (int l = 0; l < K; l++)
          w[i][j][k][l] = rand();

  for (int n = 0; n < N; n++)
    for (int depth = 0; depth < F; depth++)
      for (int r = 0; r < H_R; r++)
        for (int c = 0; c < W_R; c++)
        {
          for (int cc = 0; cc < C; cc++)
            for (int hh = 0; hh < K; hh++)
              for (int ww = 0; ww < K; ww++)
                out_groundtruth[n][depth][r][c] += x_pad[n][cc][r * S + hh][c * S + ww] * w[depth][cc][hh][ww];
        }
}

void test()
{
  for (int i = 0; i < N; i++)
    for (int j = 0; j < F; j++)
      for (int k = 0; k < H_R; k++)
        for (int l = 0; l < W_R; l++)
          assert(out[i][j][k][l] == out_groundtruth[i][j][k][l]);
}

void stand_conv()
{
  memset(out, 0, sizeof(out));
  for (int n = 0; n < N; n++)
    for (int depth = 0; depth < F; depth++)
      for (int r = 0; r < H_R; r++)
        for (int c = 0; c < W_R; c++)
        {
          for (int cc = 0; cc < C; cc++)
            for (int hh = 0; hh < K; hh++)
              for (int ww = 0; ww < K; ww++)
                out[n][depth][r][c] += x_pad[n][cc][r * S + hh][c * S + ww] * w[depth][cc][hh][ww];
        }
}

void im2col()
{
  for (int n = 0; n < N; n++)
    for (int r = 0; r < H_R; r++)
      for (int c = 0; c < W_R; c++)
      {
        for (int cc = 0; cc < C; cc++)
          for (int hh = 0; hh < K; hh++)
            for (int ww = 0; ww < K; ww++)
              im_col[n][cc * K * K + hh * K + ww][r * W_R + c] = x_pad[n][cc][r * S + hh][c * S + ww];
      }

  for (int depth = 0; depth < F; depth++)
    for (int cc = 0; cc < C; cc++)
      for (int hh = 0; hh < K; hh++)
        for (int ww = 0; ww < K; ww++)
          filter_col[depth][cc * K * K + hh * K + ww] = w[depth][cc][hh][ww];
}

void col2im()
{
  for (int n = 0; n < N; n++)
    for (int depth = 0; depth < F; depth++)
      for (int r = 0; r < H_R; r++)
        for (int c = 0; c < W_R; c++)
          out[n][depth][r][c] = mul[n][depth][r * W_R + c];
}

void im2col_conv()
{
  memset(out, 0, sizeof(out));
  memset(mul, 0, sizeof(mul));
  im2col();
  for (int n = 0; n < N; n++)
    for (int depth = 0; depth < F; depth++)
      for (int i = 0; i < C * K * K; i++)
        for (int j = 0; j < H_R * W_R; j++)
          mul[n][depth][j] += im_col[n][i][j] * filter_col[depth][i];
  col2im();
}

int main()
{
  init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++)
  {
    auto t = get_time();
    stand_conv();
    // im2col_conv();
    test();
    printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Calculation: %f\n", avg_time / 32);
  return 0;
}
