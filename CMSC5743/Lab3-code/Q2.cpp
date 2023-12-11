// g++ Q2.cpp -o Q2 -std=c++17 -O3 -Wall && ./Q2

#include "npy.hpp"
#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

using namespace std;

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int batch = 1, // batch
    height_feature = 5,  // height_feature
    width_feature = 5,   // width_feature
    in_channels = 3,     // in_channels
    out_channels = 2,    // out_channels
    kernel_size = 3,     // kernel_size
    stride = 1,          // stride
    padding = 0,         // padding

    height_out = (height_feature - kernel_size + 2 * padding) / stride + 1, // output height
    width_out = (width_feature - kernel_size + 2 * padding) / stride + 1;   // output width

int A[batch][in_channels][height_feature][width_feature];
int A_padding[batch][in_channels][height_feature + 2 * padding][width_feature + 2 * padding];
int B[out_channels][in_channels][kernel_size][kernel_size];
int C[batch][out_channels][height_out][width_out];
int C_groundtruth[batch][out_channels][height_out][width_out];

void init()
{
  // for (int i = 0; i < batch; i++)
  //   for (int j = 0; j < in_channels; j++)
  //     for (int k = 0; k < height_feature; k++)
  //       for (int l = 0; l < width_feature; l++)
  //         A[i][j][k][l] = rand();
  for (int j = 0; j < in_channels; j++)
  {
    A[0][j][1][2] = 1;
    A[0][j][2][3] = 2;
  }
  for (int i = 0; i < batch; i++)
    for (int j = 0; j < in_channels; j++)
      for (int k = 0; k < height_feature; k++)
        for (int l = 0; l < width_feature; l++)
          A_padding[i][j][padding + k][padding + l] = A[i][j][k][l];

  for (int i = 0; i < out_channels; i++)
    for (int j = 0; j < in_channels; j++)
      for (int k = 0; k < kernel_size; k++)
        for (int l = 0; l < kernel_size; l++)
          B[i][j][k][l] = rand();

  for (int i = 0; i < batch; i++)
    for (int j = 0; j < out_channels; j++)
      for (int k = 0; k < height_out; k++)
        for (int l = 0; l < width_out; l++)
        {
          for (int m = 0; m < in_channels; m++)
            for (int n = 0; n < kernel_size; n++)
              for (int o = 0; o < kernel_size; o++)
                C_groundtruth[i][j][k][l] += A_padding[i][m][k + n][l + o] * B[j][m][n][o];
        }
}

void test()
{
  for (int i = 0; i < batch; i++)
    for (int j = 0; j < out_channels; j++)
      for (int k = 0; k < height_out; k++)
        for (int l = 0; l < width_out; l++)
          assert(C[i][j][k][l] == C_groundtruth[i][j][k][l]);
}

int H_in[][2];
int P_out[][2];
int H_out[][2];
int Offset;
int rulebook;

void matmul_Sparse()
{
  memset(C, 0, sizeof(C));
}

int main()
{
  init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++)
  {
    auto t = get_time();
    matmul_Sparse();
    test();
    printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Calculation: %f\n", avg_time / 32);
  return 0;
}
