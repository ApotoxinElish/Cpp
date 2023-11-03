// g++ Q3.cpp -o Q3 -std=c++17 -O3 -Wall && ./Q3

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

constexpr int n = 512;
int A[n][n];
int B[n][n];
int BT[n][n];
int AT[n][n];
int C[n][n];
int C_groundtruth[n][n];

void init()
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      A[i][j] = rand();
      B[i][j] = rand();
    }
  }
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      for (int k = 0; k < n; k++)
      {
        C_groundtruth[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void test()
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      assert(C[i][j] == C_groundtruth[i][j]);
    }
  }
}

void matmul()
{
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      for (int k = 0; k < n; k++)
      {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void matmul_ikj()
{
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++)
  {
    for (int k = 0; k < n; k++)
    {
      for (int j = 0; j < n; j++)
      {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void matmul_AT()
{
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      AT[i][j] = A[j][i];
    }
  }
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      for (int k = 0; k < n; k++)
      {
        C[i][j] += AT[k][i] * B[k][j];
      }
    }
  }
}

void matmul_BT()
{
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      BT[i][j] = B[j][i];
    }
  }
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      for (int k = 0; k < n; k++)
      {
        C[i][j] += A[i][k] * BT[j][k];
      }
    }
  }
}

void matmul_unroll()
{
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      for (int k = 0; k < n; k += 8)
      {
        C[i][j] += A[i][k] * B[k][j];
        C[i][j] += A[i][k + 1] * B[k + 1][j];
        C[i][j] += A[i][k + 2] * B[k + 2][j];
        C[i][j] += A[i][k + 3] * B[k + 3][j];
        C[i][j] += A[i][k + 4] * B[k + 4][j];
        C[i][j] += A[i][k + 5] * B[k + 5][j];
        C[i][j] += A[i][k + 6] * B[k + 6][j];
        C[i][j] += A[i][k + 7] * B[k + 7][j];
      }
    }
  }
}

void matmul_tiling()
{
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i += 16)
  {
    for (int k = 0; k < n; k += 64)
    {
      for (int j = 0; j < n; j += 64)
      {
        for (int ii = i; ii < i + 16; ii++)
        {
          for (int kk = k; kk < k + 64; kk++)
          {
            for (int jj = j; jj < j + 64; jj++)
            {
              C[ii][jj] += A[ii][kk] * B[kk][jj];
            }
          }
        }
      }
    }
  }
}

int main()
{
  init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++)
  {
    auto t = get_time();
    // matmul_ikj();
    // matmul();
    // matmul_AT();
    // matmul_BT();
    // matmul_unroll();
    matmul_tiling();
    test();
    printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Calculation: %f\n", avg_time / 32);
  return 0;
}
