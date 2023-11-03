// g++ Q1.cpp -o Q1 -std=c++17 -O3 -Wall && ./Q1

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

constexpr int I = 256,
              J = 512,
              K = 1024;
int A[I][K];
int B[K][J];
int BT[J][K];
int AT[K][I];
int C[I][J];
int C_groundtruth[I][J];

void init()
{
  for (int i = 0; i < I; i++)
  {
    for (int k = 0; k < K; k++)
    {
      A[i][k] = rand();
    }
  }
  for (int k = 0; k < K; k++)
  {
    for (int j = 0; j < J; j++)
    {
      B[k][j] = rand();
    }
  }
  memset(C_groundtruth, 0, sizeof(C_groundtruth));
  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < J; j++)
    {
      for (int k = 0; k < K; k++)
      {
        C_groundtruth[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void test()
{
  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < J; j++)
    {
      assert(C[i][j] == C_groundtruth[i][j]);
    }
  }
}

void matmul()
{
  memset(C, 0, sizeof(C));
  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < J; j++)
    {
      for (int k = 0; k < K; k++)
      {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void matmul_ikj()
{
  memset(C, 0, sizeof(C));
  for (int i = 0; i < I; i++)
  {
    for (int k = 0; k < K; k++)
    {
      for (int j = 0; j < J; j++)
      {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void matmul_AT()
{
  memset(C, 0, sizeof(C));
  for (int k = 0; k < K; k++)
  {
    for (int i = 0; i < I; i++)
    {
      AT[k][i] = A[i][k];
    }
  }
  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < J; j++)
    {
      for (int k = 0; k < K; k++)
      {
        C[i][j] += AT[k][i] * B[k][j];
      }
    }
  }
}

void matmul_BT()
{
  memset(C, 0, sizeof(C));
  for (int j = 0; j < J; j++)
  {
    for (int k = 0; k < K; k++)
    {
      BT[j][k] = B[k][j];
    }
  }
  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < J; j++)
    {
      for (int k = 0; k < K; k++)
      {
        C[i][j] += A[i][k] * BT[j][k];
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
    matmul();
    // matmul_AT();
    // matmul_BT();
    test();
    printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Calculation: %f\n", avg_time / 32);
  return 0;
}
