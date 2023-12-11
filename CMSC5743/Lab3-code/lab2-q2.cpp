/*
1155196314 SHEN Haoran

*/
#include <sys/time.h>
#include <iostream>
#include <assert.h>
#include <cstring>
#include <cassert>

using namespace std;

#define use_im2col 1
#define use_normal 2
#define use_winograd 3

// change the value to compare different methods.
const int method = use_winograd;

// These settings can be changed.
constexpr int H = 56, W = 56, IC = 3;
constexpr int OC = 64, K = 3;

// These default settings can not be modified.
constexpr int stride = 1, pad = 0;

// In this cpp, array follow the order [no][channel][height][weight]

// original array size
constexpr int P = (H - K + 2 * pad) / stride + 1;
constexpr int Q = (W - K + 2 * pad) / stride + 1;
int IA[IC][H][W];
int WT[OC][IC][K][K];
double OA[OC][P + 5][Q + 5];
double GT[OC][P][Q]; // result of ordinary conv

// im2col array size
constexpr int NW = K * K * IC; // concat IC channels to a same row
constexpr int NH = P * Q;
int Trans_IA[NH][NW]; // NH * NW
int Trans_W[OC][NW];  // OC * NW * 1
constexpr int tile_size = (H - 2) / 2;

// used for winograd F(2,3)
double G[4][3] = {{1, 0, 0},
                  {0.5, 0.5, 0.5},
                  {0.5, -0.5, 0.5},
                  {0, 0, 1}};
double Gt[3][4] = {{1, 0.5, 0.5, 0},
                   {0, 0.5, -0.5, 0},
                   {0, 0.5, 0.5, 1}};
double GgGt[OC * IC][4][4];

double Bt[4][4] = {{1, 0, -1, 0},
                   {0, 1, 1, 0},
                   {0, -1, 1, 0},
                   {0, 1, 0, -1}};
double B[4][4] = {{1, 0, 0, 0},
                  {0, 1, -1, 1},
                  {-1, 1, 1, 0},
                  {0, 0, 0, -1}};
double BtdB[tile_size * tile_size][4][4];

double At[2][4] = {{1, 1, 1, 0},
                   {0, 1, -1, -1}};
double A[4][2] = {{1, 0},
                  {1, 1},
                  {1, -1},
                  {0, -1}};

// origin array init
void init()
{
  srand((unsigned)time(NULL));
  for (int i = 0; i < IC; ++i)
    for (int j = 0; j < H; ++j)
      for (int k = 0; k < W; ++k)
        IA[i][j][k] = rand();
  for (int i = 0; i < OC; ++i)
    for (int j = 0; j < IC; ++j)
      for (int k = 0; k < K; ++k)
        for (int l = 0; l < K; ++l)
          WT[i][j][k][l] = rand();
  memset(GT, 0, sizeof(GT));
  memset(OA, 0, sizeof(OA));
  // calculate ordinary conv result
  for (int n = 0; n < OC; ++n)
    for (int c = 0; c < IC; ++c)
      for (int i = 0; i < P; ++i)
        for (int j = 0; j < Q; ++j)
        {
          for (int di = 0; di < K; ++di)
            for (int dj = 0; dj < K; ++dj)
              GT[n][i][j] += (IA[c][i + di][j + dj] * WT[n][c][di][dj]);
        }
}

void normal_cal()
{
  // calculate ordinary conv result
  for (int n = 0; n < OC; ++n)
    for (int c = 0; c < IC; ++c)
      for (int i = 0; i < P; ++i)
        for (int j = 0; j < Q; ++j)
        {
          for (int di = 0; di < K; ++di)
            for (int dj = 0; dj < K; ++dj)
              OA[n][i][j] += (IA[c][i + di][j + dj] * WT[n][c][di][dj]);
        }
}

void im2col()
{

  // flat Kernel array
  for (int n = 0; n < OC; ++n)
    for (int c = 0; c < IC; ++c)
      for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j)
          Trans_W[n][c * K * K + i * K + j] = WT[n][c][i][j];

  // flat input array
  for (int c = 0; c < IC; ++c)
    for (int i = 0; i < P; ++i)
      for (int j = 0; j < Q; ++j)
        for (int di = 0; di < K; ++di)
          for (int dj = 0; dj < K; ++dj)
            Trans_IA[i * Q + j][c * K * K + di * K + dj] = IA[c][i + di][j + dj];
  // calculate by im2col
  for (int n = 0; n < OC; ++n)
    for (int i = 0; i < P; ++i)
      for (int j = 0; j < Q; ++j)
      {
        for (int w = 0; w < NW; ++w)
          OA[n][i][j] += Trans_IA[i * Q + j][w] * Trans_W[n][w];
      }
}
void check()
{
  for (int n = 0; n < OC; ++n)
    for (int i = 0; i < P; ++i)
      for (int j = 0; j < Q; ++j)
        assert(OA[n][i][j] == GT[n][i][j]);
}

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

void wino_kernel()
{
  memset(GgGt, 0, sizeof(GgGt));
  for (int i = 0; i < OC; ++i)
    for (int j = 0; j < IC; ++j)
    {
      double temp[4][3] = {{0}};
      for (int k = 0; k < 4; ++k)
        for (int l = 0; l < 3; ++l)
          for (int m = 0; m < 3; ++m)
            temp[k][l] += G[k][m] * WT[i][j][m][l];

      for (int k = 0; k < 4; ++k)
        for (int l = 0; l < 4; ++l)
          for (int m = 0; m < 3; ++m)
            GgGt[i * IC + j][k][l] += temp[k][m] * Gt[m][l];
    }
}

void wino_input(int ic, int oc)
{

  for (int i = 0; i < tile_size; ++i)
    for (int j = 0; j < tile_size; ++j)
    {
      double temp[4][4] = {{0}};
      for (int k = 0; k < 4; ++k)
        for (int l = 0; l < 4; ++l)
          for (int m = 0; m < 4; ++m)
            temp[k][l] += Bt[k][m] * IA[ic][2 * i + m][2 * j + l];
      for (int k = 0; k < 4; ++k)
        for (int l = 0; l < 4; ++l)
        {
          for (int m = 0; m < 4; ++m)
            BtdB[i * tile_size + j][k][l] += temp[k][m] * B[m][l];
          BtdB[i * tile_size + j][k][l] *= GgGt[oc * IC + ic][k][l];
        }
    }
}

void wino_result()
{

  wino_kernel();
  for (int oc = 0; oc < OC; ++oc)
    for (int ic = 0; ic < IC; ++ic)
    {
      memset(BtdB, 0, sizeof(BtdB));
      wino_input(ic, oc);
      for (int i = 0; i < tile_size; ++i)
        for (int j = 0; j < tile_size; ++j)
        {
          double temp[4][4] = {{0}};
          for (int k = 0; k < 2; ++k)
            for (int l = 0; l < 4; ++l)
              for (int m = 0; m < 4; ++m)
                temp[k][l] += At[k][m] * BtdB[i * tile_size + j][m][l];
          for (int k = 0; k < 2; ++k)
            for (int l = 0; l < 2; ++l)
              for (int m = 0; m < 4; ++m)
                OA[oc][i * 2 + k][j * 2 + l] += temp[k][m] * A[m][l];
        }
    }
}

int main()
{
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++)
  {
    init();
    auto t = get_time();

    wino_result();
    avg_time += get_time() - t;
    check();
    printf("%f\n", get_time() - t);
  }
  printf("Avg Time for Calculation: %f\n", avg_time / 32);
  return 0;
}