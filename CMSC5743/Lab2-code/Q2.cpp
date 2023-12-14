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

constexpr int batch = 1,     // batch size (b)
    height_feature = 64,     // input height (ih)
    width_feature = 64 * 64, // input width (iw)
    in_channels = 1,         // input channels (ic)
    out_channels = 512,      // output channels (oc)
    kernel_size = 3,         // kernel size (k1, k2)
    stride = 1,              // stride
    padding = 0,             // padding

    height_out = (height_feature - kernel_size + 2 * padding) / stride + 1, // output height (oh)
    width_out = (width_feature - kernel_size + 2 * padding) / stride + 1;   // output width (ow)

int Input[batch][in_channels][height_feature][width_feature];
int Input_padding[batch][in_channels][height_feature + 2 * padding][width_feature + 2 * padding];
int Kernel[out_channels][in_channels][kernel_size][kernel_size];
double Output[batch][out_channels][height_out][width_out];
double Output_groundtruth[batch][out_channels][height_out][width_out];

void init()
{
  for (int b = 0; b < batch; b++)
    for (int ic = 0; ic < in_channels; ic++)
      for (int ih = 0; ih < height_feature; ih++)
        for (int iw = 0; iw < width_feature; iw++)
          Input[b][ic][ih][iw] = rand();
  for (int b = 0; b < batch; b++)
    for (int ic = 0; ic < in_channels; ic++)
      for (int ih = 0; ih < height_feature; ih++)
        for (int iw = 0; iw < width_feature; iw++)
          Input_padding[b][ic][padding + ih][padding + iw] = Input[b][ic][ih][iw];

  for (int oc = 0; oc < out_channels; oc++)
    for (int ic = 0; ic < in_channels; ic++)
      for (int k1 = 0; k1 < kernel_size; k1++)
        for (int k2 = 0; k2 < kernel_size; k2++)
          Kernel[oc][ic][k1][k2] = rand();

  for (int b = 0; b < batch; b++)
    for (int oc = 0; oc < out_channels; oc++)
      for (int oh = 0; oh < height_out; oh++)
        for (int ow = 0; ow < width_out; ow++)
        {
          for (int ic = 0; ic < in_channels; ic++)
            for (int k1 = 0; k1 < kernel_size; k1++)
              for (int k2 = 0; k2 < kernel_size; k2++)
                Output_groundtruth[b][oc][oh][ow] += Input_padding[b][ic][oh * stride + k1][ow * stride + k2] * Kernel[oc][ic][k1][k2];
        }
}

void test()
{
  for (int b = 0; b < batch; b++)
    for (int oc = 0; oc < out_channels; oc++)
      for (int oh = 0; oh < height_out; oh++)
        for (int ow = 0; ow < width_out; ow++)
          assert(Output[b][oc][oh][ow] == Output_groundtruth[b][oc][oh][ow]);
}

int Input_col[batch][height_out * width_out][kernel_size * kernel_size * in_channels];
int Kernel_col[out_channels][kernel_size * kernel_size * in_channels];
int Output_col[batch][out_channels][height_out * width_out];

void im2col()
{
  for (int b = 0; b < batch; b++)
    for (int oh = 0; oh < height_out; oh++)
      for (int ow = 0; ow < width_out; ow++)
      {
        for (int ic = 0; ic < in_channels; ic++)
          for (int k1 = 0; k1 < kernel_size; k1++)
            for (int k2 = 0; k2 < kernel_size; k2++)
              Input_col[b][oh * width_out + ow][ic * kernel_size * kernel_size + k1 * kernel_size + k2] =
                  Input_padding[b][ic][oh * stride + k1][ow * stride + k2];
      }

  for (int oc = 0; oc < out_channels; oc++)
    for (int ic = 0; ic < in_channels; ic++)
      for (int k1 = 0; k1 < kernel_size; k1++)
        for (int k2 = 0; k2 < kernel_size; k2++)
          Kernel_col[oc][ic * kernel_size * kernel_size + k1 * kernel_size + k2] = Kernel[oc][ic][k1][k2];
}

void col2im()
{
  for (int b = 0; b < batch; b++)
    for (int oc = 0; oc < out_channels; oc++)
      for (int oh = 0; oh < height_out; oh++)
        for (int ow = 0; ow < width_out; ow++)
          Output[b][oc][oh][ow] = Output_col[b][oc][oh * width_out + ow];
}

void matmul_im2col()
{
  memset(Output, 0, sizeof(Output));
  memset(Output_col, 0, sizeof(Output_col));

  im2col();
  for (int b = 0; b < batch; b++)
    for (int oc = 0; oc < out_channels; oc++)
      for (int ch = 0; ch < height_out * width_out; ch++)                    // col height (ch)
        for (int cw = 0; cw < kernel_size * kernel_size * in_channels; cw++) // col width (cw)
          Output_col[b][oc][ch] += Input_col[b][ch][cw] * Kernel_col[oc][cw];
  col2im();
}

constexpr int m = 2, r = 3,
              tile_size = m + r - 1,
              height_count = (height_feature - tile_size + 2 * padding) / m + 1, // height count for tile (hc)
    width_count = (width_feature - tile_size + 2 * padding) / m + 1;             // width count for tile (wc)

int BT[tile_size][tile_size] = {{1, 0, -1, 0},
                                {0, 1, 1, 0},
                                {0, -1, 1, 0},
                                {0, 1, 0, -1}};
double G[tile_size][kernel_size] = {{1, 0, 0},
                                    {0.5, 0.5, 0.5},
                                    {0.5, -0.5, 0.5},
                                    {0, 0, 1}};
int AT[m][tile_size] = {{1, 1, 1, 0},
                        {0, 1, -1, -1}};
double GgGT[tile_size][tile_size];
int BTdB[tile_size][tile_size];

void input_Winograd(int b, int ic, int hc, int wc)
{
  memset(BTdB, 0, sizeof(BTdB));
  // BT * d
  int temp[tile_size][tile_size] = {0};
  for (int i = 0; i < tile_size; i++)
    for (int j = 0; j < tile_size; j++)
      for (int k = 0; k < tile_size; k++)
        temp[i][j] += BT[i][k] * Input_padding[b][ic][hc * 2 + k][wc * 2 + j];
  // BTd * B
  for (int i = 0; i < tile_size; i++)
    for (int j = 0; j < tile_size; j++)
      for (int k = 0; k < tile_size; k++)
        BTdB[i][j] += temp[i][k] * BT[j][k]; // B[k][j] = B_T[j][k]
}

void kernel_Winograd(int oc, int ic)
{
  memset(GgGT, 0, sizeof(GgGT));
  // G * g
  double temp[tile_size][kernel_size] = {0};
  for (int i = 0; i < tile_size; i++)
    for (int j = 0; j < kernel_size; j++)
      for (int k = 0; k < kernel_size; k++)
        temp[i][j] += G[i][k] * Kernel[oc][ic][k][j];
  // Gg * GT
  for (int i = 0; i < tile_size; i++)
    for (int j = 0; j < tile_size; j++)
      for (int k = 0; k < kernel_size; k++)
        GgGT[i][j] += temp[i][k] * G[j][k]; // GT[k][j] = G[j][k]
}

void matmul_Winograd()
{
  memset(Output, 0, sizeof(Output));
  for (int b = 0; b < batch; b++)
    for (int oc = 0; oc < out_channels; oc++)
      for (int ic = 0; ic < in_channels; ic++)
      {
        kernel_Winograd(oc, ic);
        for (int hc = 0; hc < height_count; hc++)
          for (int wc = 0; wc < width_count; wc++)
          {
            input_Winograd(b, ic, hc, wc);
            // GgGT * BTdB
            double temp1[tile_size][tile_size];
            for (int i = 0; i < tile_size; i++)
              for (int j = 0; j < tile_size; j++)
                temp1[i][j] = GgGT[i][j] * BTdB[i][j];
            // AT * GgGTBTdB
            double temp2[m][tile_size] = {0};
            for (int i = 0; i < m; i++)
              for (int j = 0; j < tile_size; j++)
                for (int k = 0; k < tile_size; k++)
                  temp2[i][j] += AT[i][k] * temp1[k][j];
            // ATGgGTBTdB * A
            for (int i = 0; i < m; i++)
              for (int j = 0; j < m; j++)
                for (int k = 0; k < tile_size; k++)
                  Output[b][oc][hc * 2 + i][wc * 2 + j] += temp2[i][k] * AT[j][k]; // A[k][j] = AT[j][k]
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
    // matmul_im2col();
    matmul_Winograd();
    test();
    printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Calculation: %f\n", avg_time / 32);
  return 0;
}
